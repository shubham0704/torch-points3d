import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import fps
import torch_points_kernels as tp
import pdb

def get_edge_features(x, idx):
    batch_size, num_points, k = idx.size() # [1, 3072, 16]
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.squeeze(2) # B C 1 N [1, 3, 1, 3072]-> B C N
    _, num_dims, _ = x.size() # 1 3 3072
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 2, 1)  # B, C, K, N
    return feature

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def grouping_operation(features, idx):
    r"""
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with
    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])



class ConditionalFPS(torch.nn.Module):
    def __init__(self, num_to_sample):
        super(ConditionalFPS, self).__init__()
        self.sample_layer = torch.nn.Linear(3, 1)
        self.num_to_sample = num_to_sample
    
    def estimate_normals_torch(self, inputpc, x):
        temp = x[..., :3] - x.mean(dim=2)[:, :, None, :3]    
        cov = temp.transpose(2, 3) @ temp / x.shape[0]
        e, v = torch.symeig(cov, eigenvectors=True)
        n = v[..., 0]
        return torch.cat([inputpc, n], dim=-1)

    def curvature(self, pc, xn):
        import math
        INNER_PRODUCT_THRESHOLD = math.pi / 2
        inner_prod = (xn * pc[:, :, None, 3:]).sum(dim=-1)
        inner_prod[inner_prod > 1] = 1
        inner_prod[inner_prod < -1] = -1
        angle = torch.acos(inner_prod)
        angle[angle > INNER_PRODUCT_THRESHOLD] = math.pi - angle[angle > INNER_PRODUCT_THRESHOLD]
        angle = angle.sum(dim=-1)
        angle = (angle - angle.mean())/angle.sum()
        return angle
    
    def density(self, pc: torch.Tensor, knn_data):
        k = knn_data.shape[2]
        max_distance, _ = (knn_data[:, :, :, :3] - pc[:, :, None, :3]).norm(dim=-1).max(dim=-1)
        dense = k / (max_distance ** 3)
        inf_mask = torch.isinf(dense)
        max_val = dense[~inf_mask].max()
        dense[inf_mask] = max_val
        dense = (dense - dense.mean())/dense.sum()
        return dense

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    def forward(self, x, pos):
        prefix="train"
        k=10
        # pos means x,y,z points
        # x means normals
        x = x.transpose(1, 2)
        # ratio = int(self.num_to_sample/pos.shape[1])
        # num_points = int(x.shape[1]*ratio)
        # x -> [1, 2048, 3]
        # feat -> [1, 64, 2048]
        # p_idx = fps(pos, ratio=ratio)
        p_idx = tp.furthest_point_sample(pos, self.num_to_sample)
        fps_feature = torch.zeros(*pos.shape[:2]).to(x.device) # [16, 2048]
        fps_feature  = fps_feature.scatter_(1, p_idx.long(), 1)
        fps_feature =  (fps_feature - fps_feature.mean())/fps_feature.sum()

        idxs = knn(pos.transpose(1, 2), k)
        xn = get_edge_features(pos.permute(0,2,1)[:,:, None, :], idxs).permute(0, 3, 2, 1)
        if x is None:
            pc_with_normals = self.estimate_normals_torch(pos, xn) 
            # Note: normals estimation can become performance bottleneck. Do once for whole dataset and keep
        else:
            # pdb.set_trace()
            pc_with_normals = torch.cat([pos, x], dim=-1) # [16, 2048, 6]
        curv = self.curvature(pc_with_normals, xn)
        dense = self.density(pc_with_normals, xn)
        # can add more features in future
        sampling_feats = torch.cat([fps_feature[..., None], curv[..., None], dense[..., None]], dim=-1)
        opt = self.sample_layer(sampling_feats).squeeze(2)
        smax= torch.nn.Softmax(dim=1)(opt)
        topk = torch.topk(smax, self.num_to_sample, dim=1)
        # bottomk = torch.topk(smax, smax.shape[1] - num_points, largest=False,dim=1)
        
        if prefix == "train":
            point_output = grouping_operation(pos.transpose(1,2).contiguous(), 
                            topk.indices.unsqueeze(1)).squeeze().transpose(1,2).contiguous()
            
            nbrs = torch.gather(xn, 1, topk.indices[..., None, None].repeat(1, 1, xn.shape[2], xn.shape[3]))
            # get distances
            shorlisted_dists = torch.norm(point_output[..., None, :].repeat(1,1,xn.shape[2], 1) - nbrs, dim=-1)
            shorlisted_dists_loss = shorlisted_dists.max(dim=-1).values + shorlisted_dists.mean(dim=-1)
            # TASK - select point with nearest distance from neighborhood and outlier
            # FPS - select farthest point subset
            # FPS + curvature + density - select point based on 3 features, leart a function
            diff = pos[..., None, :].repeat(1,1,xn.shape[2], 1) - xn
            dists = torch.norm(diff, dim=-1) #* smax[...,None].repeat(1, 1, diff.shape[2])
            dist_loss = dists.max(dim=-1).values + dists.mean(dim=-1)
            # what about distances of bottom k that should be penalised as well
            # get distances and softmax values of all points and that is your loss
            sampling_loss = dist_loss * smax
            total_loss = sampling_loss.mean() #+ dist_loss.mean()
            # check if loss.backward() updates any layer in sampling operation
            baseline_nbrs = torch.gather(xn, 1, p_idx[..., None, None].repeat(1, 1, xn.shape[2], xn.shape[3]).long())
            baseline_pnts = grouping_operation(pos.transpose(1,2).contiguous(), 
                            p_idx.unsqueeze(1)).squeeze().transpose(1,2).contiguous()
            baseline_dists = torch.norm(baseline_pnts[..., None, :].repeat(1,1,xn.shape[2], 1) - baseline_nbrs, dim=-1)
            bdist_loss = baseline_dists.max(dim=-1).values + baseline_dists.mean(dim=-1)
            losses = [total_loss, sampling_loss.mean(), shorlisted_dists_loss.mean(), bdist_loss.mean()]
        else:
            losses = None
        
        return topk.indices, losses
