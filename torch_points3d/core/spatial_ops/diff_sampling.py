import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps
from pcpnet.new_api import net as pcp_net
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

class SampleLayer(torch.nn.Module):
    def __init__(self):
        super(SampleLayer, self).__init__()
        self.sample_layer = torch.nn.Linear(3, 1)
    
    def estimate_curv_normals(self, inputpc):
        # need to call pcpnet
        normals, curv = pcp_net.get_opt(inputpc.transpose(1, 2)[..., :3].detach().cpu().squeeze())
        normals = normals[None, ...].cuda()
        curv = curv[...,  0].cuda()
        curv = curv[None, :, None]


    def forward(self, x, exchange_percent=0.1, k=10):
        
        idxs = knn(x.transpose(1, 2), k)
        xn = get_edge_features(x.permute(0,2,1)[:,:, None, :], idxs).permute(0, 3, 2, 1)
        # idxs = fps(x, 1)
        idxs = tp.furthest_point_sample(x, self.num_to_sample)
        _, curv = self.estimate_curv_normals(x)
        if len(curv.shape) > 2:
            curv = curv.squeeze(1)
        
        curv_feat = abs((curv - curv.mean(1)[..., None])/curv.sum(1)[..., None])
        curv = curv - (curv.min(dim=1).values)[..., None]
        curv = curv / (curv.max(dim=1).values)[..., None]
        curv_feat = torch.gather(curv_feat, -1, idxs.long())# curv features arranged according to furthest_points
        # these are the ranks, convert into soft rank
        scores = torch.linspace(1, 0, steps=x.shape[1]).repeat(x.shape[0], 1).to(x.device)
        scores = torch.gather(scores, -1, idxs.long())
        
        scores_curvs = scores * curv
        scores_curvs_a = scores_curvs[..., :self.num_to_sample]
        scores_curvs_b = scores_curvs[..., self.num_to_sample:]
        idxs_a = idxs[..., :self.num_to_sample]
        # idxs_b = idxs[..., self.num_to_sample:]
        total_pnts = scores_curvs.shape[-1]

        exchange_points = int(exchange_percent * self.num_to_sample)
        # take topk items from each and exchange them
        # arrange as follows - bottom_a[max element, ... , min]
        #                      top_b   [min element, ... , max]
        bottom_a = torch.topk(scores_curvs_a, self.num_to_sample, dim=1)
        top_b = torch.topk(-scores_curvs_b, scores_curvs_b.shape[-1], dim=1)
        # print(f'num points : {self.num_to_sample}')
        top_b_vals = -1*top_b.values[..., -self.num_to_sample:]
        top_b_indices = top_b.indices[..., -self.num_to_sample:]
        top_b_indices = top_b_indices + self.num_to_sample # now the comparison is on global index scale
        
        min_size = min(top_b_vals.shape[-1], bottom_a.values.shape[-1])
        # bottom a -> 1536
        bottom_a_vals = bottom_a.values[..., -min_size:]
        # print('bottom_a_val size:', bottom_a_vals.shape) % [1, 512]
        mask = top_b_vals > bottom_a_vals
        good_a_idxs = torch.where(~mask, bottom_a.indices[..., -min_size:], -torch.ones_like(bottom_a.indices[..., -min_size:]))
        # print('a :', good_a_idxs.shape) # [1, 512]
        # print('bottom a chunk size:', bottom_a.indices[..., :-min_size].shape) # [1, 1024]
        good_a_idxs = torch.cat([bottom_a.indices[..., :-min_size], good_a_idxs], -1)
        # print('a :', good_a_idxs.shape)
        good_b_idxs = torch.where(mask, top_b_indices, -torch.ones_like(top_b_indices))
        final_indices = torch.cat([good_a_idxs[..., :- exchange_points], 
        good_b_idxs[..., -exchange_points:]], -1)
        final_indices = torch.where(final_indices != -1, final_indices, bottom_a.indices)
        # visualize the final_indices that got selected

        return final_indices
