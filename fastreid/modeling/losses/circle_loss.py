# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

__all__ = ["pairwise_circleloss", "pairwise_cosface"]


def pairwise_circleloss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss


def pairwise_cosface(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    # Normalize embedding features
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
    logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    return loss


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

# class AdaFace(Module):
#     def __init__(self,
#                  embedding_size=512,
#                  classnum=70722,
#                  m=0.4,
#                  h=0.333,
#                  s=64.,
#                  t_alpha=1.0,
#                  ):
#         super(AdaFace, self).__init__()
#         self.classnum = classnum
#         self.kernel = Parameter(torch.Tensor((embedding_size),(classnum)))

#         # initial kernel
#         self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m 
#         self.eps = 1e-3
#         self.h = h
#         self.s = s

#         # ema prep
#         self.t_alpha = t_alpha
#         self.register_buffer('t', torch.zeros(1))
#         self.register_buffer('batch_mean', torch.ones(1)*(20))
#         self.register_buffer('batch_std', torch.ones(1)*100)

#         # print('\n\AdaFace with the following property')
#         # print('self.m', self.m)
#         # print('self.h', self.h)
#         # print('self.s', self.s)
#         # print('self.t_alpha', self.t_alpha)

#     def forward(self, embbedings, label):
#         norms = torch.norm(embbedings, 2, -1, keepdim=True)
#         normalized_embedding =  embbedings / norms

#         kernel_norm = l2_norm(self.kernel,axis=0)
#         cosine = torch.mm(normalized_embedding,kernel_norm)
#         cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

#         safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
#         safe_norms = safe_norms.clone().detach()

#         # update batchmean batchstd
#         with torch.no_grad():
#             mean = safe_norms.mean().detach()
#             std = safe_norms.std().detach()
#             self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
#             self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

#         margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
#         margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
#         margin_scaler = torch.clip(margin_scaler, -1, 1)
#         # ex: m=0.5, h:0.333
#         # range
#         #       (66% range)
#         #   -1 -0.333  0.333   1  (margin_scaler)
#         # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

#         # g_angular
#         m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
#         m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
#         g_angular = self.m * margin_scaler * -1
#         m_arc = m_arc * g_angular
#         theta = cosine.acos()
#         theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
#         cosine = theta_m.cos()

#         # g_additive
#         m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
#         m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
#         g_add = self.m + (self.m * margin_scaler)
#         m_cos = m_cos * g_add
#         cosine = cosine - m_cos

#         # scale
#         scaled_cosine_m = cosine * self.s
#         return scaled_cosine_m