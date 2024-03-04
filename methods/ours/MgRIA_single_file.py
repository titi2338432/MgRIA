import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as tDataset
import matplotlib.pyplot as plt
import random
import time
from numpy import mean
import datetime


'''
Embedding layer
'''


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, device):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg['vocab_size'], cfg['dim'])  # token embedding
        self.time_embed = TemporalEmbedding(cfg['dim'])
        self.time_interval_embed_K = nn.Embedding(cfg['time_span'] + 1, cfg['dim'])
        self.time_interval_embed_V = nn.Embedding(cfg['time_span'] + 1, cfg['dim'])
        # self.pos_embed = nn.Embedding(cfg['max_len']+1, cfg['dim']) # position embedding
        # self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg['p_drop_hidden'])
        self.time_matrix_K_dropout = nn.Dropout(cfg['p_drop_hidden'])
        self.time_matrix_V_dropout = nn.Dropout(cfg['p_drop_hidden'])
        self.device = device
        # self.batch_size = cfg['batch_size']
        # self.dim = cfg['dim']
        # self.ab_time = cfg['ab_time']
        # self.position = cfg['position']

    import pysnooper
    # @pysnooper.snoop()
    def forward(self, x, stamp, time_matrix):
        seq_len = x.size(1)
        # if self.position:
        #    pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        #    pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
        # print(pos.shape)
        # x += self.pos_embed(pos)
        # if self.ab_time:
        t_e = self.time_embed(stamp)    # torch.Size([128, 11, 3]) -> torch.Size([128, 12, 35])
        zero = torch.zeros(t_e.size()[0], 1, t_e.size()[2]).to(self.device) # torch.Size([128, 1, 35])
        t_e = torch.cat((zero, t_e), 1)
        # print(t_e.size(),t_e[0][0])
        x = self.tok_embed(x) + t_e # x.shape = torch.Size([128, 12])
        # else:
        #    x = self.tok_embed(x)
        # if self.position:
        #    x += self.pos_embed(pos)
        # x = self.tok_embed(x) + self.pos_embed(pos)# + self.seg_embed(seg)
        t_interval_K = self.time_interval_embed_K(time_matrix) #B,U,L,100
        t_interval_V = self.time_interval_embed_V(time_matrix)
        t_interval_K = self.time_matrix_K_dropout(t_interval_K)  # 在emb层经历dropout
        t_interval_V = self.time_matrix_V_dropout(t_interval_V)
        # print(t_interval_K.shape)#[B,S+1,S+1,D]
        return self.drop(self.norm(x)), t_interval_K, t_interval_V


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        # minute_size = 4;
        # hour_size = 25
        weekday_size = 8
        day_size = 32
        month_size = 13

        # Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        Embed = nn.Embedding
        # if freq=='t':
        #    self.minute_embed = Embed(minute_size, d_model)
        # self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        # month_x = self.month_embed(x[:,:])

        # return hour_x + weekday_x + day_x + month_x
        return month_x + day_x  # + weekday_x

'''
Transformer layer
'''
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def gelu(x):#无需改
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):#无需改
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg['dim']))
        self.beta  = nn.Parameter(torch.zeros(cfg['dim']))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class MultiHeadedSelfAttention(nn.Module):  
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg,device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], cfg['dim'])
        self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.scores = None  # for visualization
        self.n_heads = cfg['n_heads']
        self.device = device

    def forward(self, x, t_K, t_V, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # time_matrix#[B,S,S,D]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, S, S, D) -split-> (B, S, S, H, W) -trans-> (B, S, H, S, W) -trans-> (B, H, S, S, W)
        t_K, t_V = (split_last(x, (self.n_heads, -1)).transpose(-3, -2).transpose(1, 2)
                    for x in [t_K, t_V])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1)  # / np.sqrt(k.size(-1))
        # attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        scores += (t_K @ q.unsqueeze(-1)).squeeze(-1)
        scores /= np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float().to(self.device)
            scores += 10000.0 * (1.0 - mask)  # 因为mask用的id是257，所以后半部分是负数
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # (B, H, S, S) @ (B, H, S, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        # (B, H, S, S) ->  (B, H, 1, S, S) @ (B, H, S, S, W) -> (B, H, S, S, W) diag-> (B, H, W, S) -trans-> (B, H, S, W)-trans-> (B, S, H, W)
        h_t = scores.unsqueeze(2).matmul(t_V)
        h_t = torch.diagonal(h_t, offset=0, dim1=2, dim2=3)
        h_t = h_t.transpose(-1, -2)
        h_t = h_t.transpose(1, 2)
        h += h_t
        # outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['dim'], cfg['dim_ff'])
        self.fc2 = nn.Linear(cfg['dim_ff'], cfg['dim'])

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, cfg,device):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg,device)
        self.proj = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg['p_drop_hidden'])

    def forward(self, x, t_K, t_V, mask):
        h = self.attn(x, t_K, t_V, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg,device):
        super().__init__()
        self.embed = Embeddings(cfg,device)
        self.blocks = nn.ModuleList([Block(cfg,device) for _ in range(cfg['n_layers'])])

    def forward(self, x, stamp, mask, time_matrix):
        h, time_matrix_K, time_matrix_V = self.embed(x, stamp, time_matrix)
        for block in self.blocks:
            h = block(h, time_matrix_K, time_matrix_V, mask)
        return h  # 每个token的embedding

'''
User repurchasing characteristics layer
'''


class REmotion_add(nn.Module):  # 无需改
    """Additive Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.W_rec = nn.Linear(cfg['dim'], 2)  # repeat和explore俩motion [0.9,0.5]
        self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.activation = nn.Tanh()
        self.device = device

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """

        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))  # 用各个token的query来和interest的key计算相似度 -1
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])  # [B, S, D]
        # q,k = x,x[:,0,:].unsqueeze(dim=1).expand(x.shape[0],x.shape[1],x.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        # print(scores.shape) [1,1],[1] [[3]] [3]
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)
        scores = scores.squeeze(dim=-1)  # (B, S)
        # softmax
        scores = F.softmax(scores, dim=-1)  # (B, S),softmax后得到每个token和interest的匹配度
        scores = scores.unsqueeze(dim=-1)  # (B, S, 1)
        c_s = torch.sum(x * scores, dim=1)  # x*scores后(B, S, D)，(B, 1, D)对S维度求和后(B, D)
        # softmax
        # pre = F.softmax(self.W_rec(c_s), dim=-1)#W后(B, 2),softmax后(B, 2)

        pre = self.W_rec(c_s)
        # pre = self.W_rec(x[:,0,:])
        return pre


def build_map(b_map, max=None):  # 感觉是变成onehot
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)  # b,s,max
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad = False
    return b_map_


class Repeat_decoder_add(nn.Module):  # 无需改
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.activation = nn.Tanh()
        # self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.vocab_size = cfg['vocab_size']
        self.proj = cfg['repeat_proj']
        self.interest_id = cfg['interest_id']
        self.pad_id = cfg['pad_id']

        # weight
        self.fs0 = torch.nn.Parameter(torch.tensor(1.0))
        self.fs1_1 = torch.nn.Parameter(torch.tensor(1.0))
        self.fs1_2 = torch.nn.Parameter(torch.tensor(1.0))
        self.fs2_1 = torch.nn.Parameter(torch.tensor(1.0))
        self.fs2_2 = torch.nn.Parameter(torch.tensor(1.0))
        self.fs3 = torch.nn.Parameter(torch.tensor(1.0))
        self.w_p_g = torch.nn.Parameter(torch.tensor(0.5))  # 初始权重
        # Gauss
        self.mu0 = torch.nn.Parameter(torch.tensor(30.))
        self.sigma0 = torch.nn.Parameter(torch.tensor(1.0))
        self.mu1_1 = torch.nn.Parameter(torch.tensor(30.0))
        self.sigma1_1 = torch.nn.Parameter(torch.tensor(1.0))
        self.mu1_2 = torch.nn.Parameter(torch.tensor(60.0))
        self.sigma1_2 = torch.nn.Parameter(torch.tensor(1.0))
        self.mu2 = torch.nn.Parameter(torch.tensor(30.))
        self.sigma2 = torch.nn.Parameter(torch.tensor(1.0))
        # power law
        self.p2 = torch.nn.Parameter(torch.tensor(-0.5))
        self.p3 = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, x_ids, time_gap, cla, tr_te):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        x_ids:(B,S)
        """
        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        # q,k = self.proj_q(x),self.proj_k(x[:,0,:].unsqueeze(dim=1))#用各个token的query来和interest的key计算相似度
        # k=k.expand(q.shape[0],q.shape[1],q.shape[2])#[B, S, D]
        mask0 = x_ids.ne(self.pad_id)  # 加了mask，为了少一个softmax,其实也应该把interest给mask掉，不分softmax的权重
        mask1 = x_ids.ne(self.interest_id)
        mask = mask0 * mask1  # 或关系
        # mask_ = mask.unsqueeze(1)
        input_onehot = build_map(x_ids, max=self.vocab_size)
        if self.proj:
            q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))
        else:
            q, k = x, x[:, 0, :].unsqueeze(dim=1)  # .expand(x.shape[0],x.shape[1],x.shape[2])
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        scores = scores.squeeze(dim=-1)  # (B, S)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)/ np.sqrt(k.size(-1))#(B,S,S)

        if mask is not None:  # new
            scores = scores.masked_fill(~mask, -float('inf'))  # new
        scores = F.softmax(scores, dim=-1)  # new
        p_repeat = torch.bmm(scores.unsqueeze(1), input_onehot).squeeze(1)

        '''softmax'''
        # output = F.softmax(output,dim=-1)
        # p_repeat = F.softmax(p_repeat,dim = -1)
        # dis
        x_ids_ = x_ids[:, 1:]
        input_onehot = build_map(x_ids_, max=self.vocab_size)  # 看看有没问题
        clapos_0, clapos_1, clapos_2, clapos_3 = cla.ne(0), cla.ne(1), cla.ne(2), cla.ne(3)
        # 假设只有0、1、2三个类别显著，则在将除0外的类别都掩码时，3、4类别在应被随机掩码，而1、2类别应被赋予大的值。
        if tr_te == 'train':
            # pad0,pad1,pad2,pad3 = random.randint(1,90),random.randint(1,90),random.randint(1,90),random.randint(1,90)
            pad0, pad1, pad2, pad3 = 180, 180, 180, 180
        elif tr_te == 'test':
            pad0, pad1, pad2, pad3 = 180, 180, 180, 180
        cla_0, cla_1, cla_2, cla_3 = time_gap.masked_fill(clapos_0, pad0), time_gap.masked_fill(clapos_1,
                                                                                                pad1), time_gap.masked_fill(
            clapos_2, pad2), time_gap.masked_fill(clapos_3, pad3)
        p_gap_0, p_gap_1, p_gap_2 = self.Distri(cla_0, 0).unsqueeze(1), self.Distri(cla_1, 1).unsqueeze(1), self.Distri(
            cla_2, 2).unsqueeze(1)
        p_gap_0, p_gap_1, p_gap_2 = torch.bmm(p_gap_0, input_onehot).squeeze(1), torch.bmm(p_gap_1, input_onehot).squeeze(
            1), torch.bmm(p_gap_2, input_onehot).squeeze(1)
        # p_gap = self.Distri(time_gap).unsqueeze(1)#[B,S]
        # p_gap = torch.bmm(p_gap, input_onehot).squeeze(1)
        p_repeat_ = (1 - self.w_p_g) * p_repeat + self.w_p_g * (p_gap_0 + p_gap_1 + p_gap_2)  # [B,vocab]
        return p_repeat_  # output

    def Distri(self, x, mode):
        if mode == 0:  # 1G
            G1 = torch.distributions.Normal(loc=self.mu0, scale=self.sigma0)
            score = self.fs0 * G1.log_prob(x).exp()
        if mode == 1:  # 2G
            G1 = torch.distributions.Normal(loc=self.mu1_1, scale=self.sigma1_1)
            G2 = torch.distributions.Normal(loc=self.mu1_2, scale=self.sigma1_2)
            score = self.fs1_1 * G1.log_prob(x).exp() + self.fs1_2 * G2.log_prob(x).exp()
        if mode == 2:  # G+m
            G1 = torch.distributions.Normal(loc=self.mu2, scale=self.sigma2)
            P = torch.pow(x, self.p2)
            score = self.fs2_1 * G1.log_prob(x).exp() + self.fs2_2 * P
        if mode == 3:  # m
            P = torch.pow(x, self.p3)
            score = self.fs3 * P
        # score = self.fs1*G1.log_prob(x).exp()+self.fs2*G2.log_prob(x).exp()+self.fs3*P
        return score

    def get_par(self):
        # total
        # par = [float(self.fs0), float(self.fs1_1), float(self.fs1_2), float(self.fs2_1), float(self.fs2_2),
        #        float(self.fs3),
        #        float(self.w_p_g), float(self.mu0), float(self.sigma0), float(self.mu1_1), float(self.sigma1_1),
        #        float(self.mu1_2),
        #        float(self.sigma1_2), float(self.mu2), float(self.sigma2), float(self.p2), float(self.p3)]
        par = [float(getattr(self, attr)) for attr in ['fs0', 'fs1_1', 'fs1_2', 'fs2_1', 'fs2_2', 'fs3', 'w_p_g',
                    'mu0', 'sigma0', 'mu1_1', 'sigma1_1', 'mu1_2', 'sigma1_2',
                    'mu2', 'sigma2', 'p2', 'p3']]
        return par


class Explore_decoder_add(nn.Module):  # 无需改
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.vocab_size = cfg['vocab_size']
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.activation = nn.Tanh()
        self.W_ec = nn.Linear(2 * cfg['dim'], self.vocab_size)  # explore这个motion根据拼接后的向量，输出一个得分分布
        # self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.proj = cfg['explore_proj']
        self.interest_id = cfg['interest_id']
        self.pad_id = cfg['pad_id']

    def forward(self, x, x_ids):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        # q,k = self.proj_q(x),self.proj_k(x[:,0,:].unsqueeze(dim=1))#用各个token的query来和interest的key计算相似度
        # k=k.expand(q.shape[0],q.shape[1],q.shape[2])#[B, S, D]
        mask0 = x_ids.ne(self.pad_id)  # 加了mask，为了少一个softmax,其实也应该把interest给mask掉，不分softmax的权重
        mask1 = x_ids.ne(self.interest_id)
        mask = mask0 * mask1  # 或关系，[B,S]
        # mask = x_ids.ne(259)#全为true
        # mask_ = mask.unsqueeze(1)#不知需不需要专门扩
        input_onehot = build_map(x_ids, max=self.vocab_size)
        if self.proj:
            q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))
        else:
            q, k = x, x[:, 0, :].unsqueeze(dim=1)  # .expand(x.shape[0],x.shape[1],x.shape[2])
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        scores = scores.squeeze(dim=-1)  # (B, S)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)/ np.sqrt(k.size(-1))

        '''softmax'''
        # if mask is not None:#new
        #    scores = scores.masked_fill(~mask, -float('inf'))#new
        scores = F.softmax(scores, dim=-1)  # (B, S),softmax后得到每个token和interest的匹配度
        scores = scores.unsqueeze(dim=-1)  # (B, S, 1)
        c_s = torch.sum(x * scores, dim=1)  # x*scores后(B, S, D)，对S维度求和后(B, D)
        h_t = x[:, 0, :].squeeze(dim=1)  # (B, D)
        c_s = torch.cat((h_t, c_s), -1)  # (B,D+D)
        p_explore = self.W_ec(c_s)  # (B, vocab)
        explore_mask = torch.bmm(mask.float().unsqueeze(1), input_onehot).squeeze(1)  # 代表非padding部分
        '''softmax'''
        p_explore = p_explore.masked_fill(explore_mask.bool(), float('-inf'))
        p_explore = F.softmax(p_explore, dim=-1)
        return p_explore

'''
MgRIA & Output layer
'''


class BertRepeatModel(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"

    def __init__(self, cfg, device):
        super().__init__()
        self.transformer = Transformer(cfg,device)
        self.REmechanism = REmotion_add(cfg,device)
        self.Rdecoder = Repeat_decoder_add(cfg,device)
        self.Edecoder = Explore_decoder_add(cfg,device)
        '''
        self.fc = nn.Linear(cfg['dim'], cfg['dim'])
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm = LayerNorm(cfg)
        n_vocab, n_dim = cfg['vocab_size'],cfg['dim']
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        '''
        self.linear = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg['dim'], cfg['vocab_size'], bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg['vocab_size']))
        self.rt = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, tr_te):
        # print(masked_pos.shape)
        h = self.transformer(input_ids, stamp, input_mask, time_matrix)  # [batch,seq_len,dim],[B, S, D]
        # repeat decoder
        pre_motion = self.REmechanism(h)  # [B,2]
        pre_r, pre_e = pre_motion[:, 0].unsqueeze(dim=-1), pre_motion[:, 1].unsqueeze(dim=-1)  # [B,1],得到预测属于不同motion的概率
        p_r = self.Rdecoder(h, input_ids, time_gap, cla, tr_te)  # [B,vocab]
        p_e = self.Edecoder(h, input_ids)
        logits_lm_r = pre_r * p_r + pre_e * p_e  # [B,vocab]
        logits_lm_r = logits_lm_r.unsqueeze(dim=1)  # [B, 1, vocab]
        # bert decoder
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # [batch,pre_num,dim]
        h_masked = torch.gather(h, 1, masked_pos)  # [batch,pre_num,dim],拿到被mask位置的商品嵌入表示
        h_masked = self.norm(self.gelu(self.linear(h_masked)))
        logits_lm_t = self.decoder(h_masked) + self.decoder_bias  # 得到softmax结果
        logits_lm = self.rt * logits_lm_t + (1 - self.rt) * logits_lm_r
        return logits_lm

    def gelu(self, x):  # 无需改，一种激活函数
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


'''
得到用户购买商品数据dataset模块,不带嵌入表示的，只给input_idx
'''
#得到G_u序列

class ItemData(tDataset):
    def __init__(self, user_gp, user_gp_stamp, user_gp_stamp_t, user_gp_cla, vocabulary, cfg, tr_te):
        self.user_gp = list(user_gp)
        self.user_gp_stamp = list(user_gp_stamp)
        self.user_gp_stamp_t = list(user_gp_stamp_t)
        self.user_gp_cla = list(user_gp_cla)
        self.vocab = list(vocabulary)  # 词表

        self.pad_id = cfg['pad_id']
        self.mask_id = cfg['mask_id']
        self.interest_id = cfg['interest_id']

        self.dic_vocab_idx = self.str2idx(self.vocab)
        self.dic_vocab_idx["<PAD>"] = self.pad_id#257
        self.dic_vocab_idx['[MASK]'] = self.mask_id#256
        self.dic_vocab_idx['[Interest]'] = self.interest_id#258
        self.max_pred = cfg['max_pred']  # max tokens of prediction，训练时鉴于样本短，所以只预测一位
        self.max_len = cfg['max_len']
        self.tr_te = tr_te
        
        # 为得到self.input_mask,self.masked_ids, self.masked_pos, self.masked_weights
        self.input_mask, self.masked_ids, self.masked_pos, self.masked_weights = [], [], [], []
        # 为得到self.x,即用emb表示的各个样本，[batch,seq,emb]
        self.x, self.stamp = [], []  # 要的是换成emb的user_gp
        self.stamp_t = []
        self.stamp_gap = []
        self.cla = []
        # for user in self.user_gp:#超长序列处理可变，目前是截断
        for i in range(len(user_gp)):
            user = self.user_gp[i]
            stamp = self.user_gp_stamp[i]
            stamp_t = self.user_gp_stamp_t[i]
            cla = self.user_gp_cla[i]
            input_ids_insert, input_stamp_insert, input_mask_insert, \
                masked_ids_insert, masked_pos_insert, masked_weight_insert, \
                stamp_t_insert, stamp_gap_insert, cla_insert = \
                    self.prepare(   \
                        list(user[-self.max_len:]), list(stamp[-self.max_len:]), \
                        list(stamp_t[-self.max_len:]),list(cla[-self.max_len:]), cfg)  # 加个list防止原user被修改
            # if i in [1,2,3]:
            #    print(len(masked_ids_insert),len(masked_pos_insert),len(masked_weight_insert))
            # else:
            #    input_insert,input_mask_insert,masked_ids_insert,masked_pos_insert,masked_weight_insert = self.prepare(user[:self.max_len])
            self.x.append(input_ids_insert)
            self.stamp.append(input_stamp_insert)
            self.input_mask.append(input_mask_insert)
            self.masked_ids.append(masked_ids_insert)
            self.masked_pos.append(masked_pos_insert)
            self.masked_weights.append(masked_weight_insert)
            self.stamp_t.append(stamp_t_insert)
            self.stamp_gap.append(stamp_gap_insert)
            self.cla.append(cla_insert)

        print("... end of preparation.")
        # print(self.gap[0])
        # for i in range(11):
        #    print(i,len(self.gap[0][i]))
        self.x, self.stamp = np.array(self.x), np.array(self.stamp)  # 得到一个个emb好的样本
        self.stamp_t = np.array(self.stamp_t)
        self.stamp_gap = np.array(self.stamp_gap)
        self.input_mask, self.masked_ids, self.masked_pos, self.masked_weights = \
            np.array(self.input_mask), np.array(self.masked_ids), np.array(self.masked_pos), np.array(self.masked_weights)
        self.cla = np.array(self.cla)
        # print(self.x[0])
        # print(self.stamp[0])#[B, S, 3]
        # print(self.stamp_t[0])#[B, S+1, S+1, D]

    def prepare(self, tokens, stamp, stamp_t, cla, cfg):  # tokens是一个购买序列
        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = self.max_pred  # 假设只预测一个先
        # candidate positions of masked tokens
        '''
        要改mask位置改这个cand_pos
        '''
        if self.tr_te == 'train':
            cand_pos = np.random.randint(len(tokens), size=n_pred)  # cand_pos = [i for i, token in enumerate(tokens)]
            for pos in cand_pos:
                masked_tokens.append(tokens[pos])
                masked_pos.append(pos + 1)  # 第一个位置留给interest
                if cfg['multi_mask']:
                    if random.random() < 0.8:  # 80%
                        tokens[pos] = '[MASK]'
                    elif random.random() > 0.9:  # 10%
                        tokens[pos] = self.get_random_word(self.vocab)
                else:
                    tokens[pos] = '[MASK]'
        elif self.tr_te == 'test':
            cand_pos = [len(tokens) - 1]
            for pos in cand_pos:
                masked_tokens.append(tokens[pos])
                masked_pos.append(pos + 1)  # 第一个位置留给interest
                tokens[pos] = '[MASK]'
        # for pos in cand_pos[:n_pred]:

        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)
        # 生成time_gap,还未扩充，前面不用帮interest补充，因为onehot计算时没考虑它
        time_gap = self.computegap(stamp_t, cand_pos[0])
        # 先给token的最前面插入代表序列兴趣的token
        tokens.insert(0, '[Interest]')  # 这里在开头多加了一位，而stamp则希望是编码为0，所以在embedding中以全零加入
        stamp_t.insert(0, stamp_t[0])  # 这里在开头多加了一位，姑且用第一个时间戳来作为interest的时间戳
        input_mask = [1] * len(tokens)

        # Token Indexing
        masked_ids = self.convert_tokens_to_ids(self.dic_vocab_idx, masked_tokens)
        input_ids = self.convert_tokens_to_ids(self.dic_vocab_idx, tokens)
        # Zero Padding
        n_pad = self.max_len + 1 - len(tokens)  # 不能考虑interest的位置
        input_mask.extend([self.pad_id] * n_pad)
        input_ids.extend([self.pad_id] * n_pad)
        # stamp.extend([[12,31,7,24]]*n_pad)
        stamp.extend([[12, 31, 7]] * n_pad)
        stamp_t.extend([stamp_t[0]] * n_pad)  # 按原文，用第一个时间戳来补全pad的时间戳
        time_gap.extend([90] * n_pad)  # 即使不小心考虑，也会因为太远而获得很小的概率
        cla.extend([4] * n_pad)
        time_matrix = self.computeRePos(stamp_t, cfg['time_span'])
        # stamp.insert(0,[12,31,7])

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([self.pad_id] * n_pad)
            masked_pos.extend([self.pad_id] * n_pad)
            masked_weights.extend([self.pad_id] * n_pad)

        return input_ids, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla  #

    def computegap(self, time_seq, pos):
        time_vec = np.zeros([len(time_seq)], dtype=np.int32)
        for i in range(len(time_seq)):
            val = int(round((abs(time_seq[pos] - time_seq[i]) / 86400)))
            if val == 0:
                # time_vec[i] = 90#0不算复购，会让幂律变为inf，所以换成一个影响小的
                if self.tr_te == 'train':
                    time_vec[i] = random.randint(1, 90)
                elif self.tr_te == 'test':
                    time_vec[i] = 1
            else:
                time_vec[i] = val
        return list(time_vec)

    def computeRePos(self, time_seq, time_span):  # 一个user序列构建一个基础的时间间隔矩阵
        # size = time_seq.shape[0]
        size = len(time_seq)
        time_matrix = np.zeros([size, size], dtype=np.int32)

        # 用每个序列的最小时间间隔作为time_scale
        time_diff = set()
        for i in range(len(time_seq) - 1):
            if time_seq[i + 1] - time_seq[i] < 0:
                break
            # elif (time_seq[i+1]-time_seq[i])/86400 != 0:#
            elif (time_seq[i + 1] - time_seq[i]) / 86400 > 1:  # 最小也得是1
                time_diff.add((time_seq[i + 1] - time_seq[i]) / 86400)
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        # 根据r_u_ij的计算公式求
        for i in range(size):
            for j in range(size):
                span = int(round((abs(time_seq[i] - time_seq[j]) / 86400) / time_scale))  # 将秒变成天，此处可以先不向下取整，而是用round取整
                if span > time_span:
                    time_matrix[i][j] = time_span
                else:
                    time_matrix[i][j] = span
        return time_matrix  # [S, S]，[S]array

    # def computeScale(time_matrix,time_min):
    # list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))，这是原代码用法
    # return time_matrix#[S, S]

    def get_random_word(self, vocab_words):
        i = random.randint(0, len(vocab_words) - 1)
        return vocab_words[i]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):  # pytorch的dataset生成batch时默认有限类型
        return self.x[index], self.stamp[index], self.input_mask[index], self.masked_ids[index], self.masked_pos[index], \
               self.masked_weights[index], self.stamp_t[index], self.stamp_gap[index], self.cla[index]

    def str2idx(self, vocab):  # 识别emb序列并输出商品序列，例如输出['item1','item3']
        dic_v_idx, count_idx = {}, 0
        for i in vocab:
            dic_v_idx[i] = count_idx
            count_idx += 1
        return dic_v_idx

    def convert_tokens_to_ids(self, vocab, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        # print('x中遇到新的商品：')
        for token in tokens:
            if token in vocab:
                ids.append(vocab[token])
            else:
                print(token, end=",")
                vocab[token] = len(vocab) - 2
                ids.append(vocab[token])
        # ids = [vocab[token] for token in tokens if token in vocab else len(vocab) - 2]    # 本行可以替代上面所有
        return ids

# 数据集的处理
# 训练数据集的处理，得到S_u,T_u,C_u序列
def convert_unique_idx(df, column_name,cla_dic=False):
    if cla_dic:
        column_dict = cla_dic
    else:
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    #assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

# 非常恶劣的代码风格，将绝对路径嵌在函数里面
# def Get_user_gp(cla_dic=False,file_dir = r"D:\学习\研究生任务\子璐论文\code\code&data\数据集切割\sampledata" ,test = 10):
def Get_user_gp(cla_dic=False,file_dir = r"../../data/sampledata" ,test = 10):
    df_train=pd.DataFrame()
    data_num = 10
    for i in range(data_num):  #合并数据子集形成训练数据集
        if i+1 != test:
            file_home=file_dir + str(i + 1) + ".csv"
            df = pd.read_csv(file_home, encoding="gbk")  # 读取数据
            df_train=df_train._append(df,ignore_index=True)
    df_train_sort = df_train.sort_values(by ='createtime')
    df_train_sort,cla_mapping = convert_unique_idx(df_train_sort,'classify',cla_dic)
    print(cla_mapping)
    df_train_sort['Time']= df_train_sort.createtime.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp())
    df_train_sort['createtime'] = pd.to_datetime(df_train_sort.createtime)#time stamp
    df_train_sort['month'] = df_train_sort.createtime.dt.month #apply(lambda row:row.month,1)
    df_train_sort['day'] = df_train_sort.createtime.dt.day #apply(lambda row:row.day,1)
    df_train_sort['weekday'] = df_train_sort.createtime.dt.weekday #.apply(lambda row:row.weekday(),1)
    df_train_sort['hour'] = df_train_sort.createtime.dt.hour #.apply(lambda row:row.hour,1)
    user_gp_item = df_train_sort.groupby(['user'])['item'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_t = df_train_sort.groupby(['user'])['Time'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_month = df_train_sort.groupby(['user'])['month'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_day = df_train_sort.groupby(['user'])['day'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_weekday = df_train_sort.groupby(['user'])['weekday'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_hour = df_train_sort.groupby(['user'])['hour'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_cla = df_train_sort.groupby(['user'])['classify'].agg({list}).reset_index()['list'].values.tolist()
    return user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla#,user_gp_hour

# 非常恶劣的代码风格，将绝对路径嵌在函数里面
# def Test_user_gp(cla_dic=False,file_dir = r"D:\学习\研究生任务\子璐论文\code\code&data\数据集切割\sampledata",data_num = 10):
def Test_user_gp(cla_dic=False,file_dir = r"../../data/sampledata",data_num = 10):
    print('Testing user group on test-set:',data_num)
    df_train=pd.DataFrame()
    file_home=file_dir + str(data_num) + ".csv"
    df = pd.read_csv(file_home, encoding="gbk")  # 读取数据
    df_train=df_train._append(df,ignore_index=True)
    df_train_sort = df_train.sort_values(by ='createtime')
    df_train_sort,cla_mapping = convert_unique_idx(df_train_sort,'classify',cla_dic)
    print(cla_mapping)
    df_train_sort['Time']= df_train_sort.createtime.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp())
    df_train_sort['createtime'] = pd.to_datetime(df_train_sort.createtime)
    df_train_sort['month'] = df_train_sort.createtime.dt.month #apply(lambda row:row.month,1)
    df_train_sort['day'] = df_train_sort.createtime.dt.day #.apply(lambda row:row.day,1)
    df_train_sort['weekday'] = df_train_sort.createtime.dt.weekday #apply(lambda row:row.weekday(),1)
    df_train_sort['hour'] = df_train_sort.createtime.dt.hour #apply(lambda row:row.hour,1)
    user_gp_item = df_train_sort.groupby(['user'])['item'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_t = df_train_sort.groupby(['user'])['Time'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_month = df_train_sort.groupby(['user'])['month'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_day = df_train_sort.groupby(['user'])['day'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_weekday = df_train_sort.groupby(['user'])['weekday'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_hour = df_train_sort.groupby(['user'])['hour'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_cla = df_train_sort.groupby(['user'])['classify'].agg({list}).reset_index()['list'].values.tolist()
    return user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla


def get_loss(model, batch, device, cfg):  # make sure loss is tensor
    x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch  # masked_pos其实好像没啥用
    x = x.to(torch.int64).to(device)
    stamp = stamp.to(torch.int64).to(device)
    # gap = gap.to(torch.int64)
    masked_ids = masked_ids.to(torch.int64).to(device)
    masked_pos = masked_pos.to(torch.int64).to(device)
    time_matrix = time_matrix.to(torch.int64).to(device)
    time_gap = time_gap.to(torch.int64).to(device)
    cla = cla.to(torch.int64).to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    logits_lm = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'train')  # [batch,pre_num,vocab_num] forwawrd
    loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
    loss_lm = (loss_lm * masked_weights.float()).mean()

    return logits_lm, loss_lm

def train(user_gp, user_gp_test, user_gp_stamp, user_gp_stamp_test,user_gp_stamp_t, user_gp_stamp_t_test, user_gp_cla,user_gp_cla_test, vocabulary, cfg):
    dataset = ItemData(user_gp, user_gp_stamp, user_gp_stamp_t, user_gp_cla, vocabulary, cfg, 'train')
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    dataset_test = ItemData(user_gp_test, user_gp_stamp_test, user_gp_stamp_t_test, user_gp_cla_test, vocabulary, cfg,'test')
    loader_test = DataLoader(dataset_test, batch_size=cfg['batch_size'] * 10, shuffle=True)
    torch.manual_seed(cfg['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertRepeatModel(cfg,device)
    if cfg['pre_train']:
        model.transformer = torch.load(cfg['pre_train']).transformer
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    # if torch.cuda.is_available():
    #     print("Using GPU")
    #     device = torch.device("cuda")
    #     model = BertRepeatModel(cfg,device)
    #     if cfg['pre_train']:
    #         model.transformer = torch.load(cfg['pre_train']).transformer
    #     model = model.cuda()
    # else:
    #     device = torch.device("cpu")
    #     model = BertRepeatModel(cfg,device)
    #     if cfg['pre_train']:
    #         model.transformer = torch.load(cfg['pre_train']).transformer
    #     model = model.cpu()
    '''
    #if token embed block needs L2 norm
    model_else = []
    for name, p in model.named_parameters():
        if name != 'transformer.embed.tok_embed.weight':
            model_else += [p] 
    optim = torch.optim.Adam([{'params':model.transformer.embed.tok_embed.parameters(),'weight_decay':cfg['weight_decay']},{'params':model_else}],lr=cfg['lr'])
    '''
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])  # 也可以换其它的优化器
    loss_total = []
    loss_per_eps = []
    recall_total = []
    t1 = time.time()
    for i in range(cfg['iter_n']):
        model.train()
        count_iter = 0
        # t1 = time.time()
        for batch_idx, batch in enumerate(loader):
            optim.zero_grad()
            logits, loss = get_loss(model, batch, device, cfg)
            loss.backward()
            optim.step()
            loss_total.append(loss.detach().numpy())
            count_iter += 1
            if batch_idx % 50 == 0:
                print(
                    "Epoch: ", i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                )
                # t2 = time.time()-t1
                # t1 = time.time()
                # print('用时：',t2)
        loss_per_eps.append(sum(loss_total[-count_iter:]) / count_iter)
        print("| loss_per_eps: %.3f" % loss_per_eps[-1])
        if Config['test_while_train']:
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.cuda() if torch.cuda.is_available() else model.cpu()
            # if torch.cuda.is_available():
            #     print("GPU test avaliable")
            #     device = torch.device("cuda")
            #     model = model.cuda()
            # else:
            #     device = torch.device("cpu")
            #     model = model.cpu()
            average = 0
            correct_range = cfg['correct_range']
            # new
            for ctt in range(cfg['test_times']):
                ans = test(loader_test, model, correct_range, device)
                average += ans
                ans_test = test(loader_test, model, correct_range, device)
            print('----------------- Average Accuracy:', average / cfg['test_times'], '---------------------')
            recall_total.append(average / cfg['test_times'])
            if cfg['save']:
                torch.save(model, cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')
                print('Model saved to', cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')

    print('use time:', time.time() - t1)

    # 画loss图
    plt.plot(loss_total)
    plt.show()
    plt.plot(loss_per_eps)
    plt.show()
    print('Mean Recall after 15 epochs:', mean(recall_total[15:]))
    print(recall_total)

def test(loader_test, model, correct_range, device):
    hits_all = 0
    test_num = 0  # new # new
    for batch_idx, batch in enumerate(loader_test):
        # print('start batch', batch_idx, '----------------------')
        x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch
        x = x.to(torch.int64).to(device)
        stamp = stamp.to(torch.int64).to(device)
        time_matrix = time_matrix.to(torch.int64)
        time_gap = time_gap.to(torch.int64).to(device)
        cla = cla.to(torch.int64).to(device)
        masked_ids = masked_ids.to(torch.int64).to(device)
        masked_pos = masked_pos.to(torch.int64).to(device)
        input_mask = input_mask.to(torch.int64).to(device)
        masked_weights = masked_weights.to(torch.float32).to(device)
        # y_pre = model(x, stamp, gap, input_mask, masked_pos).tolist()  #
        y_pre = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'test')
        # new_start
        _, indices = torch.topk(y_pre, correct_range, -1)
        indices = indices.view(indices.size()[0], -1)
        masked_ids = masked_ids.view(-1, 1).expand_as(indices)
        hits = (masked_ids == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (masked_ids == indices).nonzero()[:, :-1].size(0)
        hits_all += n_hits
        test_num += masked_ids.size(0)
        # recall = float(n_hits) / masked_ids.size(0)
        # print('这个batch的准确率为：', recall)
        # recall_li.append(recall)
        # new_end
    print('hit:', hits_all, '|total:', test_num)
    return hits_all / test_num

Config = {
    'vocab_size': 259,  # Size of the vocabulary
    'dim': 35,  # Dimension
    'n_layers': 1,  # Number of layers
    'n_heads': 5,  # Number of attention heads
    'dim_ff': 35 * 4,  # Dimension of the feed-forward layer
    'p_drop_hidden': 0.1,  # Dropout probability for hidden layers
    'p_drop_attn': 0.1,  # Dropout probability for attention layers
    'max_len': 11,  # Maximum length of input sequences
    'iter_n': 100,  # Number of iterations
    'half_epoch': 50,  # Half of the epoch size
    'batch_size': 128,  # Size of each training batch
    'save': False,  # Flag indicating whether to save the model
    'max_pred': 1,  # Maximum number of predictions
    'lr': 0.002,  # Learning rate
    'weight_decay': 1e-4,  # Weight decay
    'mask_id': 256,  # ID for masking tokens
    'pad_id': 257,  # ID for padding tokens
    'interest_id': 258,  # ID for interest tokens
    'seed': 2300,  # Random seed
    'test_while_train': True,  # Flag indicating whether to test while training
    'test_times': 1,  # Number of times to run the test
    'correct_range': 3,  # Range for correctness
    'model': 'Repeat',  # Model type
    'pro_add': 'add',  # Pro add type
    'time_span': 65,  # Time span
    'pre_train': False,  # Pre-training flag
    'save_dir': 'code/model/',  # Directory to save the model
    'repeat_proj': False,  # Repeat projection flag
    'explore_proj': False,  # Explore projection flag
    'cla_dic': {'视频': 1, '游戏': 3, '出行': 3, '电商': 0, '音频': 1, '阅读': 2, '美食': 2, '酒店': 4, '医疗': 4, '生活服务': 3, '通信': 4, '工具': 0, '教育': 4, '办公': 2, '快递': 4},  # Classification dictionary
    'res': True,  # Res flag
    'multi_mask': True  # Multi-mask flag
}

if __name__ == "__main__":
    user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla \
        = Get_user_gp(cla_dic=Config['cla_dic'],test=10)#获得训练集的用户购买商品序列
    user_gp_stamp = []#把时间戳信息融合在一起
    user_gp_stamp_t = []
    for i in range(len(user_gp_item)):
        insert = []
        insert_t = []
        for j in range(len(user_gp_item[i])):
            #insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j],user_gp_hour[i][j]])
            insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j]])#user_gp_t是s，s/60/60/24向下取整为日期差
            insert_t.append(user_gp_t[i][j])
            #insert.append(user_gp_month[i][j])
        user_gp_stamp.append(insert)
        user_gp_stamp_t.append(insert_t)
    print('Timestamps loaded')

    # vocabulary=np.load('D:\学习\研究生任务\子璐论文\code\code&data\dict\\quanyi_train_vocab.npy')
    vocabulary=np.load('../../data/quanyi_train_vocab.npy')
    vocabulary=vocabulary.tolist()
    #dic_item_emb, vocabulary = Dic_item_emb(emb_dim=Config['dim'])#获得item基础向量表示字典和词表
    print(user_gp_item[:5])#,dic_item_emb.keys())
    print(user_gp_stamp[:5])
    print(user_gp_stamp_t[:5])
    print("vocab loaded.")

    # test部分
    user_gp_test, user_gp_t_test, user_gp_month_test, user_gp_day_test, user_gp_weekday_test, user_gp_cla_test = Test_user_gp(
        cla_dic=Config['cla_dic'],
        data_num=10)  # 获得训练集的用户购买商品序列#user_gp_item,user_gp_month,user_gp_day,user_gp_weekday,user_gp_hour = Get_user_gp(test=10)#获得训练集的用户购买商品序列
    user_gp_stamp_test = []  # 把时间戳信息融合在一起
    user_gp_stamp_t_test = []
    print('#Sequences in test-set:', len(user_gp_test))

    # Narm,Stamp diginetica yoochoose(click) movielen rights FM book
    # test部分
    user_gp_test, user_gp_t_test, user_gp_month_test, user_gp_day_test, user_gp_weekday_test, user_gp_cla_test = Test_user_gp(cla_dic=Config['cla_dic'],data_num=10)  # 获得训练集的用户购买商品序列#user_gp_item,user_gp_month,user_gp_day,user_gp_weekday,user_gp_hour = Get_user_gp(test=10)#获得训练集的用户购买商品序列
    user_gp_stamp_test = []  # 把时间戳信息融合在一起
    user_gp_stamp_t_test = []
    print('#Sequences in test-set:', len(user_gp_test))

    for i in range(len(user_gp_test)):
        insert_test = []
        insert_t_test = []
        for j in range(len(user_gp_test[i])):
            # insert_test.append([user_gp_month_test[i][j],user_gp_day_test[i][j],user_gp_weekday_test[i][j],user_gp_hour_test[i][j]])
            insert_test.append([user_gp_month_test[i][j], user_gp_day_test[i][j], user_gp_weekday_test[i][j]])
            insert_t_test.append(user_gp_t_test[i][j])
            # insert_test.append(user_gp_month_test[i][j])
        user_gp_stamp_test.append(insert_test)
        user_gp_stamp_t_test.append(insert_t_test)

    # print('we have %s rebuy test' % len(user_gp_stamp_test_re))
    # print('we have %s other test' % len(user_gp_stamp_test_no))

    train(user_gp_item,user_gp_test,user_gp_stamp,user_gp_stamp_test,user_gp_stamp_t,user_gp_stamp_t_test,user_gp_cla,user_gp_cla_test,vocabulary,Config)