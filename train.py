from ctypes import alignment
import torch
import numpy as np
import proposal_methods as PM
import torch.nn as nn
from scipy import ndimage
import random
from soft_dtw_cuda import SoftDTW

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def select_topk_embeddings(scores, embeddings, k):
    _, idx_DESC = scores.sort(descending=True, dim=1)
    idx_topk = idx_DESC[:, :k]
    idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
    selected_embeddings = torch.gather(embeddings, 1, idx_topk)
    return selected_embeddings

def easy_snippets_mining(actionness, embeddings, k_easy):
    select_idx = torch.ones_like(actionness).cuda()
    dropout_layer = nn.Dropout(p=0.6)
    select_idx = dropout_layer(select_idx)
    actionness_drop = actionness * select_idx

    actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
    actionness_rev_drop = actionness_rev * select_idx

    easy_act = select_topk_embeddings(actionness_drop, embeddings, k_easy)
    easy_bkg = select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)
    return easy_act, easy_bkg

def hard_snippets_mining(actionness, embeddings, k_hard):
    aness_np = actionness.cpu().detach().numpy()
    aness_median = np.median(aness_np, 1, keepdims=True)
    aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

    erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, 6))).astype(aness_np.dtype)
    erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, 3))).astype(aness_np.dtype)
    idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
    aness_region_inner = actionness * idx_region_inner
    hard_act = select_topk_embeddings(aness_region_inner, embeddings, k_hard)

    dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, 3))).astype(aness_np.dtype)
    dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, 6))).astype(aness_np.dtype)
    idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
    aness_region_outer = actionness * idx_region_outer
    hard_bkg = select_topk_embeddings(aness_region_outer, embeddings, k_hard)
    return hard_act, hard_bkg

def NCE_loss(q, k, neg, T=0.07):
    # q,k: [B, D]       neg: [B, T', D]
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    neg = neg.permute(0,2,1)            # [B, D, T']
    neg = nn.functional.normalize(neg, dim=1)
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)          # [B, 1]
    l_neg = torch.einsum('nc,nck->nk', [q, neg])                    # [B, T']
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    ce_criterion = nn.CrossEntropyLoss()
    loss = ce_criterion(logits, labels)
    return loss

def cal_snip_cont_loss(features, bt_outputs):
    k_easy = features.size()[1] // 5                                            # 5    
    k_hard = features.size()[1] // 20                                           # 20

    act_atn = bt_outputs['attn'].squeeze(-1)
    actionness = torch.sigmoid(act_atn)

    embeddings = bt_outputs['feat']
    easy_act, easy_bkg = easy_snippets_mining(actionness, embeddings, k_easy)
    hard_act, hard_bkg = hard_snippets_mining(actionness, embeddings, k_hard)
    contrast_pairs = {
        'EA': easy_act,
        'EB': easy_bkg,
        'HA': hard_act,
        'HB': hard_bkg
    }
    HA_refinement = NCE_loss(
        torch.mean(contrast_pairs['HA'], 1), 
        torch.mean(contrast_pairs['EA'], 1), 
        contrast_pairs['EB']
    )
    HB_refinement = NCE_loss(
        torch.mean(contrast_pairs['HB'], 1), 
        torch.mean(contrast_pairs['EB'], 1), 
        contrast_pairs['EA']
    )
    snip_cont_loss = HA_refinement + HB_refinement
    return snip_cont_loss


def train(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels, pairs_id, bt_vn = dataset.load_data(n_similar=args.num_similar)

    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]
    
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    bt_outputs = model(features, device, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    snip_cont_loss = cal_snip_cont_loss(features, bt_outputs)

    total_loss = model.criterion(bt_outputs, labels, seq_len=seq_len, device=device, logger=logger, opt=args, itr=itr, pairs_id=pairs_id, inputs=features)
    total_loss = total_loss + snip_cont_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy()