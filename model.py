import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), 
                                       nn.Conv1d(512, 1, 1), nn.Dropout(0.5), nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat


class STCL_THU(torch.nn.Module):
    def __init__(self, dataset, device, n_class,**args):
        super().__init__()
        n_feature = dataset.feature_size
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio

        self.num_segments = dataset.num_segments
        
        # self.branch_num=args['opt'].branch_num
        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.fusion = nn.Sequential(nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))

        # intra-video temporal relation modeling layers
        intermediate_channel = 512
        self.intra_temp_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.intra_temp_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.intra_temp_conv2_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=3, stride=1, padding=1)                ######
        self.intra_temp_conv2_v = nn.Conv1d(n_feature, n_feature, kernel_size=3, stride=1, padding=1)                           ######
        # inter-video relation modeling layers
        self.inter_vid_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.inter_vid_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.inter_exem_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.inter_exem_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.clswise_atn_conv = nn.Conv1d(n_feature, 1, kernel_size=1, stride=1, padding=0)
        # relu layer
        self.opt = nn.ReLU()
        # intra classifier layer
        self.intra_temp_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        # inter classifier layer
        self.inter_temp_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim * 2, embed_dim, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        # classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))

        self.kl_criterion = nn.KLDivLoss(reduction='mean')
        
        self.dataset = dataset

        self.apply(weights_init)


    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def load_act_vid_feats(self, classwiseidx, video_features, act_vid_num, device):
        all_acts_vids_feat = []
        for act_idxs in classwiseidx:
            act_sampled_idxs = np.random.choice(act_idxs, act_vid_num, replace=False)
            act_sampled_feats = []
            for act_sampled_idx in act_sampled_idxs:
                act_sampled_vid_feat = video_features[act_sampled_idx]
                vid_sample_idx = self.random_perturb(act_sampled_vid_feat.shape[0])
                act_sampled_vid_feat = act_sampled_vid_feat[vid_sample_idx]
                act_sampled_feats.append(act_sampled_vid_feat)
            act_sampled_feats = np.array(act_sampled_feats)
            all_acts_vids_feat.append(act_sampled_feats)
        act_vid_feats = np.array(all_acts_vids_feat)
        act_vid_feats = torch.from_numpy(act_vid_feats).float().to(device)
        return act_vid_feats

    def intra_weight(self, temp_feat):
        temp_feat = temp_feat.transpose(-1, -2) 
        cos = lambda m: F.normalize(m) @ F.normalize(m).t()
        cos_similarity = torch.stack([cos(m) for m in temp_feat])
        cos_similarity = F.softmax(cos_similarity, dim=1)
        return cos_similarity

    def inter_weight(self, vid_feat, exem_feat):
        vid_feat = vid_feat.transpose(-1, -2)
        cos_similarity = []
        for i in range(vid_feat.size()[0]):
            cos = F.normalize(vid_feat[i]) @ F.normalize(exem_feat)
            cos_similarity.append(cos)
        cos_similarity = torch.cat(cos_similarity, dim=0)
        cos_similarity = F.softmax(cos_similarity, dim=-1)
        return cos_similarity

    def weighted_sum(self, temp_feat, cos_sim):
        weighted_feat = torch.bmm(temp_feat, cos_sim)
        weighted_feat = weighted_feat + temp_feat
        return weighted_feat

    def inter_weighted_sum(self, clswise_exem_feat_v, clswise_exem_sim_weight):
        clswise_exem_sim_weight = clswise_exem_sim_weight.permute(0, 2, 1)
        cls_weighed_exem_feat = torch.matmul(clswise_exem_feat_v, clswise_exem_sim_weight)
        return cls_weighed_exem_feat

    def forward(self, inputs, device, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)

        # intra-video temporal relation modeling (layer1)
        layer1_key = self.intra_temp_conv1_k(nfeat)
        layer1_value = self.intra_temp_conv1_v(nfeat)
        layer1_sim_weight = self.intra_weight(layer1_key)
        layer1_feat = self.weighted_sum(layer1_value, layer1_sim_weight)
        layer1_feat = self.opt(layer1_feat)
        # intra-video temporal relation modeling (layer2)
        layer2_key = self.intra_temp_conv2_k(layer1_feat)
        layer2_value = self.intra_temp_conv2_v(layer1_feat)
        layer2_sim_weight = self.intra_weight(layer2_key)
        layer2_feat = self.weighted_sum(layer2_value, layer2_sim_weight)
        layer2_feat = self.opt(layer2_feat)
        intra_cls = self.intra_temp_classifier(layer2_feat)

        # class inter-video semantic relation modeling
        # extract M exemplary features for each action and background category
        act_vid_num = 10
        all_video_feats = self.load_act_vid_feats(self.dataset.classwiseidx, self.dataset.features, act_vid_num, device)
        
        num_class, vid_proto_num, snip_num, feat_dim = all_video_feats.size()
        all_video_feats = all_video_feats.view(-1, snip_num, feat_dim).transpose(-1, -2)

        all_vid_rgb_attn, all_vid_rgb_feat = self.vAttn(all_video_feats[:, :1024, :], all_video_feats[:, 1024:, :])
        all_vid_flow_attn, all_vid_flow_feat = self.fAttn(all_video_feats[:, 1024:, :], all_video_feats[:, :1024, :])
        all_vid_attn = (all_vid_rgb_attn + all_vid_flow_attn) / 2
        all_vid_feat = torch.cat((all_vid_rgb_feat, all_vid_flow_feat), dim=1)
        all_vid_feat = self.fusion(all_vid_feat)

        all_vid_cls = self.classifier(all_vid_feat)
        all_vid_feat_ori_shape = all_vid_feat.view(num_class, vid_proto_num, feat_dim, snip_num).transpose(-1, -2)
        all_vid_attned_cls = self._multiply(all_vid_cls.transpose(-1, -2), all_vid_attn.transpose(-1, -2), include_min=True)
        all_vid_fore_bck = torch.cat((all_vid_attned_cls[:,:,:-1], all_vid_cls.transpose(-1, -2)[:,:,-1:]), dim=-1)
        all_vid_fore_bck = all_vid_fore_bck.view(num_class, vid_proto_num, snip_num, -1)
        # extract exemplar feature for each action category
        all_cls_fore_exemplar = []
        all_cls_bkg_exemplar = []
        for act_cls in range(num_class):
            act_fore_score = all_vid_fore_bck[act_cls][:, :, act_cls : act_cls+1]
            act_bkg_score = all_vid_fore_bck[act_cls][:, :, -1:]
            _, topk_fore_idx = torch.topk(act_fore_score, dim=1, k=snip_num//10)
            topk_fore_feat = all_vid_feat_ori_shape[act_cls].gather(dim=1, index=topk_fore_idx.repeat(1, 1, feat_dim))
            act_fore_exemplar = torch.mean(topk_fore_feat, dim=1)
            all_cls_fore_exemplar.append(act_fore_exemplar)

            _, topk_bkg_idx = torch.topk(act_bkg_score, dim=1, k=snip_num//10)
            topk_bkg_feat = all_vid_feat_ori_shape[act_cls].gather(dim=1, index=topk_bkg_idx.repeat(1, 1, feat_dim))
            act_bkg_exemplar = torch.mean(topk_bkg_feat, dim=1)
            all_cls_bkg_exemplar.append(act_bkg_exemplar)
        all_cls_fore_exemplar = torch.stack(all_cls_fore_exemplar, dim=0)
        all_cls_bkg_exemplar = torch.stack(all_cls_bkg_exemplar, dim=0)
        all_cls_mean_bkg_exemplar = torch.mean(all_cls_bkg_exemplar, dim=0, keepdim=True)
        all_cls_exemplar = torch.cat((all_cls_fore_exemplar, all_cls_mean_bkg_exemplar), dim=0)
        all_cls_exemplar = all_cls_exemplar.transpose(-1, -2)

        # measure the similarity between temporal snippet feature and each exemplar from each category
        inter_vid_feat_k = self.inter_vid_conv1_k(nfeat)
        inter_vid_feat_v = self.inter_vid_conv1_v(nfeat)

        feature_clswise_w = torch.empty(nfeat.shape[0], 21, nfeat.shape[2]).cuda()
        for act in range(0, 21):
            clswise_exem_feat = all_cls_exemplar[act:act+1, :, :]
            clswise_exem_feat_k = self.inter_exem_conv1_k(clswise_exem_feat)
            clswise_exem_feat_v = self.inter_exem_conv1_v(clswise_exem_feat)
            clswise_exem_sim_weight = self.inter_weight(inter_vid_feat_k, clswise_exem_feat_k)
            clswise_sim_weighted_vid_feat = self.inter_weighted_sum(clswise_exem_feat_v, clswise_exem_sim_weight)
            if act == 0:
                all_exem_rela_feats = clswise_sim_weighted_vid_feat.unsqueeze(-1)
            else:
                all_exem_rela_feats = torch.cat((all_exem_rela_feats, clswise_sim_weighted_vid_feat.unsqueeze(-1)), dim=-1)
            feature_clswise_w[:, act:act+1, :] = self.clswise_atn_conv(clswise_sim_weighted_vid_feat)
        all_exem_rela_feats = all_exem_rela_feats.transpose(1, 2).contiguous().view(b*n, c, num_class+1)
        feature_clswise_w = F.softmax(feature_clswise_w, dim=1).transpose(1, 2).contiguous().view(b*nfeat.shape[2], num_class+1).unsqueeze(-1)
        all_exem_rela_feats = torch.bmm(all_exem_rela_feats, feature_clswise_w)
        all_exem_rela_feats = all_exem_rela_feats.view(b, nfeat.shape[2], c).transpose(-1, -2)
        inter_out = torch.cat((inter_vid_feat_v, all_exem_rela_feats), dim=1)
        inter_out = self.opt(inter_out)                                              
        inter_cls = self.inter_temp_classifier(inter_out)
        
        x_cls = self.classifier(nfeat)

        x_cls_fusion = 0.55 * x_cls + 0.36 * intra_cls + 0.09 * inter_cls


        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls_fusion.transpose(-1, -2), 'cas_origin': x_cls.transpose(-1, -2),
                'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2), 'f_atn':f_atn.transpose(-1, -2),
                'intra_cas': intra_cls.transpose(-1, -2), 'inter_cas':inter_cls.transpose(-1, -2)}


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        intra_cas = outputs['intra_cas']
        origin_cas = outputs['cas_origin']
        inter_cas = outputs['inter_cas']
        # intra_cas classification loss and inter_cas classification loss
        loss_intra_cls, _ = self.topkloss(intra_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        loss_origin_cls, _ = self.topkloss(origin_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        loss_inter_cls, _ = self.topkloss(inter_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        intra_cas = intra_cas.contiguous().view(-1, 21).unsqueeze(-1)
        origin_cas = origin_cas.contiguous().view(-1, 21).unsqueeze(-1)
        inter_cas = inter_cas.contiguous().view(-1, 21).unsqueeze(-1)
        intra_cas = F.softmax(intra_cas, dim=1)
        origin_cas = F.softmax(origin_cas, dim=1)
        inter_cas = F.softmax(inter_cas, dim=1)
        # loss_intra_inter_KL = self.kl_criterion(intra_cas, inter_cas)
        loss_inter_intra_KL = self.kl_criterion(origin_cas, intra_cas) + self.kl_criterion(intra_cas, origin_cas) + \
                              self.kl_criterion(origin_cas, inter_cas) + self.kl_criterion(inter_cas, origin_cas) + \
                              self.kl_criterion(inter_cas, intra_cas) + self.kl_criterion(intra_cas, inter_cas)

        loss_KL = loss_inter_intra_KL

        # loss_ml
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        #learning weight dynamic, lambda1 (1-lambda1) 
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        # loss_mil
        # element_logits: (10, 500, 21)     labels: (10, 20)
        loss_mil_orig, _ = self.topkloss(element_logits, labels, is_back=True, rat=args['opt'].k, reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp, labels, is_back=False, rat=args['opt'].k, reduce=None)
        # loss_cas
        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)
        
        # loss_oppo and loss_norm
        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() + loss_intra_cls.mean() + loss_origin_cls.mean() + loss_KL + loss_inter_cls.mean() +
                      args['opt'].alpha3 * loss_3_supp_Contrastive +
                      args['opt'].alpha4 * mutual_loss +
                      args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                      args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)
       
        return total_loss

    def topkloss(self, element_logits, labels, is_back=True, lab_rand=None, rat=8, reduce=None):
        if is_back:
            labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(element_logits, k=max(1, int(element_logits.shape[-2] // rat)), dim=-2)
        instance_logits = torch.mean(topk_val, dim=-2)              # (10, 21)
        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self,x,element_logits,labels,is_back=False):
        if is_back:
            labels = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn   = outputs
        
        return element_logits,element_atn

class STCL_ACT(torch.nn.Module):
    def __init__(self, dataset, device, n_class, **args):
        super().__init__()
        n_feature = dataset.feature_size
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio

        self.num_segments = dataset.num_segments

        self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.fAttn = getattr(model,args['opt'].AWM)(1024,args)

        self.fusion = nn.Sequential(nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))

        # intra-video temporal relation modeling layers
        intermediate_channel = 512
        self.intra_temp_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.intra_temp_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.intra_temp_conv2_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=3, stride=1, padding=1)
        self.intra_temp_conv2_v = nn.Conv1d(n_feature, n_feature, kernel_size=3, stride=1, padding=1)
        # inter-video relation modeling layers
        self.inter_vid_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.inter_vid_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.inter_exem_conv1_k = nn.Conv1d(n_feature, intermediate_channel, kernel_size=1, stride=1, padding=0)
        self.inter_exem_conv1_v = nn.Conv1d(n_feature, n_feature, kernel_size=1, stride=1, padding=0)
        self.clswise_atn_conv = nn.Conv1d(n_feature, 1, kernel_size=1, stride=1, padding=0)
        # relu layer
        self.opt = nn.ReLU()
        # intra classifier layer
        self.intra_temp_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        # inter classifier layer
        self.inter_temp_classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim * 2, embed_dim, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        # baseline classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        self.kl_criterion = nn.KLDivLoss(reduction='mean')
        self.dataset = dataset
        
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)                                                               ###
        self.pool=nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) if _kernel is not None else nn.Identity()       ###
        self.apply(weights_init)

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def load_act_vid_feats(self, classwiseidx, video_features, act_vid_num, device):
        all_acts_vids_feat = []
        for act_idxs in classwiseidx:
            act_sampled_idxs = np.random.choice(act_idxs, act_vid_num, replace=False)
            act_sampled_feats = []
            for act_sampled_idx in act_sampled_idxs:
                act_sampled_vid_feat = video_features[act_sampled_idx]
                vid_sample_idx = self.random_perturb(act_sampled_vid_feat.shape[0])
                act_sampled_vid_feat = act_sampled_vid_feat[vid_sample_idx]
                act_sampled_feats.append(act_sampled_vid_feat)
            act_sampled_feats = np.array(act_sampled_feats)
            all_acts_vids_feat.append(act_sampled_feats)
        act_vid_feats = np.array(all_acts_vids_feat)
        act_vid_feats = torch.from_numpy(act_vid_feats).float().to(device)
        return act_vid_feats

    def intra_weight(self, temp_feat):
        temp_feat = temp_feat.transpose(-1, -2)
        cos = lambda m: F.normalize(m) @ F.normalize(m).t()
        cos_similarity = torch.stack([cos(m) for m in temp_feat])
        cos_similarity = F.softmax(cos_similarity, dim=1)
        return cos_similarity

    def inter_weight(self, vid_feat, exem_feat):
        vid_feat = vid_feat.transpose(-1, -2)
        cos_similarity = []
        for i in range(vid_feat.size()[0]):
            cos = F.normalize(vid_feat[i]) @ F.normalize(exem_feat)
            cos_similarity.append(cos)
        cos_similarity = torch.cat(cos_similarity, dim=0)
        cos_similarity = F.softmax(cos_similarity, dim=-1)
        return cos_similarity

    def weighted_sum(self, temp_feat, cos_sim):
        weighted_feat = torch.bmm(temp_feat, cos_sim)
        weighted_feat = weighted_feat + temp_feat
        return weighted_feat

    def inter_weighted_sum(self, clswise_exem_feat_v, clswise_exem_sim_weight):
        clswise_exem_sim_weight = clswise_exem_sim_weight.permute(0, 2, 1)
        cls_weighed_exem_feat = torch.matmul(clswise_exem_feat_v, clswise_exem_sim_weight)
        return cls_weighed_exem_feat

    def forward(self, inputs, device, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        # feat = self.feat_encoder(x)
        v_atn,vfeat = self.vAttn(feat[:,:1024,:],feat[:,1024:,:])
        f_atn,ffeat = self.fAttn(feat[:,1024:,:],feat[:,:1024,:])
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        # intra-video temporal relation modeling (layer1)
        layer1_key = self.intra_temp_conv1_k(nfeat)
        layer1_value = self.intra_temp_conv1_v(nfeat)
        layer1_sim_weight = self.intra_weight(layer1_key)
        layer1_feat = self.weighted_sum(layer1_value, layer1_sim_weight)
        layer1_feat = self.opt(layer1_feat)
        # intra-video temporal relation modeling (layer2)
        layer2_key = self.intra_temp_conv2_k(layer1_feat)
        layer2_value = self.intra_temp_conv2_v(layer1_feat)
        layer2_sim_weight = self.intra_weight(layer2_key)
        layer2_feat = self.weighted_sum(layer2_value, layer2_sim_weight)
        layer2_feat = self.opt(layer2_feat)
        intra_cls = self.intra_temp_classifier(layer1_feat)
        # extract M exemplary features for each action and background category
        act_vid_num = 5
        all_video_feats = self.load_act_vid_feats(self.dataset.classwiseidx, self.dataset.features, act_vid_num, device)
        
        num_class, vid_proto_num, snip_num, feat_dim = all_video_feats.size()
        all_video_feats = all_video_feats.view(-1, snip_num, feat_dim).transpose(-1, -2)

        all_vid_rgb_attn, all_vid_rgb_feat = self.vAttn(all_video_feats[:, :1024, :], all_video_feats[:, 1024:, :])
        all_vid_flow_attn, all_vid_flow_feat = self.fAttn(all_video_feats[:, 1024:, :], all_video_feats[:, :1024, :])
        all_vid_attn = (all_vid_rgb_attn + all_vid_flow_attn) / 2
        all_vid_feat = torch.cat((all_vid_rgb_feat, all_vid_flow_feat), dim=1)
        all_vid_feat = self.fusion(all_vid_feat) 

        all_vid_cls = self.classifier(all_vid_feat)
        all_vid_feat_ori_shape = all_vid_feat.view(num_class, vid_proto_num, feat_dim, snip_num).transpose(-1, -2)
        all_vid_attned_cls = self._multiply(all_vid_cls.transpose(-1, -2), all_vid_attn.transpose(-1, -2), include_min=True)
        all_vid_fore_bck = torch.cat((all_vid_attned_cls[:,:,:-1], all_vid_cls.transpose(-1, -2)[:,:,-1:]), dim=-1)
        all_vid_fore_bck = all_vid_fore_bck.view(num_class, vid_proto_num, snip_num, -1)
        # extract exemplar feature for each action category
        all_cls_fore_exemplar = []
        all_cls_bkg_exemplar = []
        for act_cls in range(num_class):
            act_fore_score = all_vid_fore_bck[act_cls][:, :, act_cls : act_cls+1]
            act_bkg_score = all_vid_fore_bck[act_cls][:, :, -1:]
            _, topk_fore_idx = torch.topk(act_fore_score, dim=1, k=snip_num//10)
            topk_fore_feat = all_vid_feat_ori_shape[act_cls].gather(dim=1, index=topk_fore_idx.repeat(1, 1, feat_dim))
            act_fore_exemplar = torch.mean(topk_fore_feat, dim=1)
            all_cls_fore_exemplar.append(act_fore_exemplar)

            _, topk_bkg_idx = torch.topk(act_bkg_score, dim=1, k=snip_num//10)
            topk_bkg_feat = all_vid_feat_ori_shape[act_cls].gather(dim=1, index=topk_bkg_idx.repeat(1, 1, feat_dim))
            act_bkg_exemplar = torch.mean(topk_bkg_feat, dim=1)
            all_cls_bkg_exemplar.append(act_bkg_exemplar)
        all_cls_fore_exemplar = torch.stack(all_cls_fore_exemplar, dim=0)
        all_cls_bkg_exemplar = torch.stack(all_cls_bkg_exemplar, dim=0)
        all_cls_mean_bkg_exemplar = torch.mean(all_cls_bkg_exemplar, dim=0, keepdim=True)
        all_cls_exemplar = torch.cat((all_cls_fore_exemplar, all_cls_mean_bkg_exemplar), dim=0)
        all_cls_exemplar = all_cls_exemplar.transpose(-1, -2)
        # measure the similarity between temporal snippet feature and each exemplar from each category

        inter_vid_feat_k = self.inter_vid_conv1_k(nfeat)                     
        inter_vid_feat_v = self.inter_vid_conv1_v(nfeat)

        feature_clswise_w = torch.empty(nfeat.shape[0], 101, nfeat.shape[2]).cuda()
        for act in range(0, 101):
            clswise_exem_feat = all_cls_exemplar[act:act+1, :, :]
            clswise_exem_feat_k = self.inter_exem_conv1_k(clswise_exem_feat)
            clswise_exem_feat_v = self.inter_exem_conv1_v(clswise_exem_feat)
            clswise_exem_sim_weight = self.inter_weight(inter_vid_feat_k, clswise_exem_feat_k)
            clswise_sim_weighted_vid_feat = self.inter_weighted_sum(clswise_exem_feat_v, clswise_exem_sim_weight)
            if act == 0:
                all_exem_rela_feats = clswise_sim_weighted_vid_feat.unsqueeze(-1)
            else:
                all_exem_rela_feats = torch.cat((all_exem_rela_feats, clswise_sim_weighted_vid_feat.unsqueeze(-1)), dim=-1)
            feature_clswise_w[:, act:act+1, :] = self.clswise_atn_conv(clswise_sim_weighted_vid_feat)
        all_exem_rela_feats = all_exem_rela_feats.transpose(1, 2).contiguous().view(b*n, c, num_class+1)
        feature_clswise_w = F.softmax(feature_clswise_w, dim=1).transpose(1, 2).contiguous().view(b*nfeat.shape[2], num_class+1).unsqueeze(-1)
        all_exem_rela_feats = torch.bmm(all_exem_rela_feats, feature_clswise_w)
        all_exem_rela_feats = all_exem_rela_feats.view(b, nfeat.shape[2], c).transpose(-1, -2)
        inter_out = torch.cat((inter_vid_feat_v, all_exem_rela_feats), dim=1)
        inter_out = self.opt(inter_out)                                            
        inter_cls = self.inter_temp_classifier(inter_out)

        intra_cls = self.pool(intra_cls)
        inter_cls = self.pool(inter_cls)
        x_cls=self.pool(x_cls)
        x_cls_fusion = 0.55 * x_cls + 0.36 * intra_cls + 0.09 * inter_cls
        x_atn=self.pool(x_atn)
        f_atn=self.pool(f_atn)
        v_atn=self.pool(v_atn)

        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls_fusion.transpose(-1, -2), 'inter_cas':inter_cls.transpose(-1, -2),  
                'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2), 'f_atn':f_atn.transpose(-1, -2), 
                'intra_cas': intra_cls.transpose(-1, -2), 'cas_origin':x_cls.transpose(-1, -2)}


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        intra_cas = outputs['intra_cas']
        origin_cas = outputs['cas_origin']
        inter_cas = outputs['inter_cas']
        # intra_cas classification loss and inter_cas classification loss
        loss_intra_cls, _ = self.topkloss(intra_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        loss_origin_cls, _ = self.topkloss(origin_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        loss_inter_cls, _ = self.topkloss(inter_cas, labels, is_back=True, rat=args['opt'].k, reduce=None)
        intra_cas = intra_cas.contiguous().view(-1, 101).unsqueeze(-1)
        origin_cas = origin_cas.contiguous().view(-1, 101).unsqueeze(-1)
        inter_cas = inter_cas.contiguous().view(-1, 101).unsqueeze(-1)
        intra_cas = F.softmax(intra_cas, dim=1)
        origin_cas = F.softmax(origin_cas, dim=1)
        inter_cas = F.softmax(inter_cas, dim=1)
        # loss_intra_inter_KL = self.kl_criterion(intra_cas, inter_cas)
        loss_inter_intra_KL = self.kl_criterion(origin_cas, intra_cas) + self.kl_criterion(intra_cas, origin_cas) + \
                              self.kl_criterion(origin_cas, inter_cas) + self.kl_criterion(inter_cas, origin_cas) + \
                              self.kl_criterion(inter_cas, intra_cas) + self.kl_criterion(intra_cas, inter_cas)
        
        loss_KL = loss_inter_intra_KL

        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())

        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)
        

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss  + loss_inter_cls.mean()
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() + loss_origin_cls.mean() + 0.1 * loss_KL + loss_intra_cls.mean() + loss_inter_cls.mean() +
                      args['opt'].alpha3*loss_3_supp_Contrastive + 
                      mutual_loss +
                      args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
                      args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)                                                      ###

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self,x,element_logits,labels,is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss
    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn   = outputs
        
        return element_logits,element_atn