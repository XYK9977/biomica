import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from dgl.nn.functional import edge_softmax
import dgl.function as fn   
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import SAGEConv


from layers.layer import *
from .model import BaseModel


class SimpleAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(SimpleAdaptorLayer, self).__init__()
        in_dim, out_dim = layers
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, r=None):
        out = self.net(x)
        B, D = out.size()
        disen = out.view(B, 1, D)
        gates = torch.ones(B, 1, device=x.device)
        return out, disen, gates


class GraphSAGEEncoderDGL(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1,
                 agg='mean', dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            in_feats = in_dim if i == 0 else hidden_dim
            out_feats = hidden_dim
            conv = SAGEConv(
                in_feats=in_feats,
                out_feats=out_feats,
                aggregator_type=agg
            )
            self.layers.append(conv)

    def forward(self, g, x):
        h = x
        for li, conv in enumerate(self.layers):
            h = conv(g, h)
            if li != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, bio_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.bio_dim = bio_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.bio_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        batch_size = modal1_emb.size(0)
        joint_list = []
        last_attn = None

        for i in range(self.multi):
            s_i = self.modal1_layers[i](modal1_emb)
            v_i = self.modal2_layers[i](modal2_emb)
            t_i = self.modal3_layers[i](modal3_emb)

            vt_stack = torch.stack((v_i, t_i), dim=1)
            scores = self.ent_attn(vt_stack).squeeze(-1) 
            attn = torch.softmax(scores, dim=-1)
            mm_i = torch.sum(attn.unsqueeze(-1) * vt_stack, dim=1)

            joint_i = 0.5 * s_i + 0.5 * mm_i

            joint_list.append(joint_i)
            last_attn = attn

        x_mm = torch.stack(joint_list, dim=1).mean(1) 
        x_mm = torch.relu(x_mm)
        return x_mm, last_attn

    def relation_gated_fuse(self, modal1_emb, modal2_emb, modal3_emb, rel):
        joint_list = []
        last_attn = None

        g = torch.sigmoid(rel) 
        for i in range(self.multi):
            s_i = self.modal1_layers[i](modal1_emb) 
            v_i = self.modal2_layers[i](modal2_emb)
            t_i = self.modal3_layers[i](modal3_emb) 

            vt_stack = torch.stack((v_i, t_i), dim=1)
            scores = self.ent_attn(vt_stack).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            mm_i = torch.sum(attn.unsqueeze(-1) * vt_stack, dim=1)

            joint_i = g * s_i + (1.0 - g) * mm_i

            joint_list.append(joint_i)
            last_attn = attn

        x_mm = torch.stack(joint_list, dim=1).mean(1)
        x_mm = torch.relu(x_mm)
        return x_mm, last_attn



class CouplingLayer(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        self.register_buffer("mask", mask)

        in_dim = x_dim + cond_dim
        self.s_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Tanh() 
        )
        self.t_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, cond):
        x1 = self.mask * x
        x2 = (1.0 - self.mask) * x

        x1_in = torch.cat([x1, cond], dim=-1)
        s = self.s_net(x1_in) * (1.0 - self.mask)
        t = self.t_net(x1_in) * (1.0 - self.mask)

        y2 = x2 * torch.exp(s) + t
        y = x1 + y2
        log_det = s.sum(dim=-1)
        s_l2 = (s ** 2).sum(dim=-1)

        return y, log_det, s_l2

    def inverse(self, y, cond):
        y1 = self.mask * y
        y2 = (1.0 - self.mask) * y

        y1_in = torch.cat([y1, cond], dim=-1)
        s = self.s_net(y1_in) * (1.0 - self.mask)
        t = self.t_net(y1_in) * (1.0 - self.mask)

        x2 = (y2 - t) * torch.exp(-s)
        x = y1 + x2
        log_det = (-s).sum(dim=-1)
        return x, log_det


class ConditionalRealNVP(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim=256, num_flows=4):
        super(ConditionalRealNVP, self).__init__()
        self.x_dim = x_dim

        masks = []
        for i in range(num_flows):
            if i % 2 == 0:
                m = torch.cat([
                    torch.ones(x_dim // 2),
                    torch.zeros(x_dim - x_dim // 2)
                ])
            else:
                m = 1.0 - masks[-1]
            masks.append(m)

        self.flows = nn.ModuleList([
            CouplingLayer(x_dim, cond_dim, hidden_dim, mask)
            for mask in masks
        ])

    def forward(self, x, cond):
        B = x.size(0)
        z = x
        log_det_sum = torch.zeros(B, device=x.device)
        s_l2_sum = torch.zeros(B, device=x.device)

        for flow in self.flows:
            z, log_det, s_l2 = flow(z, cond)
            log_det_sum += log_det
            s_l2_sum += s_l2

        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + self.x_dim * math.log(2 * math.pi))
        log_px = log_pz + log_det_sum
        s_l2_mean = s_l2_sum.mean()

        return log_px, z, s_l2_mean

    def sample(self, cond):
        B = cond.size(0)
        z = torch.randn(B, self.x_dim, device=cond.device)
        for flow in reversed(self.flows):
            z, _ = flow.inverse(z, cond)
        return z



class RelMoE(BaseModel):
    def __init__(self, args):
        super(RelMoE, self).__init__(args)

        self.entity_type = args.entity_type.to(self.device)

        self.bio_mask = getattr(args, "bio_mask", None)
        self.txt_mask = getattr(args, "txt_mask", None)

        if self.bio_mask is not None:
            self.bio_mask = self.bio_mask.to(self.device)
        if self.txt_mask is not None:
            self.txt_mask = self.txt_mask.to(self.device)

        num_types = int(self.entity_type.max().item()) + 1
        self.type_emb_nf_dim = getattr(args, "type_emb_nf_dim", 32)

        self.type_emb_nf = nn.Embedding(num_types, self.type_emb_nf_dim)
        nn.init.xavier_uniform_(self.type_emb_nf.weight)

        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained_freeze == 0:
            pre_trained_freeze = False
        else:
            pre_trained_freeze = True
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                args.ent_f.to(self.device), freeze=pre_trained_freeze
            )
            self.relation_embeddings = nn.Embedding.from_pretrained(
                args.rel_f.to(self.device), freeze=pre_trained_freeze
            )

            e_in = args.ent_f.shape[1]
            r_in = args.rel_f.shape[1]

            if e_in != args.dim:
                self.struct_proj_e = nn.Linear(e_in, args.dim, bias=False).to(self.device)
            if r_in != args.r_dim:
                self.struct_proj_r = nn.Linear(r_in, args.r_dim, bias=False).to(self.device)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        if "DB15K" in args.dataset:
            bio = args.bio.to(self.device).view(args.bio.size(0), -1) 
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            bio = args.bio.to(self.device).view(args.bio.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
            txt = txt.view(txt.size(0), -1)
        else:
            bio = args.bio.to(self.device).view(args.bio.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        modal_freeze = bool(getattr(args, "modal_freeze", 0))
        self.bio_entity_embeddings = nn.Embedding.from_pretrained(bio, freeze=modal_freeze)
        self.bio_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.bio_relation_embeddings.weight)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=modal_freeze)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.bio_raw_dim = self.bio_entity_embeddings.weight.data.shape[1]
        self.txt_raw_dim = self.txt_entity_embeddings.weight.data.shape[1]

        self.uni_dim = self.dim 
        self.fuse_out_dim = self.uni_dim

        self.bio_dim = self.uni_dim
        self.txt_dim = self.uni_dim

        self.g = args.g.to(self.device)
        self.etypes = args.etypes.to(self.device)
        num_rels = int(self.etypes.max().item()) + 1
        self.struct_gnn = GraphSAGEEncoderDGL(
            in_dim=self.dim,
            hidden_dim=self.dim,
            num_layers=getattr(args, "gnn_layers", 2),
            agg='mean',
            dropout=0.0,
        )
        self._struct_all = None 


        self.type_temp_param = nn.Embedding(num_types, 1)
        nn.init.constant_(self.type_temp_param.weight, 0.0)

        self.align_weight = getattr(args, "align_weight", 0.01)

        self._last_nf_loss = None
        self._last_align_loss = None

        cond_dim = self.dim + self.txt_raw_dim + self.type_emb_nf_dim
        self.bio_flow = ConditionalRealNVP(
            x_dim=self.bio_raw_dim, 
            cond_dim=cond_dim,
            hidden_dim=getattr(args, "flow_hidden_dim", 256),
            num_flows=getattr(args, "flow_num_layers", 4),
        )

        self.nf_weight = getattr(args, "nf_weight", 0.001)
        self.flow_s_l2 = 0.0001
        self.use_nf_impute_online = getattr(args, "use_nf_impute_online", 1) 

        self.TuckER_S = TuckERLayer(self.uni_dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.uni_dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.uni_dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(self.uni_dim, self.uni_dim)

        self.structure_moe = SimpleAdaptorLayer(
            n_exps=args.n_exp,
            layers=[self.dim, self.uni_dim]
        )
        self.visual_moe = SimpleAdaptorLayer(
            n_exps=args.n_exp,
            layers=[self.bio_raw_dim, self.uni_dim]
        )
        self.text_moe = SimpleAdaptorLayer(
            n_exps=args.n_exp,
            layers=[self.txt_raw_dim, self.uni_dim]
        )

        self.fuse_e = ModalFusionLayer(
            in_dim=self.uni_dim,
            out_dim=self.uni_dim,
            multi=2,
            bio_dim=self.uni_dim,
            txt_dim=self.uni_dim
        )
        self.fuse_r = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.uni_dim,
            multi=2,
            bio_dim=args.r_dim,
            txt_dim=args.r_dim
        )

        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()
        # self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, batch_inputs):
        self._last_nf_loss = None 
        self._last_align_loss = None

        all_x = self.entity_embeddings.weight
        if hasattr(self, "struct_proj_e"):
            all_x = self.struct_proj_e(all_x) 
        self._struct_all = self.struct_gnn(self.g, all_x) 

        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)

        e_raw = self._struct_all[head]               # (B, e_in)
        r_raw = self.relation_embeddings(relation)         # (B, r_in)
        if hasattr(self, "struct_proj_r"):
            r_raw = self.struct_proj_r(r_raw)              # (B, r_dim)


        e_embed, disen_str, atten_s = self.structure_moe(e_raw, rel_gate)  # (B, uni_dim)
        r_embed = r_raw                                                   # (B, r_dim)

        bio_raw = self.bio_entity_embeddings(head)  # (B, bio_raw_dim)

        e_bio_embed_raw, disen_bio, atten_i = self.visual_moe(bio_raw, rel_gate)  # (B, uni_dim)
        e_bio_embed = e_bio_embed_raw.clone() 

        r_bio_embed = self.bio_relation_embeddings(relation)  # (B, r_dim)

        txt_raw = self.txt_entity_embeddings(head)  # (B, txt_raw_dim)
        e_txt_embed, disen_txt, atten_t = self.text_moe(txt_raw, rel_gate)  # (B, uni_dim)

        r_txt_embed = self.txt_relation_embeddings(relation)  # (B, r_dim)

        if self.training and (self.bio_mask is not None):
            has_bio = self.bio_mask[head] 
            if has_bio.any():
                bio_x = bio_raw[has_bio].detach() 


                cond_info   = self.get_nf_condition(head)
                struct_cond = cond_info["struct"][has_bio].detach() 
                text_cond   = cond_info["text"][has_bio].detach() 

                type_ids_nf = cond_info["type"][has_bio]
                type_emb_nf = self.type_emb_nf(type_ids_nf)  

                cond_vec = torch.cat(
                    [struct_cond, text_cond, type_emb_nf],
                    dim=-1
                )  # [B_has, cond_dim]

                log_px, _, s_l2_mean = self.bio_flow(bio_x, cond_vec)    # log p(x | cond)
                self._last_nf_loss = -log_px.mean() + self.flow_s_l2 * s_l2_mean
            else:
                self._last_nf_loss = None
        else:
            self._last_nf_loss = None

        if (self.bio_mask is not None) and (self.use_nf_impute_online == 1):
            missing = ~self.bio_mask[head]  
            if missing.any():
                cond_info_full = self.get_nf_condition(head)
                struct_m = cond_info_full["struct"][missing].detach()  # [B_m, dim]
                text_m   = cond_info_full["text"][missing].detach()    # [B_m, txt_dim]
                type_ids_m = cond_info_full["type"][missing]
                type_emb_m = self.type_emb_nf(type_ids_m) 

                cond_vec_m = torch.cat(
                    [struct_m, text_m, type_emb_m],
                    dim=-1
                )  # [B_m, cond_dim]

                if self.training:
                    gen_raw = self.bio_flow.sample(cond_vec_m)                 # [B_m, bio_dim] NF 生成 raw bio
                    gen_bio_embed, _, _ = self.visual_moe(
                        gen_raw, rel_gate[missing]
                    ) 
                else:
                    with torch.no_grad():
                        gen_raw = self.bio_flow.sample(cond_vec_m)
                        gen_bio_embed, _, _ = self.visual_moe(
                            gen_raw, rel_gate[missing]
                        )

                e_bio_embed[missing] = gen_bio_embed

        if self.training:
            self._last_align_loss = self.compute_align_loss(
                e_embed, e_bio_embed, e_txt_embed, head
            )
        else:
            self._last_align_loss = None

        e_mm_embed, attn_f = self.fuse_e.relation_gated_fuse(
            e_embed, e_bio_embed, e_txt_embed, rel_gate
        )
        r_mm_embed, _ = self.fuse_r.relation_gated_fuse(
            r_embed, r_bio_embed, r_txt_embed, rel_gate
        )

        pred_s  = self.TuckER_S(e_embed, r_embed)
        pred_i  = self.TuckER_I(e_bio_embed, r_bio_embed)  
        pred_d  = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)

        all_s, _, _ = self.structure_moe(self._struct_all) 

        all_bio_raw = self.bio_entity_embeddings.weight

        if (self.bio_mask is not None) and (self.use_nf_impute_online == 1):
            has_bio_all = self.bio_mask                          # (|E|,)
            missing_all = ~has_bio_all
            if missing_all.any():
                ent_ids_all = torch.arange(
                    all_bio_raw.size(0),
                    device=self.device
                )
                cond_all = self.get_nf_condition(ent_ids_all)    # struct / text / type

                struct_m = cond_all["struct"][missing_all].detach()   # (num_missing, dim)
                text_m   = cond_all["text"][missing_all].detach()     # (num_missing, txt_raw_dim)
                type_ids_m = cond_all["type"][missing_all]            # (num_missing,)
                type_emb_m = self.type_emb_nf(type_ids_m)             # (num_missing, type_emb_nf_dim)

                cond_vec_m = torch.cat([struct_m, text_m, type_emb_m], dim=-1)

                with torch.no_grad():
                    gen_bio_all = self.bio_flow.sample(cond_vec_m)

                all_bio_raw_imputed = all_bio_raw.clone()
                all_bio_raw_imputed[missing_all] = gen_bio_all
            else:
                all_bio_raw_imputed = all_bio_raw
        else:
            all_bio_raw_imputed = all_bio_raw

        # map full bio to unified space
        all_v, _, _ = self.visual_moe(all_bio_raw_imputed) 

        all_t_raw = self.txt_entity_embeddings.weight 
        all_t, _, _ = self.text_moe(all_t_raw) 

        all_f, _ = self.fuse_e(all_s, all_v, all_t)              # (|E|, uni_dim)

        pred_s  = torch.mm(pred_s,  all_s.t())                   # (B, |E|)
        pred_i  = torch.mm(pred_i,  all_v.t())
        pred_d  = torch.mm(pred_d,  all_t.t())
        pred_mm = torch.mm(pred_mm, all_f.t())

        pred_s  = torch.sigmoid(pred_s)
        pred_i  = torch.sigmoid(pred_i)
        pred_d  = torch.sigmoid(pred_d)
        pred_mm = torch.sigmoid(pred_mm)

        if not self.training:
            return [pred_s, pred_i, pred_d, pred_mm], [atten_s, atten_i, atten_t, attn_f]
        else:
            return [pred_s, pred_i, pred_d, pred_mm], [disen_str, disen_bio, disen_txt]


    def loss_func(self, output, target, return_components=False):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)

        main_loss = loss_s + loss_i + loss_d + loss_mm

        nf_loss = None
        align_loss = None
        total_loss = main_loss

        if self.training and (self._last_nf_loss is not None):
            nf_loss = self._last_nf_loss
            total_loss = total_loss + self.nf_weight * nf_loss

        if self.training and (self._last_align_loss is not None):
            align_loss = self._last_align_loss
            total_loss = total_loss + self.align_weight * align_loss

        if return_components:
            main_detach   = main_loss.detach()
            nf_detach     = nf_loss.detach() if nf_loss is not None else None
            align_detach  = align_loss.detach() if align_loss is not None else None
            return total_loss, main_detach, nf_detach, align_detach

        return total_loss

    def compute_align_loss(self, e_struct, e_bio, e_txt, entity_ids):
        s = F.normalize(e_struct, dim=-1)
        i = F.normalize(e_bio,    dim=-1) 
        t = F.normalize(e_txt,    dim=-1)  

        B = s.size(0)
        device = s.device
        labels = torch.arange(B, device=device)

        types = self.entity_type[entity_ids]                     # (B,)
        tau_vec = F.softplus(self.type_temp_param(types)) + 1e-4 # (B, 1)

        def clip_loss(x, y, tau_row):
            sim_xy = x @ y.t() 
            sim_yx = y @ x.t()

            logit_xy = sim_xy / tau_row 
            logit_yx = sim_yx / tau_row

            loss_xy = F.cross_entropy(logit_xy, labels)
            loss_yx = F.cross_entropy(logit_yx, labels)
            return 0.5 * (loss_xy + loss_yx)


        loss_st = clip_loss(s, t, tau_vec)
        loss_si = clip_loss(s, i, tau_vec)
        loss_ti = clip_loss(t, i, tau_vec)

        return loss_st + loss_si + loss_ti

    def get_nf_condition(self, entity_ids):
        if not torch.is_tensor(entity_ids):
            entity_ids = torch.LongTensor(entity_ids)
        entity_ids = entity_ids.to(self.device)

        text_cond = self.txt_entity_embeddings(entity_ids)

        struct_cond = self._struct_all[entity_ids]

        type_cond = self.entity_type[entity_ids]

        return {
            "text":   text_cond,    # (B, txt_dim)
            "struct": struct_cond,  # (B, dim)
            "type":   type_cond     # (B,)
        }
    



    
