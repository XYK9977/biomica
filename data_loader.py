import torch
import numpy as np
from collections import defaultdict
import dgl


class KGData:
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        self.device = args.device
        self.train_triples = train_data[0]
        self.val_triples = val_data[0]
        self.test_triples = test_data[0]
        self.max_batch_num = 1

        self.entity2id = {k: v for k, v in entity2id.items()}
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.relation2id = {k: v for k, v in relation2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.batch_size = args.batch_size

    def shuffle(self):
        raise NotImplementedError

    def get_batch(self, batch_num):
        raise NotImplementedError


class MoPEData(KGData):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(MoPEData, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        rel_num = len(relation2id)
        self.rel_num = rel_num
        for k, v in relation2id.items():
            self.relation2id[k+'_reverse'] = v+rel_num
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        sr2o = {}  # er_vocab, include inverse.
        for (head, relation, tail) in self.train_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        self.train_indices = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])} 
                              for (head, relation), tail in sr2o.items()]

        if len(self.train_indices) % self.batch_size == 0:
            self.max_batch_num = len(self.train_indices) // self.batch_size
        else:
            self.max_batch_num = len(self.train_indices) // self.batch_size + 1

        for (head, relation, tail) in self.val_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        for (head, relation, tail) in self.test_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)
 
        self.val_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.val_triples]
        self.val_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.val_triples]
        self.test_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.test_triples]
        self.test_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.test_triples]
        
        self.sr2o = sr2o
        
        self._build_dgl_graph_for_rgcn()


    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def _build_dgl_graph_for_rgcn(self):
        num_ent = len(self.entity2id)

        src = []
        dst = []
        etypes = []

        for (h, r, t) in self.train_triples:
            src.append(h)
            dst.append(t)
            etypes.append(r)

            src.append(t)
            dst.append(h)
            etypes.append(r + self.rel_num)

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        etypes = torch.LongTensor(etypes)

        self.g = dgl.graph((src, dst), num_nodes=num_ent)
        self.g.edata["etype"] = etypes

        self.etypes = etypes

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            batch = self.train_indices[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
        else:
            batch = self.train_indices[batch_num * self.batch_size:]
        batch_indices = torch.LongTensor([indice['triple'] for indice in batch])
        label = [np.int32(indice['label']) for indice in batch] 
        y = np.zeros((len(batch), len(self.entity2id)), dtype=np.float32)
        for idx in range(len(label)):
            for l in label[idx]:
                y[idx][l] = 1.0
        y = 0.9 * y + (1.0 / len(self.entity2id))
        batch_values = torch.FloatTensor(y)

        return batch_indices, batch_values


    def _score_given_hr(self, model, head_id, rel_id,
                        filtered=True, filter_key=None,
                        device=None, ensemble=True):
        if device is None:
            device = self.device

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            triples = torch.LongTensor([[head_id, rel_id, 0]]).to(device)
            preds, _ = model.forward(triples)

            if ensemble:
                score = (preds[0] + preds[1] + preds[2] + preds[3]) / 4.0
            else:
                score = preds[3]

            score = score.squeeze(0)
            if filtered:
                if filter_key is None:
                    filter_key = (head_id, rel_id)
                if hasattr(self, "sr2o") and filter_key in self.sr2o:
                    tails = list(self.sr2o[filter_key])
                    if len(tails) > 0:
                        mask_idx = torch.LongTensor(tails).to(device)
                        score[mask_idx] = float('-inf')

        return score


    def score_tails(self, model, head_id, rel_id,
                    filtered=True, device=None,
                    ensemble=True, as_numpy=False):
        scores = self._score_given_hr(
            model=model,
            head_id=head_id,
            rel_id=rel_id,
            filtered=filtered,
            filter_key=(head_id, rel_id),
            device=device,
            ensemble=ensemble
        )
        if as_numpy:
            return scores.detach().cpu().numpy()
        return scores

    def score_tails_by_name(self, model, head_name, relation_name,
                            filtered=True, device=None,
                            ensemble=True, as_numpy=False):
        if head_name not in self.entity2id:
            raise KeyError(f"Unknown entity name: {head_name}")
        if relation_name not in self.relation2id:
            raise KeyError(f"Unknown relation name: {relation_name}")

        head_id = self.entity2id[head_name]
        rel_id = self.relation2id[relation_name]

        return self.score_tails(
            model=model,
            head_id=head_id,
            rel_id=rel_id,
            filtered=filtered,
            device=device,
            ensemble=ensemble,
            as_numpy=as_numpy
        )

    def score_heads(self, model, tail_id, rel_id,
                    filtered=True, device=None,
                    ensemble=True, as_numpy=False):
        rev_rel_id = rel_id + self.rel_num

        scores = self._score_given_hr(
            model=model,
            head_id=tail_id,
            rel_id=rev_rel_id,
            filtered=filtered,
            filter_key=(tail_id, rev_rel_id),
            device=device,
            ensemble=ensemble
        )
        if as_numpy:
            return scores.detach().cpu().numpy()
        return scores

    def score_heads_by_name(self, model, relation_name, tail_name,
                            filtered=True, device=None,
                            ensemble=True, as_numpy=False):
        if tail_name not in self.entity2id:
            raise KeyError(f"Unknown entity name: {tail_name}")
        if relation_name not in self.relation2id:
            raise KeyError(f"Unknown relation name: {relation_name}")

        tail_id = self.entity2id[tail_name]
        rel_id_full = self.relation2id[relation_name]

        return self.score_heads(
            model=model,
            tail_id=tail_id,
            rel_id=rel_id_full,
            filtered=filtered,
            device=device,
            ensemble=ensemble,
            as_numpy=as_numpy
        )


    def get_validation_pred(self, model, split='test', return_details=False, topk=20):
        device = self.device
        
        att_s, att_i, att_t, att_mm = [], [], [], []

        if split == 'valid':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices

        def eval_side(batch_list, collect_attention=True):
            device  = 'cuda:0'
            triples = torch.LongTensor([b['triple'] for b in batch_list]).to(device)  # [B,3]
            labels = [b['label'] for b in batch_list]

            preds, atts = model.forward(triples) 
            if collect_attention:
                for i in range(triples.size(0)):
                    h, r, t = triples[i].tolist()
                    att_s.append((h, r, t, atts[0][i]))
                    att_i.append((h, r, t, atts[1][i]))
                    att_t.append((h, r, t, atts[2][i]))
                    att_mm.append((h, r, t, atts[3][i]))

            pred = (preds[0] + preds[1] + preds[2] + preds[3]) / 4.0 

            mask = torch.zeros_like(pred, dtype=torch.bool, device=device)
            for i, labs in enumerate(labels):
                mask[i, labs] = True

            target = triples[:, 2]
            target_score = pred[torch.arange(pred.size(0), device=device), target]

            pred = pred.masked_fill(mask, float('-inf'))
            pred[torch.arange(pred.size(0), device=device), target] = target_score

            greater = pred > target_score.unsqueeze(1)
            ranks = greater.sum(dim=1).float() + 1
            return ranks

        ranks_head = []
        for i in range(0, len(head_indices), self.batch_size):
            batch = head_indices[i:i + self.batch_size]
            ranks_head.append(eval_side(batch, collect_attention=True))
        ranks_head = torch.cat(ranks_head, dim=0)

        ranks_tail = []
        for i in range(0, len(tail_indices), self.batch_size):
            batch = tail_indices[i:i + self.batch_size]
            ranks_tail.append(eval_side(batch, collect_attention=False)) 
        ranks_tail = torch.cat(ranks_tail, dim=0)

        all_ranks = torch.cat([ranks_head, ranks_tail], dim=0)
        mrr = (1.0 / all_ranks).mean().item()

        def compute_metrics(ranks_tensor):
            mrr = (1.0 / ranks_tensor).mean().item()
            return {
                "Hits@100": (ranks_tensor <= 100).float().mean().item(),
                "Hits@50":  (ranks_tensor <= 50).float().mean().item(),
                "Hits@20":  (ranks_tensor <= 20).float().mean().item(),
                "Hits@10":  (ranks_tensor <= 10).float().mean().item(),
                "Hits@3":   (ranks_tensor <= 3).float().mean().item(),
                "Hits@1":   (ranks_tensor <= 1).float().mean().item(),
                "MR":       ranks_tensor.mean().item(),
                "MRR":      mrr
            }

        metrics      = compute_metrics(all_ranks)
        metrics_head = compute_metrics(ranks_head)
        metrics_tail = compute_metrics(ranks_tail)


        if not return_details:
            return metrics, metrics_head, metrics_tail, [att_s, att_i, att_t, att_mm]
        else:
            head_details, tail_details = [], []
            with torch.no_grad():
                for idx_list, details, inverse in (
                        (head_indices, head_details, False),
                        (tail_indices, tail_details, True)
                    ):
                    for entry in idx_list:
                        q, r, t = entry['triple']
                        all_labels = entry['label'] 
                        
                        q_tensor = torch.LongTensor([q]).to(device)
                        r_tensor = torch.LongTensor([r]).to(device)
                        preds, _ = model.forward(torch.stack([q_tensor, r_tensor], dim=1))
                        score = ((preds[0] + preds[1] + preds[2] + preds[3]) / 4.0).squeeze(0)  # [E]

                        mask = torch.zeros_like(score, dtype=torch.bool)
                        mask[all_labels] = True
                        score_filtered = score.masked_fill(mask, float('-inf'))
                        
                        true_score = score[t].item()
                        score_filtered[t] = true_score
                        
                        rank = int((score_filtered > true_score).sum().item() + 1)

                        topk_ids = torch.topk(score_filtered, topk).indices.cpu().tolist()
                        
                        details.append({
                            'query':     (q, r),
                            'label':     t,
                            'topk':      topk_ids,
                            'rank':      rank,
                            'inverse':   inverse,
                            'query_name': (self.id2entity[q], self.id2relation[r]),
                            'label_name': self.id2entity[t],
                            'pred_name': [self.id2entity[i] for i in topk_ids],
                        })

            return metrics, metrics_head, metrics_tail, [att_s, att_i, att_t, att_mm], head_details, tail_details



    def get_validation_pred_tucker(self, model, split='test', return_details=False, topk=20):
        ranks_head, ranks_tail = [], []

        if split == 'valid':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        elif split == 'test':
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices
            
        def eval_side(batch_list):
            device  = 'cuda:0'
            triples = torch.LongTensor([b['triple'] for b in batch_list]).to(device)
            labels = [b['label'] for b in batch_list]

            e1 = triples[:, 0]
            rel = triples[:, 1]
            target = triples[:, 2]

            pred = model.forward(e1, rel)

            mask = torch.zeros_like(pred, dtype=torch.bool, device=device)
            for i, lab in enumerate(labels):
                mask[i, lab] = True

            target_score = pred[torch.arange(pred.size(0), device=device), target]

            pred = pred.masked_fill(mask, float('-inf'))

            pred[torch.arange(pred.size(0), device=device), target] = target_score

            greater = pred > target_score.unsqueeze(1)
            rank = greater.sum(dim=1) + 1  

            return rank.float()


        ranks_head = []
        for i in range(0, len(head_indices), self.batch_size):
            batch = head_indices[i:i+self.batch_size]
            ranks_head.append(eval_side(batch))
        ranks_head = torch.cat(ranks_head, dim=0)  # [N]

        ranks_tail = []
        for i in range(0, len(tail_indices), self.batch_size):
            batch = tail_indices[i:i+self.batch_size]
            ranks_tail.append(eval_side(batch))
        ranks_tail = torch.cat(ranks_tail, dim=0)  # [N]

        all_ranks = torch.cat([ranks_head, ranks_tail], dim=0)
        mrr = (1.0 / all_ranks).mean().item()
        def hits_at(k):
            return (all_ranks <= k).float().mean().item()

        metrics = {
            "Hits@100": hits_at(100),
            "Hits@50":  hits_at(50),
            "Hits@20":  hits_at(20),
            "Hits@10":  hits_at(10),
            "Hits@3":   hits_at(3),
            "Hits@1":   hits_at(1),
            "MR":       all_ranks.mean().item(),
            "MRR":      mrr
        }

        if not return_details:
            return metrics

        head_details, tail_details = [], []
        device = 'cuda:0'
        with torch.no_grad():
            for idx_list, store, inv_flag in (
                    (head_indices, head_details, False),
                    (tail_indices, tail_details, True)
                ):
                for entry in idx_list:
                    q_ent, rel, true_ent = entry['triple']
                    all_labels = entry['label'] 

                    q_tensor = torch.LongTensor([q_ent]).to(device)
                    r_tensor = torch.LongTensor([rel]).to(device)
                    scores = model(q_tensor, r_tensor).squeeze(0)

                    mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
                    mask[all_labels] = True
                    scores_filtered = scores.masked_fill(mask, float('-inf'))

                    true_score = scores[true_ent].item()
                    scores_filtered[true_ent] = true_score

                    rank = int((scores_filtered > true_score).sum().item() + 1)


                    topk_ids = torch.topk(scores_filtered, topk).indices.cpu().tolist()

                    store.append({
                        'query':      (q_ent, rel),
                        'label':      true_ent,
                        'topk':       topk_ids,
                        'rank':       rank,
                        'inverse':    inv_flag,
                        'query_name': (self.id2entity[q_ent], self.id2relation[rel]),
                        'label_name': self.id2entity[true_ent],
                        'pred_name':  [self.id2entity[i] for i in topk_ids],
                    })

        return metrics, head_details, tail_details