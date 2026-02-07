import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

from models.BioMICA import *
from models.model import *
from data_loader import *
from data_util import load_data


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 2000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'RMoE',
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': None,
        'pre_trained': 0, 
        'pre_trained_freeze': 0,
        'modal_freeze': 0, 
        'encoder': 0,
        'entity_features': 0,
        'relation_features': 0,
        'bio_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 100,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 3,
        'mu': 0.0001,
        'bio_dim': 256,
        'txt_dim': 256,
        'nf_weight': 0.001,
        'gnn_layers': 2,
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args


args = parse_args()
print(args)
if args.seed >= 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, ent_features, rel_features, bio_features, text_features, entity_type, \
    train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

pre_freeze = int(getattr(args, "pre_trained_freeze", 0))
modal_freeze = int(getattr(args, "modal_freeze", 0)) 

print(f"pre_trained_freeze: {'yes' if pre_freeze != 0 else 'no'}")
print(f"modal_freeze: {'yes' if modal_freeze != 0 else 'no'}")

args.entity_type = entity_type.long()

num_ent = bio_features.size(0)

bio_mask = (bio_features.abs().sum(dim=1) > 0)    # [num_ent]
text_mask = (text_features.abs().sum(dim=1) > 0)  # [num_ent]

args.bio_mask = bio_mask 
args.txt_mask = text_mask 

corpus = BKGData(args, train_data, val_data, test_data, entity2id, relation2id)

args.g = corpus.g 
args.etypes = corpus.etypes  

if args.entity_features:
    args.ent_f = F.normalize(torch.Tensor(ent_features), p=2, dim=1)
if args.relation_features:
    args.rel_f = F.normalize(torch.Tensor(rel_features), p=2, dim=1)
if args.bio_features:
    args.bio = F.normalize(torch.Tensor(bio_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)

args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'RMoE': BioMICA
}
time.sleep(5)


def train_decoder(args):
    model = model_name[args.model](args)
    args.bio_dim = model.bio_dim
    args.txt_dim = model.txt_dim

    print(str(model))
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)

    t_total = time.time()
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num

    training_range = tqdm(range(args.epochs))
    for epoch in training_range:
        model.train()
        epoch_main_loss = []
        epoch_nf_loss = []
        epoch_align_loss = []

        t = time.time()
        corpus.shuffle()
        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)

            output, _ = model.forward(train_indices)
            loss, main_loss, nf_loss, align_loss = model.loss_func(
                output, train_values, return_components=True
            )
            loss.backward()
            optimizer.step()

            epoch_main_loss.append(main_loss.item())
            epoch_nf_loss.append(0.0 if nf_loss is None else nf_loss.item())
            epoch_align_loss.append(0.0 if align_loss is None else align_loss.item())

        avg_main = sum(epoch_main_loss) / len(epoch_main_loss)
        avg_nf = sum(epoch_nf_loss) / len(epoch_nf_loss) if epoch_nf_loss else 0.0
        avg_align = sum(epoch_align_loss) / len(epoch_align_loss) if epoch_align_loss else 0.0

        training_range.set_postfix(
            main=f"{avg_main:.5f}",
            nf=f"{avg_nf:.5f}",
            align=f"{avg_align:.5f}",
        )

        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print(
                "Epoch {:04d} , average main loss {:.4f} , epoch_time {:.4f}\n".format(
                    epoch + 1, avg_main, time.time() - t
                )
            )
            model.eval()
            with torch.no_grad():
                val_metrics, val_head_metrics, val_tail_metrics, _ = corpus.get_validation_pred(
                    model, split='test'
                )

            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']

            print(f"Epoch: {epoch + 1:04d}")
            print("  Overall:")
            print(model.format_metrics(val_metrics, 'test'))
            print("  Head prediction:")
            print(model.format_metrics(val_head_metrics, 'test_head'))
            print("  Tail prediction:")
            print(model.format_metrics(val_tail_metrics, 'test_tail'))
            print("\n\n")

    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    model.eval()
    with torch.no_grad():
        valid_test_metrics, valid_head_metrics, valid_tail_metrics, _, head_details_val, tail_details_val = \
            corpus.get_validation_pred(
                model,
                split='valid',
                topk=50,
                return_details=True
            )

        test_test_metrics, test_head_metrics, test_tail_metrics, _, head_details, tail_details = \
            corpus.get_validation_pred(
                model,
                split='test',
                topk=50,
                return_details=True
            )

    print('Validation set results (overall):')
    print(model.format_metrics(valid_test_metrics, 'val'))
    print('  Validation head prediction:')
    print(model.format_metrics(valid_head_metrics, 'val_head'))
    print('  Validation tail prediction:')
    print(model.format_metrics(valid_tail_metrics, 'val_tail'))

    print('Final test set results (overall):')
    print(model.format_metrics(test_test_metrics, 'test'))
    print('  Test head prediction:')
    print(model.format_metrics(test_head_metrics, 'test_head'))
    print('  Test tail prediction:')
    print(model.format_metrics(test_tail_metrics, 'test_tail'))

    print('\n'.join(['Test set (best overall metrics during training):',
                     model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        ckpt_dir = "./checkpoint"
        os.makedirs(ckpt_dir, exist_ok=True)

        datapt = f'./result/{args.dataset}/'
        os.makedirs(datapt, exist_ok=True)

        torch.save(model.state_dict(), f'./checkpoint/mmblp_{args.dataset}_{args.model}.pth')
        print('Saved model!')

        with open(datapt + 'head2tail.json', 'w', encoding='utf-8') as f:
            json.dump(head_details, f, ensure_ascii=False, indent=2)
        with open(datapt + 'tail2head.json', 'w', encoding='utf-8') as f:
            json.dump(tail_details, f, ensure_ascii=False, indent=2)
        with open(datapt + 'head2tail_val.json', 'w', encoding='utf-8') as f:
            json.dump(head_details_val, f, ensure_ascii=False, indent=2)
        with open(datapt + 'tail2head_val.json', 'w', encoding='utf-8') as f:
            json.dump(tail_details_val, f, ensure_ascii=False, indent=2)

        with open(datapt + 'entity2id.json', 'w') as f:
            json.dump(corpus.entity2id, f, indent=2, ensure_ascii=False)

        with open(datapt + 'relation2id.json', 'w') as f:
            json.dump(corpus.relation2id, f, indent=2, ensure_ascii=False)
        print('Saved entity2id/relation2id')

        e_s_raw = model.entity_embeddings.weight
        if hasattr(model, "struct_proj_e"):
            e_s = model.struct_proj_e(e_s_raw)
        else:
            e_s = e_s_raw
        e_s_m, _, _ = model.structure_moe(e_s)

        e_i = model.bio_entity_embeddings.weight
        e_i_m, _, _ = model.visual_moe(e_i)

        e_t = model.txt_entity_embeddings.weight
        e_t_m, _, _ = model.text_moe(e_t)

        e_mm, _ = model.fuse_e(e_s_m, e_i_m, e_t_m)

        r_s_raw = model.relation_embeddings.weight
        if hasattr(model, "struct_proj_r"):
            r_s = model.struct_proj_r(r_s_raw)
        else:
            r_s = r_s_raw

        r_i = model.bio_relation_embeddings.weight
        r_t = model.txt_relation_embeddings.weight
        r_gate = model.rel_gate.weight
        r_mm, _ = model.fuse_r.relation_gated_fuse(r_s, r_i, r_t, r_gate)

        torch.save(e_mm.cpu(), datapt + 'entity_mm_embeddings.pt')
        torch.save(r_mm.cpu(), datapt + 'relation_mm_embeddings.pt')

        torch.save(e_s_m.cpu(), datapt + 'entity_s_embeddings.pt')
        torch.save(r_s.cpu(),  datapt + 'relation_s_embeddings.pt')

        torch.save(e_i_m.cpu(), datapt + 'entity_i_embeddings.pt')
        torch.save(r_i.cpu(),  datapt + 'relation_i_embeddings.pt')

        torch.save(e_t_m.cpu(), datapt + 'entity_t_embeddings.pt')
        torch.save(r_t.cpu(),  datapt + 'relation_t_embeddings.pt')

        torch.save(e_s.cpu(),  datapt + 'entity_spre_embeddings.pt')
        torch.save(e_i.cpu(),  datapt + 'entity_ipre_embeddings.pt')
        torch.save(e_t.cpu(),  datapt + 'entity_tpre_embeddings.pt')
        print('Saved embedding file')


if __name__ == '__main__':
    train_decoder(args)
