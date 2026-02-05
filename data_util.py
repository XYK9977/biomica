import torch
import numpy as np


def read_entity_from_id(path):
    entity2id = {}
    with open(path + 'entity2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id


def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id


def read_entity_type(path, entity2id):
    num_ent = len(entity2id)
    type_arr = np.zeros(num_ent, dtype=np.int64) 

    try:
        with open(path + 'entity2type.txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                ent_name = parts[0]
                type_id = int(parts[2])

                if ent_name in entity2id:
                    eid = entity2id[ent_name]
                    type_arr[eid] = type_id

    except FileNotFoundError:
        pass

    return torch.from_numpy(type_arr)


def get_adj(path, split):
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path+split+'.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()  
            e1, r, e2 = instance[0], instance[1], instance[2]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            data.append(relation2id[r])

    return triples, (rows, cols, data), unique_entities

def load_data(datasets):
    path = 'datasets/'+datasets+'/'
    structure_path = None
    train_triples, train_adj, train_unique_entities = get_adj(path, 'train')
    val_triples, val_adj, val_unique_entities = get_adj(path, 'valid')
    test_triples, test_adj, test_unique_entities = get_adj(path, 'test')
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    entity_type = read_entity_type(path, entity2id)


    try:
        ent_features = torch.load(open(structure_path+'entity_embeddings.pt', 'rb'))
        print("load structure entity embedding")
    except:
        ent_features = torch.randn(len(entity2id), 200)
        print("initialize structure entity embedding")
    try:
        rel_features = torch.load(open(structure_path+'relation_embeddings.pt', 'rb'))
        print("load structure relation embedding")
    except:
        rel_features = torch.randn(len(relation2id)*2, 200)
        print("initialize structure entity embedding")
    try:
        bio_features = torch.load(open(path+'bio_features.pth', 'rb'))
        print("load bio entity embedding")
    except:
        bio_features = torch.randn(len(entity2id), 64*6)
        print("initialize bio entity embedding")
    try: 
        text_features = torch.load(open(path+'text_features.pth', 'rb'))
        print("load text entity embedding")
    except:
        text_features = torch.randn(len(entity2id), 100)
        print("initialize text entity embedding")

    return entity2id, relation2id, ent_features, rel_features, bio_features, text_features, entity_type,\
           (train_triples, train_adj, train_unique_entities), \
           (val_triples, val_adj, val_unique_entities), \
           (test_triples, test_adj, test_unique_entities)

if __name__ == "__main__":
    e2id, r2id, ent_f, rel_f, bio_f, txt_f, trn, val, tst = load_data("MKG-Y")
