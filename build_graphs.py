"""
Graph schema per day

graph.x             [N,387] BERT
graph.edge_index    [2,E]   COO
graph.edge_relation [E,7]   one-hot
graph.node_type     [N]     0: process, 1: file, 2: netflow
graph.edge_y        [E]     0: normal, 1: anomalous
graph.y             [E]     0: normal day, 1: attack day
graph.t             [E]     timestamp
graph.day           [int]

"""

import os
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import tqdm

from utils import *
from config import *
from label_data import *
from logger import logger

import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def load_node_lookup(cur):
    sql = "SELECT * FROM node2id ORDER BY index_id"
    cur.execute(sql)
    rows = cur.fetchall()

    id2info = {}
    for row in rows:
        node_hash = str(row[0])
        node_type = row[1]
        node_value = row[2]
        index_id = row[3]
        id2info[index_id] = (node_hash, node_type, node_value)

    return id2info


def node_features_from_id2info(id2info):
    texts = []

    for idx in sorted(id2info.keys()):
        _, node_type, node_value = id2info[idx]

        if node_type == 'subject':
            tokens = ['process'] + path2higlist(node_value)

        elif node_type == 'file':
            tokens = ['file'] + path2higlist(node_value)

        elif node_type == 'netflow':
            tokens = ['netflow'] + ip2higlist(node_value)

        else:
            tokens = [node_type, str(node_value)]

        texts.append(list2str(tokens))

    return texts


def gen_node_embeddings(texts, model):
    emd_path = os.path.join(artifact_dir, "node_embeddings.pt")

    if os.path.exists(emd_path):
        logger.info(f"Loading node embeddings from {emd_path}")
        return torch.load(emd_path, weights_only=True)

    logger.info(f"Encoding {len(texts)} node texts with BERT")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=256,
        normalize_embeddings=False
    )

    torch.save(embeddings, emd_path)
    return embeddings


def gen_node_type_tensor(id2info):
    num_nodes = len(id2info)
    node_type = torch.zeros(num_nodes, dtype=torch.long)

    for idx, (_, ntype, _) in id2info.items():
        node_type[idx] = node_type_map[ntype]

    return node_type

def gen_rel2vec(rel2id):
    num_rel = len(include_edge_type)
    rel2vec = {}

    for rel_name in include_edge_type:
        idx = rel2id[rel_name] - 1
        onehot = torch.zeros(num_rel, dtype=torch.float)
        onehot[idx] = 1.0
        rel2vec[rel_name] = onehot

    return rel2vec

def build_graphs(cur, id2info, node_embeddings, node_type_tensor, rel2vec):
    for day in tqdm.tqdm(range(2,14)):
        tracker = TaintTracker()
        day_tag = "ATTACK" if day in ATTACK_DAYS else "BENIGN"

        logger.info(f"\nBuilding graph for day {day} ({day_tag})")

        start_ts = datetime_to_ns_time_US(f'2018-04-{day} 00:00:00')
        end_ts = datetime_to_ns_time_US(f'2018-04-{day+1} 00:00:00')

        sql = """
        SELECT * from event_table
        WHERE timestamp_rec > '%s' AND timestamp_rec < '%s'
        ORDER BY timestamp_rec
        """ % (start_ts, end_ts)

        cur.execute(sql)
        events = cur.fetchall()
        logger.info(f"Loaded {len(events)} events for day {day}")

        src_list = []
        dst_list = []
        edge_rel_list = []
        edge_label_list = []
        t_list = []

        for e in events:
            src_id = int(e[1])
            dst_id = int(e[4])
            rel_type = str(e[2])
            ts = int(e[5])

            if rel_type not in include_edge_type:
                continue
            
            edge_rel = rel2vec[rel_type]

            src_info = id2info[src_id]
            dst_info = id2info[dst_id]

            if src_info and dst_info:
                src_hash, src_type, src_val = src_info
                dst_hash, dst_type, dst_val = dst_info

                reason, _ = compute_edge_label(src_hash, src_type, src_val, dst_hash, dst_type, dst_val, rel_type, tracker, day=day)
                edge_label = 0 if reason == "Benign" else 1
            
            else:
                edge_label = 0

            src_list.append(src_id)
            dst_list.append(dst_id)
            edge_rel_list.append(edge_rel)
            edge_label_list.append(edge_label)
            t_list.append(ts)

        edge_y = torch.tensor(edge_label_list, dtype=torch.long)

        active_nodes = sorted(set(src_list) | set(dst_list))
        old_to_new = {old: new for new, old in enumerate(active_nodes)}

        src_list = [old_to_new[s] for s in src_list]
        dst_list = [old_to_new[d] for d in dst_list]

        node_embeddings_day = node_embeddings[active_nodes]
        node_type_day = node_type_tensor[active_nodes]

        node_type_onehot = torch.nn.functional.one_hot(
            node_type_day, num_classes=3
        ).float().to(node_embeddings_day.device)

        node_embeddings_day = torch.cat(
            [node_embeddings_day, node_type_onehot],
            dim=1
        )

        graph_y = torch.tensor(1 if day in ATTACK_DAYS else 0, dtype=torch.long)

        graph = Data(
            x=node_embeddings_day,
            edge_index=torch.tensor([src_list, dst_list]),
            edge_relation=torch.vstack(edge_rel_list).float(),
            edge_y=edge_y,
            node_type=node_type_day,
            y=graph_y,
            t=torch.tensor(t_list),
            day=day
        )

        n_mal = edge_y.sum().item()
        n_ben = len(edge_y) - n_mal
        logger.info(f"Edges total: {len(edge_y)}, malicious: {n_mal}, benign: {n_ben}")
        logger.info(f"Graph label: {graph_y.item()}({day_tag})")
        logger.info(f"Tainted nodes: {tracker.get_stats()['Total Malicious Nodes']}")

        save_path = os.path.join(graphs_dir, f"graph_day{day}.pt")
        torch.save(graph, save_path)
        logger.info(f"Saved graph for day {day} to {save_path}")

        # ── Debug: confirm graph tensor shapes and stats ──
        logger.debug(f"[Day {day}] Graph summary:")
        logger.debug(f"  x (node features)  : {graph.x.shape}  | device: {graph.x.device}")
        logger.debug(f"  edge_index          : {graph.edge_index.shape}")
        logger.debug(f"  edge_relation       : {graph.edge_relation.shape}")
        logger.debug(f"  edge_y              : {graph.edge_y.shape}  | unique: {graph.edge_y.unique().tolist()}")
        logger.debug(f"  node_type           : {graph.node_type.shape} | counts: { {int(k): int((graph.node_type==k).sum()) for k in graph.node_type.unique()} }")
        logger.debug(f"  t (timestamps)      : min={graph.t.min().item()}, max={graph.t.max().item()}")
        logger.debug(f"  y (graph label)     : {graph.y.item()} ({day_tag})")
        logger.debug(f"  num_nodes           : {graph.x.shape[0]}, num_edges: {graph.edge_index.shape[1]}")



if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Building graphs from database")
    logger.info("="*60)

    os.makedirs(graphs_dir, exist_ok=True)
    
    cur, connect = init_database_connection()

    id2info = load_node_lookup(cur)
    logger.info(f"Loaded {len(id2info)} nodes")

    node_feat = node_features_from_id2info(id2info)
    logger.info(f"Generated node features")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.max_seq_length = 64
    node_embeddings = gen_node_embeddings(node_feat, model)
    logger.info(f"Generated node embeddings")

    node_type_tensor = gen_node_type_tensor(id2info)
    logger.info(f"Generated node type tensor")

    rel2vec = gen_rel2vec(rel2id)
    logger.info(f"Generated relation one-hot vectors")

    build_graphs(cur = cur, id2info = id2info, node_embeddings = node_embeddings, node_type_tensor = node_type_tensor, rel2vec = rel2vec)

    connect.close()
    logger.info("Finished building graphs")

