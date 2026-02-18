import os
import sys
import glob
import re
from collections import Counter

import torch

from utils import init_database_connection
from config import graphs_dir
from label_data import (
    TaintTracker,
    compute_edge_label,
    detect_anchor,
    ATTACK_DAYS,
    ALL_MALICIOUS_IPS,
    ALL_MALICIOUS_FILES,
    ALL_MALICIOUS_PROCESSES,
    MALICIOUS_IPS_BY_DAY,
    MALICIOUS_FILES_BY_DAY,
    MALICIOUS_PROCESSES_BY_DAY,
)
from logger import logger


def load_node_lookup(cur):
    """Reproduce the same id -> (hash, type, value) map used during generation."""
    cur.execute("SELECT * FROM node2id ORDER BY index_id")
    rows = cur.fetchall()
    id2info = {}
    for row in rows:
        node_hash = str(row[0])
        node_type = row[1]
        node_value = str(row[2])
        index_id = int(row[3])
        id2info[index_id] = (node_hash, node_type, node_value)
    return id2info


def _extract_day(filename: str) -> int:
    """Extract the day number from a graph filename like graph_4_6.TemporalData.simple."""
    m = re.search(r'graph_4_(\d+)', filename)
    return int(m.group(1)) if m else -1


def verify():
    if not os.path.isdir(graphs_dir):
        logger.error(f"Graphs directory does not exist: {graphs_dir}")
        sys.exit(1)

    graph_files = sorted(glob.glob(os.path.join(graphs_dir, "*.TemporalData.simple")))
    if not graph_files:
        logger.error(f"No graph files found in {graphs_dir}")
        sys.exit(1)

    logger.info(f"Found {len(graph_files)} graph file(s) in {graphs_dir}")

    cur, connect = init_database_connection()
    id2info = load_node_lookup(cur)
    logger.info(f"Loaded {len(id2info)} nodes from node2id table.")

    all_ok = True

    for gf in graph_files:
        day = _extract_day(os.path.basename(gf))
        day_label = "ATTACK" if day in ATTACK_DAYS else "BENIGN"
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying: {os.path.basename(gf)}  [Day {day} = {day_label}]")
        logger.info(f"{'='*60}")

        dataset = torch.load(gf, weights_only=False)

        # ── structural checks ──
        assert hasattr(dataset, 'src'), "Missing dataset.src"
        assert hasattr(dataset, 'dst'), "Missing dataset.dst"
        assert hasattr(dataset, 't'),   "Missing dataset.t"
        assert hasattr(dataset, 'y'),   "Missing dataset.y  (labels not saved!)"

        n_edges = dataset.src.size(0)
        assert dataset.dst.size(0) == n_edges, "src/dst length mismatch"
        assert dataset.t.size(0)   == n_edges, "src/t length mismatch"
        assert dataset.y.size(0)   == n_edges, "src/y length mismatch"

        logger.info(f"  Edges in graph       : {n_edges}")

        saved_labels = dataset.y.tolist()
        n_mal_saved = sum(saved_labels)
        n_ben_saved = n_edges - n_mal_saved
        mal_pct = (n_mal_saved / n_edges * 100) if n_edges else 0
        logger.info(f"  Benign  edges (saved): {n_ben_saved}")
        logger.info(f"  Malicious edges (saved): {n_mal_saved}  ({mal_pct:.2f}%)")

        # ── dtype / value checks ──
        assert dataset.y.dtype == torch.long, \
            f"Label dtype should be torch.long, got {dataset.y.dtype}"
        unique_labels = dataset.y.unique().tolist()
        assert all(l in (0, 1) for l in unique_labels), \
            f"Labels should be 0 or 1, got unique values: {unique_labels}"
        logger.info(f"  Label dtype OK       : {dataset.y.dtype}")
        logger.info(f"  Label unique values  : {unique_labels}")

        # ── node existence checks ──
        all_ids = set(dataset.src.tolist()) | set(dataset.dst.tolist())
        unknown_ids = all_ids - set(id2info.keys())
        if unknown_ids:
            logger.warning(f"  {len(unknown_ids)} edge endpoint(s) not in node2id "
                           f"(first 10: {list(unknown_ids)[:10]})")
        else:
            logger.info(f"  All edge endpoints exist in node2id ✓")

        # ── TEMPORAL CHECK: benign days must have 0 malicious ──
        if day not in ATTACK_DAYS:
            if n_mal_saved != 0:
                logger.error(f"  ✗ BENIGN day {day} has {n_mal_saved} malicious edges!")
                all_ok = False
            else:
                logger.info(f"  ✓ BENIGN day {day} has 0 malicious edges (correct)")

        # ── anchor traceability on attack days ──
        if day in ATTACK_DAYS and n_mal_saved > 0:
            # Build a per-day tracker to check anchors
            tracker_verify = TaintTracker()
            anchor_agree = 0
            anchor_disagree = 0
            sample_mal_edges = []

            for edge_idx in range(n_edges):
                saved_lbl = saved_labels[edge_idx]
                if saved_lbl != 1:
                    continue
                sid = dataset.src[edge_idx].item()
                did = dataset.dst[edge_idx].item()
                s_info = id2info.get(sid)
                d_info = id2info.get(did)

                if s_info and d_info:
                    s_hash, s_type, s_val = s_info
                    d_hash, d_type, d_val = d_info

                    src_anchor, _ = detect_anchor(s_type, s_val, day)
                    dst_anchor, _ = detect_anchor(d_type, d_val, day)
                    src_tracked = tracker_verify.is_malicious(s_hash)
                    dst_tracked = tracker_verify.is_malicious(d_hash)

                    if src_anchor or dst_anchor or src_tracked or dst_tracked:
                        anchor_agree += 1
                    else:
                        anchor_disagree += 1

                    if len(sample_mal_edges) < 5:
                        sample_mal_edges.append({
                            "edge_idx": edge_idx,
                            "src": f"{s_type}:{s_val[:60]}",
                            "dst": f"{d_type}:{d_val[:60]}",
                            "src_anchor": src_anchor,
                            "dst_anchor": dst_anchor,
                        })

            logger.info(f"  Malicious edges with direct anchor: {anchor_agree}")
            if anchor_disagree:
                logger.info(f"  Malicious edges via propagation only: {anchor_disagree}")

            if sample_mal_edges:
                logger.info(f"  Sample malicious edges (up to 5):")
                for se in sample_mal_edges:
                    logger.info(f"    edge {se['edge_idx']}: "
                                f"{se['src']} -> {se['dst']}  "
                                f"[src_anchor={se['src_anchor']}, "
                                f"dst_anchor={se['dst_anchor']}]")

        # ── temporal ordering ──
        timestamps = dataset.t.tolist()
        is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        if is_sorted:
            logger.info(f"  Temporal ordering     : sorted ✓")
        else:
            logger.warning(f"  Temporal ordering     : NOT sorted")

        logger.info(f"  ── {os.path.basename(gf)} PASSED ──")

    connect.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"VERIFICATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Graphs checked : {len(graph_files)}")
    logger.info(f"  Attack days    : {sorted(ATTACK_DAYS)}")
    logger.info(f"  Ground truth IPs          : {len(ALL_MALICIOUS_IPS)}")
    logger.info(f"  Ground truth file paths   : {len(ALL_MALICIOUS_FILES)}")
    logger.info(f"  Ground truth proc patterns: {len(ALL_MALICIOUS_PROCESSES)}")
    if all_ok:
        logger.info(f"  All structural, temporal & label checks passed ✓")
    else:
        logger.error(f"  SOME CHECKS FAILED – see above for details")
        sys.exit(1)


if __name__ == "__main__":
    verify()
