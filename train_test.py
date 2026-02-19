import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, average_precision_score,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import *
from logger import logger

from model import GINEEdgeModel


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, model_path="./artifact/models/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(model.state_dict(), self.model_path)
        else:
            self.counter += 1

        return self.counter >= self.patience

    def load_best(self, model):
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        logger.info(f"Loaded best model from epoch {self.best_epoch} (val_loss={self.best_loss:.4f})")
        return model

def balance_edges(data, benign_ratio=benign_ratio):
    data = data.clone()

    edge_y = data.edge_y
    malicious_idx = (edge_y == 1).nonzero(as_tuple=True)[0]
    benign_idx = (edge_y == 0).nonzero(as_tuple=True)[0]

    num_mal = len(malicious_idx)
    num_benign_sample = min(num_mal * benign_ratio, len(benign_idx))

    sampled_benign = benign_idx[
        torch.randperm(len(benign_idx))[:num_benign_sample]
    ]

    selected_idx = torch.cat([malicious_idx, sampled_benign])
    selected_idx = selected_idx[torch.randperm(len(selected_idx))]

    data.edge_index = data.edge_index[:, selected_idx]
    data.edge_relation = data.edge_relation[selected_idx]
    data.edge_y = data.edge_y[selected_idx]

    return data


def evaluate(model, data, device):

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(
            data.x,
            data.edge_index,
            data.edge_relation
        )

    probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
    labels = data.edge_y.cpu().numpy()

    auc = roc_auc_score(labels, probs)

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    preds = (probs >= best_threshold).astype(int)

    return auc, best_f1, best_threshold, preds, probs, labels


def evaluate_aggregate(model, graphs, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for g in graphs:
            data = g.to(device)
            out = model(data.x, data.edge_index, data.edge_relation)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(data.edge_y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    preds = (all_probs >= best_threshold).astype(int)
    return auc, best_f1, best_threshold, preds, all_probs, all_labels


def plot_day_analysis(d, labels, probs, preds, auc, f1, threshold, plots_dir):
    ap = average_precision_score(labels, probs)
    precision, recall, pr_thresh = precision_recall_curve(labels, probs)
    fpr, tpr, roc_thresh = roc_curve(labels, probs)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Day {d}  |  AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  threshold={threshold:.3f}", fontsize=11)

    ax = axes[0]
    ax.plot(recall, precision, color='steelblue', lw=2)
    ax.axvline(recall[np.argmax(2*precision*recall/(precision+recall+1e-8))],
               color='red', linestyle='--', lw=1, label=f'Best thresh={threshold:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve  (AP={ap:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc:.4f}')
    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign","Malicious"]).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f'Confusion Matrix\nTP={tp} FP={fp} FN={fn} TN={tn}')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"analysis_day{d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path, cm


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_graph = [
        torch.load(f"./artifact/graphs/graph_day{d}.pt", weights_only=False)
        for d in train_days
    ]
    test_graphs = [
        torch.load(f"./artifact/graphs/graph_day{d}.pt", weights_only=False)
        for d in test_days
    ]

    for i in range(len(train_graph)):
        train_graph[i] = train_graph[i].cpu()

    for i in range(len(test_graphs)):
        test_graphs[i] = test_graphs[i].cpu()

    train_graph = [balance_edges(g) for g in train_graph]

    val_graph = balance_edges(test_graphs[0]).to(device)

    model = GINEEdgeModel(node_in_dim=387).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    os.makedirs("./artifact", exist_ok=True)

    logger.info(f"Starting training | epochs={epochs} | train_days={train_days} | test_days={test_days} | device={device}")
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for g in train_graph:
            data = g.to(device)

            out = model(
                data.x,
                data.edge_index,
                data.edge_relation
            )

            loss = criterion(out, data.edge_y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_graph)

        model.eval()
        with torch.no_grad():
            val_out = model(val_graph.x, val_graph.edge_index, val_graph.edge_relation)
            val_loss = criterion(val_out, val_graph.edge_y).item()

        train_auc, train_f1, _, _, _, _ = evaluate_aggregate(model, train_graph, device)
        val_auc, val_f1, _, _, _, _ = evaluate(model, val_graph, device)

        logger.info(
            f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"train_auc={train_auc:.4f}  train_f1={train_f1:.4f} | "
            f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f} | "
            f"patience={early_stopping.counter}/{early_stopping.patience}"
        )

        if early_stopping.step(val_loss, model, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    model = early_stopping.load_best(model)

    os.makedirs("./artifact/plots", exist_ok=True)

    logger.info("Starting evaluation on test days")
    all_test_labels = []
    all_test_preds = []

    for d, test_graph in zip(test_days, test_graphs):

        test_graph_bal = balance_edges(test_graph, benign_ratio=benign_ratio)
        auc, f1, threshold, preds, probs, labels = evaluate(model, test_graph_bal, device)

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        logger.info("=" * 60)
        logger.info(f"Test Day {d} | AUC={auc:.4f}  F1={f1:.4f}  threshold={threshold:.3f}")
        logger.info(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        report = classification_report(labels, preds, target_names=["Benign", "Malicious"], digits=4)
        logger.info(f"Classification Report — Day {d}:\n{report}")

        all_test_labels.append(labels)
        all_test_preds.append(preds)

        save_path, _ = plot_day_analysis(
            d, labels, probs, preds, auc, f1, threshold,
            plots_dir="./artifact/plots"
        )
        logger.info(f"Saved analysis plot to {save_path}")

    combined_labels = np.concatenate(all_test_labels)
    combined_preds = np.concatenate(all_test_preds)
    logger.info("=" * 60)
    logger.info("Overall Classification Report (All Test Days Combined):")
    overall_report = classification_report(combined_labels, combined_preds, target_names=["Benign", "Malicious"], digits=4)
    logger.info(f"\n{overall_report}")


