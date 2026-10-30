import numpy as np
from typing import Dict, List, Set, Union

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cpc_classes: List[str] = None
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics: Micro/Macro/Weighted F1, Precision, Recall,
    Hamming Loss, and Subset Accuracy.
    """
    eps = 1e-12
    # Ensure binary format
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    
    # Per-class precision, recall, f1
    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)
    
    # Micro metrics
    micro_tp = np.sum(tp)
    micro_fp = np.sum(fp)
    micro_fn = np.sum(fn)
    micro_precision = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    
    # Macro metrics
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Weighted metrics
    support = np.sum(y_true, axis=0)
    total_support = np.sum(support) + eps
    weighted_f1 = np.sum(f1_per_class * support) / total_support
    
    # Hamming loss
    hamming_loss = np.mean(y_true != y_pred)
    
    # Subset accuracy (exact match)
    subset_acc = np.mean(np.all(y_true == y_pred, axis=1))
    
    metrics = {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "hamming_loss": float(hamming_loss),
        "subset_accuracy": float(subset_acc)
    }
    
    return metrics


def compute_retrieval_metrics(
    retrieved_rankings: List[List[str]],
    ground_truth_positives: List[Set[str]],
    ks: List[int] = [5, 10, 50]
) -> Dict[str, float]:
    """
    Compute IR metrics across queries: Precision@K, Recall@K, MRR, MAP@K, nDCG@K.
    """
    metrics = {}
    n_queries = len(retrieved_rankings)
    if n_queries == 0:
        return metrics

    for k in ks:
        p_at_k = []
        r_at_k = []
        map_at_k = []
        ndcg_at_k = []

        for rank_list, pos_set in zip(retrieved_rankings, ground_truth_positives):
            if not pos_set:
                continue
            
            top_k = rank_list[:k]
            hits = [1 if doc_id in pos_set else 0 for doc_id in top_k]
            n_hits = sum(hits)

            # Precision@K & Recall@K
            p_at_k.append(n_hits / k)
            r_at_k.append(n_hits / len(pos_set))

            # AP@K
            running_hits = 0
            score_sum = 0.0
            for idx, hit in enumerate(hits):
                if hit:
                    running_hits += 1
                    score_sum += running_hits / (idx + 1)
            ap = score_sum / min(len(pos_set), k)
            map_at_k.append(ap)

            # nDCG@K
            dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(pos_set), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_at_k.append(ndcg)

        metrics[f"precision@{k}"] = float(np.mean(p_at_k)) if p_at_k else 0.0
        metrics[f"recall@{k}"] = float(np.mean(r_at_k)) if r_at_k else 0.0
        metrics[f"map@{k}"] = float(np.mean(map_at_k)) if map_at_k else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_at_k)) if ndcg_at_k else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr_list = []
    for rank_list, pos_set in zip(retrieved_rankings, ground_truth_positives):
        if not pos_set:
            continue
        rr = 0.0
        for idx, doc_id in enumerate(rank_list):
            if doc_id in pos_set:
                rr = 1.0 / (idx + 1)
                break
        mrr_list.append(rr)

    metrics["mrr"] = float(np.mean(mrr_list)) if mrr_list else 0.0
    return metrics
