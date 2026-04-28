import numpy as np
from typing import List, Tuple

def calculate_recall_at_k(ground_truth: int, predictions: List[int], k: int = 10) -> float:
    """
    计算Recall@K
    Args:
        ground_truth: 实际点击的广告ID
        predictions: 预测的广告ID列表
        k: 取前k个预测结果
    Returns:
        Recall@K值
    """
    if ground_truth in predictions[:k]:
        return 1.0
    return 0.0

def calculate_ndcg_at_k(ground_truth: int, predictions: List[int], k: int = 10) -> float:
    """
    计算NDCG@K
    Args:
        ground_truth: 实际点击的广告ID
        predictions: 预测的广告ID列表
        k: 取前k个预测结果
    Returns:
        NDCG@K值
    """
    dcg = 0.0
    idcg = 1.0  # 理想情况下ground truth在第一位

    for i, pred in enumerate(predictions[:k]):
        if pred == ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2是因为i从0开始，而公式中是从1开始
            break
    
    return dcg / idcg

def evaluate_predictions(ground_truths: List[int], all_predictions: List[List[int]], k: int = 10) -> Tuple[float, float]:
    """
    评估预测结果
    Args:
        ground_truths: 所有用户的实际点击广告ID列表
        all_predictions: 所有用户的预测广告ID列表
        k: 取前k个预测结果
    Returns:
        (平均Recall@K, 平均NDCG@K)
    """
    recalls = []
    ndcgs = []
    
    for gt, preds in zip(ground_truths, all_predictions):
        recall = calculate_recall_at_k(gt, preds, k)
        ndcg = calculate_ndcg_at_k(gt, preds, k)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    
    return avg_recall, avg_ndcg 