import os
import sys
import argparse
from typing import List, Tuple
from metrics import evaluate_predictions

def load_ground_truth(input_path: str) -> List[int]:
    """
    从输入文件加载真实标签
    Args:
        input_path: 输入文件路径
    Returns:
        ground_truth列表
    """
    ground_truths = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            sequence = parts[1].split()
            if sequence:  # 确保序列不为空
                ground_truth = int(sequence[-1])
                ground_truths.append(ground_truth)
    return ground_truths

def load_predictions(output_path: str) -> List[List[int]]:
    """
    从输出文件加载预测结果
    Args:
        output_path: 预测结果文件路径
    Returns:
        预测结果列表的列表
    """
    predictions = []
    with open(output_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pred_items = [int(x) for x in parts[1].split()]
                predictions.append(pred_items)
    return predictions

def main():
    parser = argparse.ArgumentParser(description='评估推荐系统结果')
    parser.add_argument('--input_path', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--model', type=str, choices=['sasrec', 'dcrec', 'statistics', 'sasrecf'], required=True, help='使用的模型')
    args = parser.parse_args()

    # 根据选择的模型导入相应的infer模块
    if args.model == 'sasrec':
        sys.path.append('SASRec')
        from infer import run as model_run
    elif args.model == 'dcrec':
        sys.path.append('DCRec')
        from inference import run as model_run
    elif args.model == 'statistics':
        sys.path.append('statistics')
        from infer import run as model_run
    elif args.model == 'sasrecf':
        sys.path.append('SASRecF')
        from infer import run as model_run

    try:
        # 调用模型的run函数生成预测结果
        print(f"\n使用 {args.model} 模型生成预测...")
        model_run(args.input_path, args.output_path)
        
        # 加载真实标签和预测结果
        print("\n加载真实标签和预测结果...")
        ground_truths = load_ground_truth(args.input_path)
        predictions = load_predictions(args.output_path)
        
        # 确保数据长度匹配
        if len(ground_truths) != len(predictions):
            print(f"警告：真实标签数量 ({len(ground_truths)}) 与预测结果数量 ({len(predictions)}) 不匹配")
            min_len = min(len(ground_truths), len(predictions))
            ground_truths = ground_truths[:min_len]
            predictions = predictions[:min_len]
        
        # 计算评估指标
        print("\n计算评估指标...")
        recall, ndcg = evaluate_predictions(ground_truths, predictions)
        
        print(f"\n评估结果:")
        print(f"Recall@10: {recall:.8f}")
        print(f"NDCG@10: {ndcg:.8f}")
        
        # 将结果保存到文件
        metrics_file = f"{os.path.splitext(args.output_path)[0]}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Recall@10: {recall:.4f}\n")
            f.write(f"NDCG@10: {ndcg:.4f}\n")
        print(f"\n评估指标已保存到: {metrics_file}")
        
    except Exception as e:
        print(f"\n运行出错：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
