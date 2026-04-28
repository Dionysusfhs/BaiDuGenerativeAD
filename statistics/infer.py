from typing import List, Tuple
from tqdm import tqdm
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate.metrics import evaluate_predictions

def load_top_ads(top_ads_file: str, num_items: int = 10) -> List[int]:
    """
    加载排名前N的广告ID
    Args:
        top_ads_file: top_200_ads.txt的路径
        num_items: 需要返回的广告数量
    Returns:
        前N个广告ID列表
    """
    top_ads = []
    with open(top_ads_file, 'r') as f:
        # 跳过前两行（标题和分隔线）
        next(f)
        next(f)
        # 读取前N个广告ID
        for _ in range(num_items):
            line = next(f)
            ad_id = int(line.strip().split('\t')[1])
            top_ads.append(ad_id)
    return top_ads

def process_users(input_path: str, output_path: str, top_ads: List[int]) -> Tuple[List[int], List[List[int]]]:
    """
    为每个用户生成推荐列表并写入文件，同时返回ground truth和预测结果
    Args:
        input_path: 输入文件路径（用户序列数据）
        output_path: 输出文件路径（推荐结果）
        top_ads: 要推荐的广告ID列表
    Returns:
        (ground_truths, predictions)元组
    """
    print("\n开始生成推荐列表...")
    ground_truths = []
    predictions = []
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc="处理进度"):
            try:
                # 解析行数据
                parts = line.strip().split('\t')
                user_id = parts[0].split('|')[0]
                
                # 获取序列并提取最后一个广告ID作为ground truth
                sequence = parts[1].split()
                if sequence:  # 确保序列不为空
                    ground_truth = int(sequence[-1])
                    ground_truths.append(ground_truth)
                    predictions.append(top_ads)
                
                # 将推荐列表写入文件
                fout.write(f"{user_id}\t{' '.join(map(str, top_ads))}\n")
            except Exception as e:
                print(f"处理用户时出错：{str(e)}")
                continue
                
    print(f"\n推荐结果已保存到文件：{output_path}")
    return ground_truths, predictions

def run(input_path: str, output_path: str):
    """
    主函数：读取用户数据并生成推荐结果，同时计算评估指标
    Args:
        input_path: 输入文件路径（用户序列数据）
        output_path: 输出文件路径（推荐结果）
    """
    try:
        # 加载前10个最热门广告
        top_ads_file = 'statics/top_200_ads.txt'
        top_10_ads = load_top_ads(top_ads_file, num_items=10)
        print(f"已加载前10个最热门广告：{top_10_ads}")
        
        # 处理用户数据并生成推荐
        ground_truths, predictions = process_users(input_path, output_path, top_10_ads)
        
        # 计算评估指标
        recall, ndcg = evaluate_predictions(ground_truths, predictions)
        print(f"\n评估结果:")
        print(f"Recall@10: {recall:.4f}")
        print(f"NDCG@10: {ndcg:.4f}")
        
    except Exception as e:
        print(f"运行出错：{str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_path = "data/sequence_data"  # 用户序列数据文件路径
    output_path = "statics/recommendations.txt"  # 推荐结果输出路径
    run(input_path, output_path) 