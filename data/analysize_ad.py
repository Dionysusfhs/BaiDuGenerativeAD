from tqdm import tqdm

def create_token_mapping():
    # 初始化特殊token
    PAD, UNK = 0, 1
    token2idx = {'<pad>': PAD, '<unk>': UNK}
    
    # 用于收集所有unique tokens
    unique_tokens = set()
    
    print("第一步：收集所有unique tokens...")
    # 首先统计文件总行数
    total_lines = sum(1 for _ in open('data/data_n_core/ad_data_filtered', 'r'))
    
    # 读取ad_data文件并收集所有token
    with open('data/data_n_core/ad_data_filtered', 'r') as f:
        for line in tqdm(f, total=total_lines, desc="收集tokens"):
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            
            content = fields[1]
            token_ids = [int(x) for x in content.split(',')]
            unique_tokens.update(token_ids)
    
    print(f"\n找到 {len(unique_tokens)} 个unique tokens")
    
    # 创建映射字典
    print("第二步：创建token到index的映射...")
    # 先写入特殊token
    with open('data/token2idx.txt', 'w') as f:
        f.write('<pad>\t0\n')  # PAD token
        f.write('<unk>\t1\n')  # UNK token
        # 对其他token排序后写入
        for token in tqdm(sorted(unique_tokens), desc="保存映射"):
            idx = len(token2idx)
            token2idx[token] = idx
            f.write(f"{token}\t{idx}\n")
    
    # 打印统计信息
    print(f"\n映射统计信息：")
    print(f"词表大小: {len(token2idx)}")
    print(f"特殊token: PAD={PAD}, UNK={UNK}")
    
    # 验证保存的映射
    print("\n验证保存的映射...")
    loaded_map = {}
    with open('data/token2idx.txt', 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            loaded_map[token] = int(idx)
    print(f"加载的映射大小: {len(loaded_map)}")
    print("验证成功！")
    
def token_difference():
    # 加载过滤后的token映射
    filtered_tokens = {}
    print("加载过滤后的token映射...")
    with open('data/data_n_core/token2idx.txt', 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            if token not in ['<pad>', '<unk>']:  # 跳过特殊token
                filtered_tokens[int(token)] = int(idx)
    
    # 统计变量
    missing_tokens = set()
    total_tokens = 0
    missing_count = 0
    affected_ads = 0
    
    # 分析原始数据集
    print("分析原始数据集中的token缺失情况...")
    total_lines = sum(1 for _ in open('data/ad_data', 'r'))
    with open('data/ad_data', 'r') as f:
        for line in tqdm(f, total=total_lines, desc="分析原始数据"):
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            
            content = fields[1]
            token_ids = [int(x) for x in content.split(',')]
            
            # 统计当前广告的缺失情况
            current_missing = [t for t in token_ids if t not in filtered_tokens]
            
            total_tokens += len(token_ids)
            if current_missing:
                affected_ads += 1
                missing_count += len(current_missing)
                missing_tokens.update(current_missing)
    
    # 打印统计结果
    print("\n缺失统计信息：")
    print(f"原始数据集中总token数: {total_tokens}")
    print(f"缺失的unique token数: {len(missing_tokens)}")
    print(f"缺失token总次数: {missing_count}")
    print(f"受影响的广告数: {affected_ads}")
    print(f"token缺失率: {missing_count/total_tokens*100:.2f}%")
    print(f"广告影响率: {affected_ads/total_lines*100:.2f}%")
    
    # 输出一些缺失token的例子
    if missing_tokens:
        print("\n缺失token示例（最多显示10个）:")
        for token in sorted(list(missing_tokens))[:10]:
            print(f"Token ID: {token}")

def token_statistics():
    # 初始化变量
    token_avg_len = 0
    token_max_len = float('-inf')
    token_min_len = float('inf')
    ad_ctr = 0
    max_token_id = float('-inf')
    min_token_id = float('inf')
    unique_tokens = set()
    
    # 读取ad_data文件
    with open('data/data_n_core/ad_data_filtered', 'r') as f:
        for line in f:
            # 分割每行数据
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
                
            # 获取content字段（第二列）
            content = fields[1]
            # 分割token ids
            token_ids = [int(x) for x in content.split(',')]
            token_avg_len += len(token_ids)
            ad_ctr += 1
            token_max_len = max(token_max_len, len(token_ids))
            token_min_len = min(token_min_len, len(token_ids))
            # 更新统计信息
            max_token_id = max(max_token_id, max(token_ids))
            min_token_id = min(min_token_id, min(token_ids))
            unique_tokens.update(token_ids)
            
    token_avg_len /= ad_ctr
    
    print(f"Token ID 的统计信息：")
    print(f"最大token ID: {max_token_id}")
    print(f"最小token ID: {min_token_id}")
    print(f"不同token的总数: {len(unique_tokens)}")
    print(f"token平均长度: {token_avg_len}")
    print(f"token最大长度: {token_max_len}")
    print(f"token最小长度: {token_min_len}")

if __name__ == "__main__":
    # create_token_mapping()
    # token_statistics()
    token_difference()