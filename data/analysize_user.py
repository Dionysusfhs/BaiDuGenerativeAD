import numpy as np

def calculate_avg_sequence_length():
    # 存储所有序列的长度
    sequence_lengths = []
    
    # 读取sequence_data文件
    try:
        with open('data/sequence_data', 'r') as f:
            for line in f:
                # 按tab分割获取用户ID和序列
                user_id, sequence = line.strip().split('\t')
                # 计算序列长度（按空格分割后计算元素个数）
                seq_length = len(sequence.split())
                sequence_lengths.append(seq_length)
    except FileNotFoundError:
        print("错误：找不到sequence_data文件，请确保文件位于data目录下")
        return
    
    # 计算统计信息
    avg_length = np.mean(sequence_lengths)
    min_length = np.min(sequence_lengths)
    max_length = np.max(sequence_lengths)
    
    print(f"序列数量: {len(sequence_lengths)}")
    print(f"平均序列长度: {avg_length:.2f}")
    print(f"最短序列长度: {min_length}")
    print(f"最长序列长度: {max_length}")

if __name__ == "__main__":
    calculate_avg_sequence_length()
