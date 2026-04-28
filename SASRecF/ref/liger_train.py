# dataset.py
import paddle
from paddle.io import Dataset
import numpy as np
import random
# train.py
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
from liger_model import LIGERModel
import numpy as np
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm

class Args():
    def __init__(self):
        self.dataset_dir = '../w_data' # data路径
        self.train_file = '1w_train.txt' # 用户交互数据文件
        self.unitid_file = '1w_tokenized_unitid/part-00000' # 广告内容
        self.seq_len = 20 # 交互序列长度
        self.emb_size = 1024 # 广告维度大小
        self.sid_size= 256 # sid 大小
        self.sid_len=3  # 单个广告 sid 长度
        self.hidden_size=128 #模型隐层维度 
        self.num_heads=4 # 注意力头个数
        self.dropout_rate = 0.2
        self.dim_feedforward = 512 # encoder decoder 中间层维度
        self.num_layers = 2 # encoder 和decoder层数
        self.loss_temperature = 0.7 # loss/=temperature

class Liger_Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.unitid_data = self.load_ad_data()
        self.data = []

        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f, desc="Loading sequence data"):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                user_id, ad_seq = parts
                ad_ids = list(map(int, ad_seq.strip().split()))
                if len(ad_ids) <= 1:
                    continue

                ad_seq_ids = ad_ids[:-1]   # 输入序列
                next_ad_id = ad_ids[-1]    # 预测目标

                # 过滤掉不在 unitid_data 中的广告
                ad_seq_ids = [aid for aid in ad_seq_ids if aid in self.unitid_data]
                if len(ad_seq_ids) == 0 or next_ad_id not in self.unitid_data:
                    continue

                # 构造 text_reps: embedding
                text_reps = [self.unitid_data[aid]['embedding'] for aid in ad_seq_ids]
                sid_seq = [self.unitid_data[aid]['token'] for aid in ad_seq_ids]

                if len(text_reps) > args.seq_len:
                    text_reps = text_reps[-args.seq_len:]
                    sid_seq = sid_seq[-args.seq_len:]

                # padding 到固定长度
                text_reps = self.pad_2d(text_reps, args.seq_len, args.emb_size)
                sid_seq = self.pad_2d(sid_seq, args.seq_len, args.sid_len, pad_value=0)

                # next_text & next_sid
                next_text = self.unitid_data[next_ad_id]['embedding']
                next_sid = self.unitid_data[next_ad_id]['token']
                if len(next_sid) < args.sid_len + 1:
                    next_sid = next_sid + [0] * (args.sid_len + 1 - len(next_sid))
                else:
                    next_sid = next_sid[:args.sid_len + 1]

                self.data.append({
                    "sid_seq": sid_seq,
                    "text_reps": text_reps,
                    "pos_ids": list(range(args.seq_len)),
                    "next_text": next_text,
                    "next_sid": next_sid
                })

    def pad_2d(self, data, max_len, dim, pad_value=0.0):
        padded = []
        for row in data:
            if len(row) < dim:
                row = row + [pad_value] * (dim - len(row))
            padded.append(row)
        while len(padded) < max_len:
            padded.append([pad_value] * dim)
        return padded

    def load_ad_data(self):
        ad_map = {}
        file_path = f"{self.args.dataset_dir}/{self.args.unitid_file}"
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Loading ad_data"):
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                ad_id = int(parts[0])
                token = list(map(int, parts[1].strip().split(',')))
                emb = list(map(float, parts[2].strip().split(',')))
                ad_map[ad_id] = {'token': token, 'embedding': emb}
        return ad_map

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class MyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=2, sid_len=3, text_dim=1024, vocab_size=256):
        self.data = []
        for _ in range(num_samples):
            sid_seq = np.random.randint(0, vocab_size, size=(seq_len, sid_len))          # [T, m]
            text_reps = np.random.randn(seq_len, text_dim).astype('float32')             # [T, 768]
            pos_ids = np.arange(seq_len)                                                 # [T]
            next_text = np.random.randn(text_dim).astype('float32')                      # [768]
            next_sid = np.random.randint(0, vocab_size, size=(sid_len + 1,))             # [m+1]

            self.data.append({
                "sid_seq": sid_seq,
                "text_reps": text_reps,
                "pos_ids": pos_ids,
                "next_text": next_text,
                "next_sid": next_sid
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    sid_seq = paddle.to_tensor([item['sid_seq'] for item in batch], dtype='int64')          # [B, T, m]
    text_reps = paddle.to_tensor([item['text_reps'] for item in batch], dtype='float32')    # [B, T, 768]
    item_pos_ids = paddle.to_tensor([item['pos_ids'] for item in batch], dtype='int64')     # [B, T]
    next_text = paddle.to_tensor([item['next_text'] for item in batch], dtype='float32')    # [B, 768]
    next_sid = paddle.to_tensor([item['next_sid'] for item in batch], dtype='int64')        # [B, m+1]
    return sid_seq, text_reps, item_pos_ids, next_text, next_sid


def train(args):
    # Data loader
    # dataset = MyDataset(num_samples=1000)
    dataset = Liger_Dataset(args)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Model
    model = LIGERModel(args)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=3e-4, weight_decay=0.01)

    # Checkpoint logic
    # os.makedirs('checkpoint', exist_ok=True)
    # if os.path.exists('checkpoint/current.pdparams'):
    #     model.set_state_dict(paddle.load('checkpoint/current.pdparams'))
    #     optimizer.set_state_dict(paddle.load('checkpoint/current.pdopt'))
    #     print("✅ Loaded checkpoint from 'checkpoint/current.pdparams'")
    # else:
    #     print("🆕 No checkpoint found, loading pretrained T5.")

    best_loss = float('inf')

    # Training loop
    model.train()
    for epoch in range(10):
        for step, batch in enumerate(loader):
            sid_seq, text_reps, item_pos_ids, next_text, next_sid = batch
            loss = model(sid_seq, text_reps, item_pos_ids, next_text, next_sid)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            print(f"[Epoch {epoch}] [Step {step}] Loss = {loss.item():.4f}")

            # Save checkpoints
            paddle.save(model.state_dict(), 'checkpoint/current.pdparams')
            paddle.save(optimizer.state_dict(), 'checkpoint/current.pdopt')

            if loss.item() < best_loss:
                best_loss = loss.item()
                paddle.save(model.state_dict(), 'checkpoint/best.pdparams')
                paddle.save(optimizer.state_dict(), 'checkpoint/best.pdopt')
                print(f"✅ Best model updated: loss = {best_loss:.4f}")

if __name__ == '__main__':
    now_time = datetime.now()
    day = now_time.strftime("%Y_%m_%d_%H_%M") # 2023_10_05_10_00
    folder = "./" + "checkpoints" + "/" + day 
    if not os.path.exists(folder):
        os.makedirs(folder)

    args = Args()
    with open(os.path.join(folder, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    train(args)