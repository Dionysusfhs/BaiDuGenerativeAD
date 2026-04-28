import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
from utils import *
from model import *
from paddle.device import cuda

# 配置文件
class Args():
    def __init__(self):
        self.dataset_dir = "data/data_n_core" # root directory containing the datasets       data[/data_n_core]
        self.unitid_file = "ad_data_filtered"  #ad_data[_filtered]
        self.train_file = "sequence_data_filtered"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.loss_function = "LossWithNegativeSample" # "CustomContrastiveLoss" "MeanSquareError" "CosSimilarity" "LossWithNegativeSample" "BCELossWithNegativeSample"
        self.batch_size = 64
        self.lr = 0.00005
        self.maxlen = 50
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 10
        self.num_heads = 1
        self.dropout_rate = 0.5
        self.device = "gpu"
        self.norm_first = False
        self.inference_only = False
        # 多负样本相关参数
        self.num_neg_samples = 5  # 每个位置的负样本数量
        self.temperature = 1.0    # InfoNCE损失的温度参数
        # self.state_dict_path = "checkpoints/2025_06_01_10_52/SASRec.epoch=1.lr=0.0001.layer=32.head=4.hidden=1024.maxlen=50.pth"
        # self.epoch_start_idx = 2
        self.state_dict_path = None
        
now_time = datetime.now()
day = now_time.strftime("%Y_%m_%d_%H_%M") # 2023_10_05_10_00
folder = "/home/luyimeng/BaiDuGenerativeAD" + "/" + "checkpoints" + "/" + day 
if not os.path.exists(folder):
    os.makedirs(folder)

args = Args()
with open(os.path.join(folder, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

model = SASRec(args).to(args.device) # no ReLU activation in original SASRec implementation?

# 定义一个 XavierNormal 初始化器
xavier_normal_init = paddle.nn.initializer.XavierNormal()

for name, param in model.named_parameters():
    try:
        xavier_normal_init(param, param.block)
    except:
        print(f"{name} xaiver 初始化失败")
        pass  # 忽略初始化失败的层

model.train() # enable model training

epoch_start_idx = 1 if args.state_dict_path is None else args.epoch_start_idx
if args.state_dict_path is not None:
    try:
        model.set_dict(paddle.load(args.state_dict_path))
        print("model state dict loaded.")
    except:  
        print('failed loading state_dicts, pls check file path: ')

print("开始加载训练数据...")
dataset = TrainDataset(args)
# 原始dataloader
# dataloader =  paddle.io.DataLoader(dataset, batch_size=args.batch_size,collate_fn=dataset.collate_fn)
# 修改DataLoader的配置
dataloader = paddle.io.DataLoader(
    dataset, 
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn,
    num_workers=4,  # 添加多个工作进程
    use_buffer_reader=True,
    prefetch_factor=2,  # 预加载因子
    shuffle = True, # 随机采样
) 
print("数据加载完成")

criterion_map = {
    "CustomContrastiveLoss": CustomContrastiveLoss(),
    "MeanSquareError"      : MeanSquareError(),
    "CosSimilarity"        : CosSimilarity(),
    "LossWithNegativeSample": LossWithNegativeSample(
        num_neg_samples=args.num_neg_samples, 
        temperature=args.temperature
    ),
    "BCELossWithNegativeSample": BCELossWithNegativeSample(),
}
criterion = criterion_map[args.loss_function]
    
lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
    learning_rate=args.lr,
    gamma=0.96,
    last_epoch=-1,
    verbose=True
)
adam_optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_scheduler, beta1=0.9, beta2=0.98)
best_val_ndcg, best_val_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
T = 0.0
t0 = time.time()
step = 0
accumulated_step = 0
print("开始训练")
print(f"使用损失函数: {args.loss_function}")
if args.loss_function == "LossWithNegativeSample":
    print(f"负样本数量: {args.num_neg_samples}, 温度参数: {args.temperature}")

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # just to decrease identition
    for step, (padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids) in enumerate(tqdm(dataloader)):
        logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
        
        # 根据损失函数类型调用不同的forward方法
        if args.loss_function == "LossWithNegativeSample":
            # 多负样本损失: padded_neg_emb shape is [batch_size, seq_len, num_neg_samples, dim]
            loss = criterion(logits, padded_pos_emb, padded_neg_emb, pad_mask)
        elif args.loss_function == "BCELossWithNegativeSample":
            # BCE损失: 需要处理多负样本的情况，取第一个负样本或做平均
            if len(padded_neg_emb.shape) == 4:  # [B, S, num_neg, D]
                padded_neg_emb = padded_neg_emb[:, :, 0, :]  # 只取第一个负样本
            loss = criterion(logits, padded_pos_emb, pad_mask, padded_neg_emb)
        elif args.loss_function == "CustomContrastiveLoss":
            loss = criterion(logits, padded_pos_emb, pad_mask, ad_ids)
        else:
            # 其他损失函数
            loss = criterion(logits, padded_pos_emb, pad_mask)
            
        loss.backward()
        adam_optimizer.step()
        adam_optimizer.clear_grad()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        step += 1
    lr_scheduler.step()
    # 打印当前学习率
    print(f'Epoch {epoch}, Current learning rate: {adam_optimizer.get_lr()}')
    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    paddle.save(model.state_dict(), os.path.join(folder, fname))
    t0 = time.time()
    model.train()
print("Done")