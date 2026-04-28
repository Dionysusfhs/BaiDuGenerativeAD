import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
from utils import *
from model import *
import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse

# 配置文件
class Args():
    def __init__(self):
        self.dataset_dir = "data" # root directory containing the datasets
        self.unitid_file = "ad_data"
        self.train_file = "sequence_data"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.token2idx_file = "token2idx.txt"
        self.tokenseq_maxlen = 50
        self.idx_len = 0
        self.batch_size = 512
        self.lr = 0.0005
        self.maxlen = 50
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.sid_emb_dim = 256
        self.num_blocks = 2
        self.num_epochs = 3
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "gpu"
        self.inference_only = True
        self.norm_first = False
        self.state_dict_path = "/home/luyimeng/BaiDuGenerativeAD/SASRecF/code/SASRec.epoch=1.lr=5e-05.layer=2.head=1.hidden=1024.maxlen=50.pth"

def infer(model, args, dataset, dataloader):
    # 全库embedding
    item_embs = paddle.to_tensor([v["embedding"] for k,v in dataset.unitid_data.items()]) 
    id2item = dict(zip([i for i in range(dataset.lenth_unit_data)],list(dataset.unitid_data.keys())))
    sf = paddle.nn.Softmax()
    with paddle.no_grad():
        with open(args.output_path,"w") as f:
            for user_ids,padded_embeddings, pad_mask, ad_tokenidx_seqs in tqdm(dataloader):
                logits = model.predict(padded_embeddings,pad_mask,item_embs,ad_tokenidx_seqs) # 全库检索
                probs = sf(logits)
                topk_values, topk_indices = paddle.topk(probs, k=10, axis=-1)
                for idx in range(topk_indices.shape[0]):
                    items = []
                    for jdx in range(topk_indices.shape[1]):
                        items.append(id2item[int(topk_indices[idx,jdx])]) # 全库检索
                    temp = [dataset.id2u[int(user_ids[idx])]," ".join([str(i) for i in items])]
                    f.write("\t".join(temp))
                    f.write("\n")
        print("Done")

def run(input_path, output_path):
    """ 
    评估脚本会回调 infer.run(input_path, output_path) 生成结果
    需要从input_path读入输入文件，并将结果写入到output_path
    """
    working_root = os.path.dirname(__file__)

    args = Args()
    args.input_path = input_path
    args.output_path = output_path
    args.dataset_dir = os.path.dirname(input_path)

    with open(os.path.join(args.dataset_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    
    print("开始加载测试数据...")
    dataset = TestDataset(args)
    dataloader =  paddle.io.DataLoader(
                                    dataset, 
                                    batch_size=args.batch_size,
                                    collate_fn=dataset.collate_fn) 
    print("数据加载完成")

    model = SASRecF(args).to(args.device) 

    # 定义一个 XavierNormal 初始化器
    xavier_normal_init = paddle.nn.initializer.XavierNormal()

    for name, param in model.named_parameters():
        try:
            xavier_normal_init(param, param.block)
        except:
            print(f"{name} xaiver 初始化失败")
            pass  # 忽略初始化失败的层
    if args.state_dict_path is not None:
        absolute_state_dict_path = os.path.join(working_root, args.state_dict_path)
        try:
            model.set_dict(paddle.load(absolute_state_dict_path))
            print("model state dict loaded.")
        except:  
            print('failed loading state_dicts, pls check file path: {}'.format(absolute_state_dict_path))
            
    model.eval()
    infer(model, args, dataset, dataloader)
    