import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
from tqdm import tqdm
import random
import numpy as np
import pdb
import os 
import time
def read_data(f,unitid_data):
    with open(f, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            ad_id = int(parts[0])
            embedding = list(map(np.float32, parts[2].split(',')))
            unitid_data[ad_id] = {'embedding': embedding}

def safe_process_file(f,unitid_data):
    try:
        read_data(f,unitid_data)
    except Exception as e:
        print(e)

class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.unitid_data,self.lenth_unit_data = self.load_unit()
        self.train_data = []
        
        # 读取train.txt文件
        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split()))[:-1] # 将训练数据最后一位暂时当测试数据
                ad_filter_ids = []
                for ad_id in ad_ids:
                    if ad_id in self.unitid_data:
                        ad_filter_ids.append(ad_id)
                    else:
                        print(f"{ad_id} not in unit_map")
                self.train_data.append({'ad_ids': ad_filter_ids})
        print(f"train.txt loaded sucessfully ,{len(self.train_data)}")
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx]
        ad_ids = sample['ad_ids']
        
        ad_embeddings = []
        for idx,ad_id in enumerate(ad_ids):
            if ad_id in self.unitid_data:
                ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
            else:
                print(f"{ad_id} not in unit_map")
        if len(ad_embeddings) > self.args.maxlen:
            ad_embeddings = ad_embeddings[-self.args.maxlen:]  
            ad_ids = ad_ids[-self.args.maxlen:]
        return ad_embeddings, ad_ids
    
    def load_unit(self):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        unitid_data = {}
        f = f"{self.args.dataset_dir}/{self.args.unitid_file}"
        safe_process_file(f,unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data,lenth_unit_data

    def load_unit_w(self):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        root_dir = f"{self.args.dataset_dir}/1w_tokenized_unitid"
        file_list = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files]
        unitid_data = {}
        for f in file_list:
            safe_process_file(f,unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data,lenth_unit_data
    
    def collate_fn(self,batch):
        # 假设batch中的每个元素是一个元组 (user_id, ad_embeddings)
        ad_embeddings,ad_ids = zip(*batch)
        # 找到最长的ad_embeddings长度
        max_len = max(len(emb[:-1]) for emb in ad_embeddings)
        
        # 初始化填充后的ad_embeddings和pad_mask
        padded_embeddings = []
        pad_mask = []
        padded_pos_embs = []
        padded_neg_embs = []
        padded_ad_ids = []
        
        for idx,emb in enumerate(ad_embeddings):
            emb_len = len(emb[:-1])
            ad_ids_vector = paddle.to_tensor(ad_ids[idx][1:],dtype='float32')
            padding_len = max_len - emb_len
            if padding_len:
                # 使用全零填充向量，与emb具有相同的维度
                padding_vector = paddle.zeros([padding_len, self.args.emb_dim], dtype='float32')
                
                # 随机初始化填充向量，与emb具有相同的维度
                # padding_vector = paddle.randn([padding_len, self.args.emb_dim],dtype='float32')
                padding_ad_vector = paddle.full([padding_len],0,dtype='float32') # pad id 0
                # 拼接原始embedding和填充向量
                padded_emb = paddle.concat([padding_vector,paddle.to_tensor(emb[:-1],dtype='float32')], axis=0)
                padded_ad_ids_vector = paddle.concat([padding_ad_vector,ad_ids_vector],axis=0)
            else:
                padded_ad_ids_vector = ad_ids_vector
                padded_emb = paddle.to_tensor(emb[:-1],dtype='float32')
            padded_ad_ids.append(padded_ad_ids_vector)
            # 创建pad_mask，1表示原始数据，0表示填充数据
            mask = paddle.ones([max_len], dtype='float32')
            mask[:padding_len] = 0
            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)
            if padding_len:
                padded_pos_emb = paddle.concat([padding_vector,paddle.to_tensor(emb[1:],dtype='float32')], axis=0)
            else:
                padded_pos_emb = paddle.to_tensor(emb[1:],dtype='float32')
            unit_data_map_ids = list(self.unitid_data.keys())
            random_neg_ids = self.generate_random_numbers(0, self.lenth_unit_data-1, [ad_ids[idx][1:]], emb_len)
            # 随机负例
            random_neg_emb = paddle.to_tensor([self.unitid_data[unit_data_map_ids[i]]['embedding'] for i in random_neg_ids],dtype='float32')
            if padding_len:
                padded_neg_emb = paddle.concat([padding_vector,random_neg_emb], axis=0)
            else:
                padded_neg_emb = random_neg_emb

            padded_pos_embs.append(padded_pos_emb)
            padded_neg_embs.append(padded_neg_emb)
        
        padded_embeddings = paddle.stack(padded_embeddings, axis=0)
        pad_mask = paddle.stack(pad_mask, axis=0)
        padded_pos_embeddings = paddle.stack(padded_pos_embs,axis=0)
        padded_neg_embeddings = paddle.stack(padded_neg_embs,axis=0)
        padded_ad_ids = paddle.stack(padded_ad_ids,axis=0)
        return padded_embeddings, padded_pos_embeddings, padded_neg_embeddings, pad_mask, padded_ad_ids # ad_ids

    def generate_random_numbers(self,start, end, exceptions, count):
        """
        生成count个在[start, end]范围内的随机数，但不能是exceptions列表中的值。
        
        :param start: 随机数的起始范围
        :param end: 随机数的结束范围
        :param exceptions: 不能生成的数值列表
        :param count: 需要生成的随机数数量
        :return: 生成的随机数列表
        """
        random_numbers = []
        while len(random_numbers) < count:
            num = random.randint(start, end)
            if num not in exceptions:
                random_numbers.append(num)
        return random_numbers

class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.gt_data = []
        self.test_data = []
        self.unitid_data,self.lenth_unit_data = self.load_unit()

        # 读取infer_data
        cnt = 0
        self.u2id = {}
        self.id2u = {}
        with open(f"{args.input_path}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split())) # 将训练数据最后一位暂时当测试数据
                # self.test_data.append({'user_id': cnt,'ad_ids': ad_ids[:-1]}) #去除最后一个，本地评测～！
                self.test_data.append({'user_id': cnt,'ad_ids': ad_ids}) #不去除最后一个，提交系统评测～！
                self.u2id[parts[0].split("|")[0]] = cnt
                self.id2u[cnt] = parts[0].split("|")[0]
                # self.gt_data.append(ad_filter_ids[-1])
                cnt += 1
        print(f"test data loaded sucessfully ,{len(self.test_data)}")

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        sample = self.test_data[idx]
        user_id = sample["user_id"]
        ad_ids = sample["ad_ids"]
        ad_embeddings = []
        for ad_id in ad_ids:
            if ad_id in self.unitid_data:
                ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
            else:
                print(f"{ad_id} not in unit_map")
        if len(ad_embeddings) > self.args.maxlen:
            ad_embeddings = ad_embeddings[-self.args.maxlen:]  
        return user_id, ad_embeddings
    
    def load_unit(self):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        unitid_data = {}
        f = f"{self.args.dataset_dir}/ad_data"
        safe_process_file(f,unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data,lenth_unit_data
    
    def collate_fn(self,batch):
        # 假设batch中的每个元素是一个元组 (user_id, ad_embeddings)
        user_ids,ad_embeddings = zip(*batch)
        # gt_embeddings = paddle.to_tensor(gt_embeddings,dtype='float32')
        # 找到最长的ad_embeddings长度
        max_len = max(len(emb) for emb in ad_embeddings)
            
        # 初始化填充后的ad_embeddings和pad_mask
        padded_embeddings = []
        pad_mask = []
            
        for emb in ad_embeddings:
            emb_len = len(emb)
            padding_len = max_len - emb_len
                
            # 全零初始化填充向量，与emb具有相同的维度
            padding_vector = paddle.zeros([padding_len, self.args.emb_dim],dtype='float32')
                
            # 拼接原始embedding和填充向量
            if padding_len:
                padded_emb = paddle.concat([padding_vector,paddle.to_tensor(emb,dtype='float32')], axis=0)
            else:
                padded_emb = paddle.to_tensor(emb,dtype='float32')
                
            # 创建pad_mask，1表示原始数据，0表示填充数据
            mask = paddle.ones([max_len], dtype='float32')
            mask[:padding_len] = 0
                
            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)
            
        padded_embeddings = paddle.stack(padded_embeddings, axis=0)
        pad_mask = paddle.stack(pad_mask, axis=0)

        return paddle.to_tensor(user_ids,dtype='int64'),padded_embeddings, pad_mask
