import paddle
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
import pdb
import paddle.nn as nn
import paddle.nn.functional as F

class CustomContrastiveLoss(nn.Layer):
    def __init__(self):
        super(CustomContrastiveLoss, self).__init__()

    def forward(self, logits, labels, pad_mask, ad_idxs):
        batch_size, seq_len, dim = logits.shape
        logits_flatten = paddle.reshape(logits,[batch_size * seq_len,dim])
        labels_flatten = paddle.reshape(labels,[batch_size * seq_len,dim])
        pad_mask = paddle.reshape(pad_mask,[batch_size * seq_len])
        ad_idxs = paddle.reshape(ad_idxs,[batch_size * seq_len])
        
        # 计算相似度矩阵
        similarity_matrix = paddle.matmul(logits_flatten, labels_flatten, transpose_y=True)
        # mask
        mask = paddle.zeros(shape=[batch_size * seq_len,batch_size * seq_len],dtype='float32')
        mask = paddle.where(pad_mask == 0,mask,paddle.to_tensor(1.0,dtype='float32')) # 纵行
        mask = paddle.where(paddle.expand(pad_mask.unsqueeze(-1),shape=[batch_size * seq_len,batch_size * seq_len]) == 0,paddle.to_tensor(0.0,dtype='float32'),paddle.to_tensor(1.0,dtype='float32')) # 横行
        similarity_matrix = similarity_matrix * mask
        sf = paddle.nn.Softmax()
        similarity_matrix = sf(similarity_matrix)
        # loss
        label = (ad_idxs.unsqueeze(0) == ad_idxs.unsqueeze(-1))
        label = paddle.where(label,paddle.to_tensor(1.0,dtype='float32'),paddle.to_tensor(0.0,dtype='float32'))
        label = paddle.where(mask == 0,paddle.to_tensor(0.0,dtype='float32'),label)
        loss = paddle.where(label==paddle.to_tensor(1.0,dtype='float32'),-paddle.log2(similarity_matrix),paddle.to_tensor(0.0,dtype='float32'))
        loss_sum = paddle.sum(loss,axis=-1)
        # 返回平均损失
        return loss_sum.mean()
    

class MeanSquareError(nn.Layer):
    
    #计算输出和标签的平均平方误差

    def __init__(self):
        super(MeanSquareError, self).__init__()

    def forward(self, logits, labels, pad_mask):
        # logits: [batch_size, seq_len, dim]
        # labels: [batch_size, seq_len, dim]
        # pad_mask: [batch_size, seq_len]
        batch_size, seq_len, dim = logits.shape
        logits_flatten = paddle.reshape(logits,[batch_size * seq_len,dim])
        labels_flatten = paddle.reshape(labels,[batch_size * seq_len,dim])
        pad_mask = paddle.reshape(pad_mask,[batch_size * seq_len])
        n = paddle.sum(pad_mask) + 1e-8  # 添加eps避免除零

        loss = (logits_flatten - labels_flatten) ** 2
        loss *= pad_mask.unsqueeze(-1)
        loss_sum = paddle.sum(loss)
        return loss_sum / n
    

class CosSimilarity(nn.Layer):

    #求输出和标签之间的余弦相似度
    #去掉paddle.sqrt(dim)可改为点乘相似度

    def __init__(self):
        super(CosSimilarity, self).__init__()

    def forward(self, logits, labels, pad_mask):
        batch_size, seq_len, dim = logits.shape
        logits_flatten = paddle.reshape(logits,[batch_size * seq_len,dim])
        labels_flatten = paddle.reshape(labels,[batch_size * seq_len,dim])
        pad_mask = paddle.reshape(pad_mask,[batch_size * seq_len])
        n = paddle.sum(pad_mask)

        loss = paddle.sum(logits_flatten * labels_flatten, axis=-1) / (paddle.sqrt(dim) + 1e-8)
        loss = -paddle.log2(paddle.clip(loss, min=1e-8))  # 添加clip避免log(0)
        loss *= pad_mask
        return paddle.sum(loss) / (paddle.sum(pad_mask) + 1e-8)
    

class LossWithNegativeSample(nn.Layer):
    """
    支持多个负样本的损失函数
    使用InfoNCE损失：-log(exp(sim(x, x+)) / (exp(sim(x, x+)) + sum(exp(sim(x, x-)))))
    """
    def __init__(self, num_neg_samples=5, temperature=1.0):
        super(LossWithNegativeSample, self).__init__()
        self.num_neg_samples = num_neg_samples
        self.temperature = temperature

    def forward(self, logits, pos_labels, neg_labels, pad_mask):
        """
        Args:
            logits: [batch_size, seq_len, dim] 模型输出
            pos_labels: [batch_size, seq_len, dim] 正样本
            neg_labels: [batch_size, seq_len, num_neg_samples, dim] 负样本
            pad_mask: [batch_size, seq_len] padding mask
        """
        batch_size, seq_len, dim = logits.shape
        num_neg_samples = neg_labels.shape[2]
        
        # Flatten处理
        logits_flatten = paddle.reshape(logits, [batch_size * seq_len, dim])
        pos_labels_flatten = paddle.reshape(pos_labels, [batch_size * seq_len, dim])
        neg_labels_flatten = paddle.reshape(neg_labels, [batch_size * seq_len, num_neg_samples, dim])
        pad_mask_flatten = paddle.reshape(pad_mask, [batch_size * seq_len])

        # 只处理非padding的位置
        valid_mask = pad_mask_flatten.astype('bool')
        valid_logits = logits_flatten[valid_mask]  # [valid_num, dim]
        valid_pos = pos_labels_flatten[valid_mask]  # [valid_num, dim]
        valid_neg = neg_labels_flatten[valid_mask]  # [valid_num, num_neg_samples, dim]

        if valid_logits.shape[0] == 0:
            return paddle.to_tensor(0.0, dtype='float32')

        # 计算正样本相似度
        pos_similarity = paddle.sum(valid_logits * valid_pos, axis=-1) / self.temperature  # [valid_num]
        
        # 计算负样本相似度
        # valid_logits: [valid_num, dim] -> [valid_num, 1, dim]
        # valid_neg: [valid_num, num_neg_samples, dim]
        valid_logits_expanded = valid_logits.unsqueeze(1)  # [valid_num, 1, dim]
        neg_similarity = paddle.sum(valid_logits_expanded * valid_neg, axis=-1) / self.temperature  # [valid_num, num_neg_samples]
        
        # 计算InfoNCE损失
        # 正样本得分: [valid_num, 1]
        pos_scores = pos_similarity.unsqueeze(-1)
        # 所有得分: [valid_num, 1 + num_neg_samples]
        all_scores = paddle.concat([pos_scores, neg_similarity], axis=-1)
        
        # 计算softmax和负对数似然
        log_softmax_scores = F.log_softmax(all_scores, axis=-1)
        # 正样本的标签是0（第一个位置）
        loss = -log_softmax_scores[:, 0]  # [valid_num]
        
        return paddle.mean(loss)


class BCELossWithNegativeSample(nn.Layer):
    def __init__(self, l2_emb=0.00001):  # 添加L2正则化参数
        super(BCELossWithNegativeSample, self).__init__()
        self.bce_loss = paddle.nn.BCEWithLogitsLoss(reduction='none')
        self.l2_emb = l2_emb
        
    def forward(self, logits, pos_items, pad_mask, neg_items=None):
        """
        Args:
            logits: [batch_size, seq_len, dim] 模型输出的logits
            pos_items: [batch_size, seq_len, dim] 正样本
            pad_mask: [batch_size, seq_len] padding mask
            neg_items: [batch_size, seq_len, dim] 负样本
        """
        pos_logits = paddle.sum(logits * pos_items, axis=-1)  # [B, S]
        neg_logits = paddle.sum(logits * neg_items, axis=-1)  # [B, S]

        pos_labels = paddle.ones_like(pos_logits)
        neg_labels = paddle.zeros_like(neg_logits)

        mask = pad_mask.astype('bool')

        pos_loss = self.bce_loss(pos_logits, pos_labels)
        neg_loss = self.bce_loss(neg_logits, neg_labels)

        # 仅取非padding部分计算平均
        pos_loss = paddle.masked_select(pos_loss, mask)
        neg_loss = paddle.masked_select(neg_loss, mask)
        
                # 添加L2正则化
        l2_loss = 0.0
        if self.l2_emb > 0:
            l2_loss = self.l2_emb * (
                paddle.norm(logits) + 
                paddle.norm(pos_items) + 
                paddle.norm(neg_items)
            )

        return paddle.mean(pos_loss) + paddle.mean(neg_loss) + l2_loss



class PointWiseFeedForward(paddle.nn.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = paddle.nn.Conv1D(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout1 = paddle.nn.Dropout(dropout_rate)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout2 = paddle.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose([0, 2, 1]))))))
        outputs = outputs.transpose([0, 2, 1])  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(paddle.nn.Layer):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.dev = args.device
        self.norm_first = args.norm_first
        self.pos_emb = paddle.nn.Embedding(num_embeddings=args.maxlen+1, embedding_dim=args.hidden_units, padding_idx=0)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = paddle.nn.LayerList()  # to be Q for self-attention
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()

        self.last_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = paddle.nn.MultiHeadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = paddle.nn.LayerNorm(normalized_shape=args.hidden_units, epsilon=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units=args.hidden_units, dropout_rate=args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, seqs , mask): 
        pos = paddle.to_tensor(np.tile(np.arange(1, seqs.shape[1] + 1), [seqs.shape[0], 1]),dtype='float32').cuda()
        pos *= mask
        seqs *= seqs.shape[-1] ** 0.5 ## https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py的实现
        # seqs *= mask.unsqueeze(-1) # 使用全零填充向量，与emb具有相同的维度
        seqs += self.pos_emb(paddle.to_tensor(pos, dtype='int64').cuda())
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~paddle.tril(paddle.ones((tl, tl), dtype='bool'), diagonal=0)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs+mha_outputs)

                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self,seqs, mask, pos_seqs, neg_seqs):  # for training        
        logits = self.log2feats(seqs,mask)  
        return logits  # B * S * D

    def predict(self, seqs, mask, item_embs):  # for inference
        log_feats = self.log2feats(seqs,mask)  

        final_feat = log_feats[:, -1, :]  

        logits = paddle.matmul(final_feat,item_embs,transpose_y=True)
        return logits  # preds  # (U, I)