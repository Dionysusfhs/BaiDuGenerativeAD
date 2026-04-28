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

    # 将负样本用于计算损失 e ^ sim(x, x+) / (e ^ sim(x, x+) + e ^ sim(x, x-))
    # 假设每个物品只采样一个负例

    def __init__(self):
        super(LossWithNegativeSample, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, logits, pos_labels, neg_labels, pad_mask):
        batch_size, seq_len, dim = logits.shape
        logits_flatten = paddle.reshape(logits, [batch_size * seq_len,dim])
        pos_labels_flatten = paddle.reshape(pos_labels, [batch_size * seq_len,dim])
        neg_labels_flatten = paddle.reshape(neg_labels, [batch_size * seq_len,dim])
        pad_mask = paddle.reshape(pad_mask, [batch_size * seq_len])

        #去掉所有填充的位置
        pos_labels_flatten = pos_labels_flatten[paddle.where(pad_mask != 0)]
        neg_labels_flatten = neg_labels_flatten[paddle.where(pad_mask != 0)]

        pos_similarity = paddle.sum(logits_flatten * pos_labels_flatten, axis=-1).unsqueeze(-1)
        neg_similarity = paddle.sum(logits_flatten * neg_labels_flatten, axis=-1).unsqueeze(-1)
        similarity = paddle.concat([pos_similarity, neg_similarity], axis=-1) # [batch_size * seq_len, 2]
        labels = paddle.zeros(batch_size * seq_len)
        loss = self.CE_loss(similarity, labels)
        return loss
    

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

class SASRecF(paddle.nn.Layer):
    def __init__(self, args):
        super(SASRecF, self).__init__()
        self.dev = args.device
        self.norm_first = args.norm_first
        self.hidden_units = args.hidden_units
        self.sid_emb_dim = args.sid_emb_dim
        self.pos_emb = paddle.nn.Embedding(num_embeddings=args.maxlen+1, embedding_dim=self.hidden_units, padding_idx=0)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)

        # sementic_id embedding
        self.sid_embedding = paddle.nn.Embedding(args.idx_len, self.sid_emb_dim)
        self.sid_pos_embedding = paddle.nn.Embedding(args.tokenseq_maxlen, self.sid_emb_dim)

        # 将token embedding从 [B, T, m, H] 投影到 [B, T, H]
        self.query_vector = paddle.nn.Linear(self.sid_emb_dim, self.sid_emb_dim)
        # 将较小的embedding维度投影到hidden_units
        self.sid_proj = paddle.nn.Linear(self.sid_emb_dim, self.hidden_units)
        # 将token embedding和sementic_id embedding融合
        self.fusion_layer = paddle.nn.Linear(self.hidden_units * 2, self.hidden_units)
        
        self.id_embedding_proj = paddle.nn.Linear(args.emb_dim, self.hidden_units)

        self.attention_layernorms = paddle.nn.LayerList()  # to be Q for self-attention
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()

        self.last_layernorm = paddle.nn.LayerNorm(normalized_shape=self.hidden_units, epsilon=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(normalized_shape=self.hidden_units, epsilon=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = paddle.nn.MultiHeadAttention(embed_dim=self.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = paddle.nn.LayerNorm(normalized_shape=self.hidden_units, epsilon=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units=self.hidden_units, dropout_rate=args.dropout_rate)
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

    def forward(self, seqs, mask, ad_tokenidx_seqs, pos_seqs=None, neg_seqs=None):
        B, T, m = ad_tokenidx_seqs.shape
        
        # 使用较小的embedding维度
        sid_embed = self.sid_embedding(ad_tokenidx_seqs)  # [B, T, m, sid_emb_dim]
        sid_pos_ids = paddle.arange(m).reshape([1, 1, m])
        sid_embed += self.sid_pos_embedding(sid_pos_ids)
        
        # 降维方式

        # 1、使用注意力降维token序列
        sid_embed_reshape = sid_embed.reshape([-1, m, self.sid_emb_dim])
        query = self.query_vector(paddle.ones([B*T, 1, self.sid_emb_dim]))
        attn_weights = paddle.matmul(query, sid_embed_reshape.transpose([0, 2, 1]))
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        sid_final = paddle.matmul(attn_weights, sid_embed_reshape)
        sid_final = sid_final.reshape([B, T, self.sid_emb_dim])
        
        # 2、求和平均
        # sid_final = paddle.mean(sid_embed, axis=2)  # [B, T, sid_emb_dim]
        
        # 投影到hidden_units维度
        sid_final = self.sid_proj(sid_final)  # [B, T, H]
        
        # text表示
        text_proj = self.id_embedding_proj(seqs)  # [B, T, H]
        
        # 融合得到最终序列表示
        seqs = paddle.tanh(self.fusion_layer(
            paddle.concat([sid_final, text_proj], axis=-1)
        ))  # [B, T, H]
        
        logits = self.log2feats(seqs, mask)  
        return logits

    def predict(self, seqs, mask, item_embs, ad_tokenidx_seqs):  # for inference
        log_feats = self.forward(seqs, mask, ad_tokenidx_seqs)  

        final_feat = log_feats[:, -1, :]  

        logits = paddle.matmul(final_feat,item_embs,transpose_y=True)
        return logits  # preds  # (U, I)
