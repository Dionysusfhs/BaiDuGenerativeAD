# model.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LIGERModel(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.sid_len = args.sid_len
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.vocab_size = args.sid_size
        self.text_emb_dim = args.emb_size
        self.dropout_rate = args.dropout_rate
        self.dim_feedforward = args.dim_feedforward
        self.num_layers = args.num_layers
        # Embeddings
        self.sid_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.sid_pos_embedding = nn.Embedding(self.sid_len, self.hidden_size)
        # self.sid_pos_embedding = nn.Embedding(3, 128)
        self.item_pos_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        self.text_proj = nn.Linear(self.text_emb_dim, self.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.num_heads, dim_feedforward=self.dim_feedforward, dropout=self.dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            self.hidden_size, nhead=self.num_heads, dim_feedforward=self.dim_feedforward, dropout=self.dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Output layer for SID generation
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        self.sim_temperature = args.loss_temperature

    def forward(self, sid_seq, text_reps, item_pos_ids, next_text_rep, next_sid):
        """
        sid_seq:       [B, T, m]
        text_reps:     [B, T, 1024]
        item_pos_ids:  [B, T]
        next_text_rep: [B, 1024]
        next_sid:      [B, m+1]
        S = T * m
        """
        B, T, m = sid_seq.shape

        # ===== Encoder embedding =====
        sid_embed = self.sid_embedding(sid_seq)                     # [B, T, m, H]
        sid_pos_ids = paddle.arange(m).reshape([1, 1, m])
        sid_embed += self.sid_pos_embedding(sid_pos_ids)

        text_proj = self.text_proj(text_reps).unsqueeze(2).expand([-1, -1, m, -1])  # [B, T, m, H]
        item_pos_embed = self.item_pos_embedding(item_pos_ids).unsqueeze(2).expand([-1, -1, m, -1])

        embed = sid_embed + text_proj + item_pos_embed             # [B, T, m, H]
        encoder_input = embed.reshape([B, T * m, self.hidden_size]).transpose([1, 0, 2])  # [L, B, H]

        memory = self.encoder(encoder_input)  # [L, B, H]
        context_vec = memory[-1]  # [B, H]

        # ===== Cosine similarity loss =====
        pred_proj = F.normalize(context_vec, axis=-1)  # [B, H]
        true_proj = F.normalize(self.text_proj(next_text_rep), axis=-1)  # [B, H]
        sim_scores = paddle.matmul(pred_proj, true_proj, transpose_y=True) / self.sim_temperature  # [B, B]
        labels = paddle.arange(B, dtype='int64')
        sim_loss = F.cross_entropy(sim_scores, labels)

        # ===== Decoder embedding =====
        tgt_input = next_sid[:, :-1]  # [B, m]
        tgt_target = next_sid[:, 1:]  # [B, m]
        tgt_embed = self.sid_embedding(tgt_input).transpose([1, 0, 2])  # [m, B, H]

        # 构造 causal mask
        # m = tgt_input.shape[1]
        # causal_mask = paddle.triu(paddle.full([m, m], float('-inf')), 1)
        # tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand([B, self.num_heads, m, m])

        # ===== Transformer decoder =====
        try:
            memory_avg = memory.mean(axis=0, keepdim=True)  # [1, B, H]
            memory_expanded = memory_avg.expand([m, -1, -1])  # [m, B, H]
            decoder_out = self.decoder(tgt=tgt_embed, memory=memory_expanded) # [m, B, H]
        except Exception as e:
            print("Decoder error:", e)
            raise e  # Re-raise to stop silently failing

        logits = self.lm_head(decoder_out).transpose([1, 0, 2])  # [B, m, vocab_size]

        # ===== LM loss =====
        lm_loss = F.cross_entropy(
            logits.reshape([-1, logits.shape[-1]]),  # [B*m, vocab_size]
            tgt_target.reshape([-1]),                # [B*m]
            ignore_index=0
        )
        return sim_loss + lm_loss
