import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    text_max_len: int = 5000
    encoder_layers: int = 4
    encoder_d_model: int = 512
    encoder_d_ff: int = 1536
    encoder_attention_heads: int = 8

    decoder_layers: int = 4
    decoder_d_model: int = 512
    decoder_d_ff: int = 1536
    decoder_attention_heads: int = 8
    dropout:float = 0.1
    def __post_init__(self):
        assert (self.encoder_d_model % self.encoder_attention_heads == 0
                and self.decoder_d_model % self.decoder_attention_heads == 0
                )

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，增加 batch 维度
        pe.requires_grad = False
        self.register_buffer('pe', pe, persistent=True)
    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # 取前 seq_len 个位置编码，广播到 batch_size
        return x

class AttentionProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
    def forward(self, x: torch.Tensor):
        return F.silu(self.w1(x)) * self.w2(x)

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.proj_q = AttentionProjection(d_model)
        self.proj_k = AttentionProjection(d_model)
        self.proj_v = AttentionProjection(d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def linear_attention(self, query, key, value, eps=1e-6, mask=None):
        # query shape: (batch_size, num_heads, seq_len_q, d_k)
        # key shape:   (batch_size, num_heads, seq_len_kv, d_k)
        # value shape: (batch_size, num_heads, seq_len_kv, d_k)

        phi_query = F.elu(query) + 1  # Shape: (batch_size, num_heads, seq_len_q, d_k)
        phi_key = F.elu(key) + 1  # Shape: (batch_size, num_heads, seq_len_kv, d_k)

        if mask is not None:
            phi_key = phi_key * mask[:, None, :, None]

        key_value = torch.matmul(phi_key.transpose(-2, -1), value)  # (batch_size, num_heads, d_k, d_k)
        key_sum_vector = phi_key.sum(dim=-2)  # (batch_size, num_heads, d_k)

        q_k_sum = torch.matmul(phi_query, key_sum_vector.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_heads, seq_len_q)
        z_inv = 1.0 / (q_k_sum.unsqueeze(-1) + eps)  # (batch_size, num_heads, seq_len_q, 1)

        numerator = torch.matmul(phi_query, key_value)  # (batch_size, num_heads, seq_len_q, d_k)

        attn_output = numerator * z_inv  # (batch_size, num_heads, seq_len_q, d_k)
        return attn_output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.proj_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        key = self.proj_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        value = self.proj_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()

        x = self.linear_attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_feedforward)
        self.w2 = nn.Linear(d_feedforward, d_model)
        self.w3 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class EncoderLayer(nn.Module):
    def __init__(self, args:TransformerConfig):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadLinearAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = FeedForward(args.encoder_d_model, args.encoder_d_ff, args.dropout)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, args:TransformerConfig):
        super(Encoder, self).__init__()
        self.d_model = args.encoder_d_model
        self.embedding = nn.Embedding(args.vocab_size, args.encoder_d_model)
        self.pos_encoder = PositionalEncoding(args.text_max_len, args.encoder_d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.encoder_layers)])
        self.norm = nn.LayerNorm(args.encoder_d_model)

    def forward(self, src, mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)

class DecoderLayer(nn.Module):
    def __init__(self, args:TransformerConfig):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadLinearAttention(args.decoder_d_model, args.decoder_attention_heads)
        self.cross_attn = MultiHeadLinearAttention(args.decoder_d_model, args.decoder_attention_heads)
        self.feed_forward = FeedForward(args.encoder_d_model, args.encoder_d_ff, args.dropout)
        self.norm1 = nn.LayerNorm(args.decoder_d_model)
        self.norm2 = nn.LayerNorm(args.decoder_d_model)
        self.norm3 = nn.LayerNorm(args.decoder_d_model)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(tgt2, tgt2, tgt2, tgt_mask))
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(tgt2, memory, memory, memory_mask))
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt2))
        return tgt

class Decoder(nn.Module):
    def __init__(self, args:TransformerConfig):
        super(Decoder, self).__init__()
        self.d_model = args.decoder_d_model
        self.embedding = nn.Linear(args.vocab_size, args.decoder_d_model)
        self.pos_encoder = PositionalEncoding(args.text_max_len, args.decoder_d_model)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.decoder_layers)])
        self.norm = nn.LayerNorm(args.decoder_d_model)
        self.tgt_vocab_map = nn.Linear(args.decoder_d_model, 1, bias=False)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        for layer in self.layers:
            tgt_emb = layer(tgt_emb, memory, tgt_mask, memory_mask)
        tgt_emb = self.norm(tgt_emb)
        output_logits = self.tgt_vocab_map(tgt_emb)  # [batch_size, tgt_len, tgt_vocab_size]
        return output_logits

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.args = TransformerConfig(vocab_size=vocab_size)
        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.001)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output= self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.sigmoid(output)