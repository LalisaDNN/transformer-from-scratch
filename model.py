import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [sed_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model)) # [, d_model/2]

        self.pe[:, 0::2] = torch.sin(pos / div_term)
        self.pe[:, 1::2] = torch.cos(pos / div_term)

        self.pe = self.pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        return x + self.dropout((self.pe[:, x.shape[1], :]).requires_grad_(False))
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * ((x - mean) / torch.sqrt(var + self.eps)) + self.bias
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, activation: str = 'relu', dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation_fn(self.linear1(x))))

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V,  attention_mask=None):
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.bmm(attn_weights, V)
        return output, attn_weights
    
def transpose_qkv(x: torch.Tensor, n_heads: int):
    x = x.view(x.size(0), x.size(1), n_heads, -1)  # [batch_size, max_len, n_heads, head_dim]
    return x.permute(0, 2, 1, 3)  # [batch_size, n_heads, max_len, head_dim]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_bias: bool = False, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.attention = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, attention_mask=None):
        Q = self.w_q(queries), K = self.w_k(keys), V = self.w_v(values)
        Q, K, V = transpose_qkv(Q, self.n_heads), transpose_qkv(K, self.n_heads), transpose_qkv(V, self.n_heads)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        x, attn_weights = self.attention(Q, K, V, attention_mask)
        # [B, H, L, d_k] -> [B, L, H, d_k] -> [B, L, d_model]
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model) 
        x = self.out_proj(x)

        return self.dropout(x), attn_weights

class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_bias: bool = False, 
                 activation: str = 'relu', dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, use_bias, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, activation, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, attention_mask=None):
        y, attn_weights = self.mhsa(x, x, x, attention_mask)
        x = self.addnorm1(x, y)
        x = self.addnorm2(x, self.ffn(x))
        return self.dropout(x), attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_block: TransformerEncoderBlock, vocab_size:int, 
                 num_blocks:int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_embedding = InputEmbeddings(encoder_block.d_model, vocab_size)
        self.pe = PositionalEncoding(encoder_block.d_model, max_len, dropout)
        self.encoder_blocks = nn.ModuleList()
        self.attn_weights = []
        for _ in range(num_blocks):
            self.encoder_blocks.append(encoder_block)

    def forward(self, x, attention_mask=None):
        x = self.input_embedding(x)
        x = self.pe(x)

        for i, block in enumerate(self.encoder_blocks):
            x, attn_weights = block(x, attention_mask)
            self.attn_weights[i] = attn_weights

        return x
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_bias: bool = False, 
                 activation: str = 'relu', dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, use_bias, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, use_bias, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, activation, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_output, target_attention_mask=None, src_attention_mask=None):
        y, target_attn_weights = self.self_attn(x, x, x, target_attention_mask)
        x = self.addnorm1(x, y)

        y, src_attn_weights = self.cross_attn(x, enc_output, enc_output, src_attention_mask)
        x = self.addnorm2(x, y)

        x = self.addnorm3(x, self.ffn(x))
        return x, target_attn_weights, src_attn_weights
    
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_block: TransformerDecoderBlock, vocab_size: int, 
                 num_blocks: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_embedding = InputEmbeddings(decoder_block.d_model, vocab_size)
        self.pe = PositionalEncoding(decoder_block.d_model, max_len, dropout)
        self.decoder_blocks = nn.ModuleList()
        self.target_attn_weights = []
        self.src_attn_weights = []
        for _ in range(num_blocks):
            self.decoder_blocks.append(decoder_block)

    def forward(self, x, enc_output, target_attention_mask=None, src_attention_mask=None):
        x = self.input_embedding(x)
        x = self.pe(x)

        for i, block in enumerate(self.decoder_blocks):
            x, target_attn_weights, src_attn_weights = block(x, enc_output, target_attention_mask, src_attention_mask)
            self.target_attn_weights[i] = target_attn_weights
            self.src_attn_weights[i] = src_attn_weights

        return x
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)  # [batch_size, seq_len, vocab_size]
    
class Transformer(nn.Module):
    def __init__(self, conf, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderBlock(
                d_model=conf['d_model'],
                num_heads=conf['num_heads'],
                use_bias=conf['use_bias'],
                activation=conf['activation'],
                dropout=conf['dropout']
            ),
            vocab_size=src_vocab_size,
            num_blocks=conf['num_encoder_blocks'],
            max_len=conf['max_len']
        )
        self.decoder = TransformerDecoder(
            TransformerDecoderBlock(
                d_model=conf['d_model'],
                num_heads=conf['num_heads'],
                use_bias=conf['use_bias'],
                activation=conf['activation'],
                dropout=conf['dropout']
            ),
            vocab_size=tgt_vocab_size,
            num_blocks=conf['num_decoder_blocks'],
            max_len=conf['max_len']
        )
        self.projection_layer = ProjectionLayer(
            d_model=conf['d_model'],
            vocab_size=tgt_vocab_size
        )

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, src, tgt, src_attention_mask=None, tgt_attention_mask=None):
        enc_output = self.encoder(src, src_attention_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_attention_mask, src_attention_mask)
        output = self.projection_layer(dec_output)
        return output

def get_model(conf, src_vocab_size, tgt_vocab_size):
    model = Transformer(conf, src_vocab_size, tgt_vocab_size)
    model.init_parameters()
    return model