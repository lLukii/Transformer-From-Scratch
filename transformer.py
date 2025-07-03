import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
  def __init__(self):
    super(PositionalEncoding, self).__init__()

  def forward(self, x: torch.tensor):
    r"""
    Computes positional encoding for transformer
    Parameters:
      X: [batch_size, seq_len, d_model]
    """
    batch_size, seq_len, d_model = x.size()
    PE = torch.zeros(batch_size, seq_len, d_model)
    for pos in range(seq_len):
      for i in range(d_model // 2):
        PE[:, pos, 2 * i] = np.sin(pos / (1e4 ** ((2 * i) / d_model)))
        PE[:, pos, 2 * i + 1] = np.cos(pos / (1e4 ** ((2 * i) / d_model)))

    return x + PE

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, d_model: int, attention_size: int):
    # man how tf do u vectorize ts 😭
    """
    Computes multi-head self-attention
    Parameters:
      num_heads: number of attention heads
      mask: whether to use mask
      d_model: dimensionality of input
      attention_size: dimensionality of Q, K, and V matricies
    """
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.attention_size = attention_size
    self.linear_K = [nn.Linear(d_model, attention_size)] * num_heads
    self.linear_V = [nn.Linear(d_model, attention_size)] * num_heads
    self.linear_Q = [nn.Linear(d_model, attention_size)] * num_heads

    self.linear_O = nn.Linear(num_heads * attention_size, d_model)

  def forward(self, x: torch.tensor, mask : bool = False):
    '''
    Parameters:
      X: [batch_size, seq_len, d_model]
      mask: bool whether or not to use mask
    '''
    batch_size, seq_len, d_model = x.size()
    attention = torch.zeros(batch_size, self.num_heads, seq_len, self.attention_size)
    mask  = torch.tril(torch.ones(seq_len, seq_len)) if mask else None
    for head in range(self.num_heads):
      K = self.linear_K[head](x)
      V = self.linear_V[head](x)
      Q = self.linear_Q[head](x)
      attention_head = F.scaled_dot_product_attention(Q, K, V,
                                                      attn_mask=mask) # ok lil bro
      attention[:, head, :, :] = attention_head

    attention = torch.reshape(attention, (batch_size, seq_len, -1)) # num_heads * attention_size
    attention = self.linear_O(attention)
    return x + attention


class MHCrossAttention(nn.Module):
  def __init__(self, num_heads: int, d_model: int, attention_size: int):
    super(MHCrossAttention, self).__init__()
    self.num_heads = num_heads
    self.attention_size = attention_size
    self.linear_K = [nn.Linear(d_model, attention_size)] * num_heads
    self.linear_V = [nn.Linear(d_model, attention_size)] * num_heads
    self.linear_Q = [nn.Linear(d_model, attention_size)] * num_heads

    self.linear_O = nn.Linear(num_heads * attention_size, d_model)
  
  def forward(self, encoder: torch.tensor, decoder: torch.tensor):
    batch_size, _, _ = encoder.size()
    _, seq_len, _ = decoder.size()
    attention = torch.zeros(batch_size, self.num_heads, seq_len, self.attention_size)
    for head in range(self.num_heads):
      Q = self.linear_Q[head](decoder)
      K = self.linear_K[head](encoder)
      V = self.linear_V[head](encoder)
      attention_head = F.scaled_dot_product_attention(Q, K, V) # softmax(QK^T / sqrt(d_k))V
      attention[:, head, :, :] = attention_head 
    
    attention = torch.reshape(attention, (batch_size, seq_len, -1)) # num_heads * attention_size
    attention = self.linear_O(attention)
    return decoder + attention
  
class FFNLayer(nn.Module):
  def __init__(self, input_size, hidden_size, dropout_rate=0.1):
    super(FFNLayer, self).__init__()
    self.ffn = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, input_size),
    )

  def forward(self, x):
    ''' X: [batch_size, seq_len, d_model] '''
    return self.ffn(x)

class Encoder(nn.Module):
  def __init__(self, num_heads=8, d_model=512, attention_size=64, hidden_size=2048):
    super(Encoder, self).__init__()
    self.multi_head_attention = MultiHeadAttention(num_heads, d_model, attention_size)
    self.ffn = FFNLayer(d_model, hidden_size)
    self.norm = nn.LayerNorm(d_model)
    self.pe = PositionalEncoding()

  def forward(self, x: torch.tensor, n_x: int):
    '''
    X: [batch_size, seq_len, d_model]
    '''
    x = self.pe(x)
    for _ in range(n_x):
      tmp = torch.clone(x)
      x = self.multi_head_attention(x)
      x = self.norm(x + tmp)
      tmp = torch.clone(x)
      x = self.ffn(x)
      x = self.norm(x + tmp)
    return x

class Decoder(nn.Module):
  def __init__(self, num_heads=8, d_model=512, attention_size=64, hidden_size=2048):
    super(Decoder, self).__init__()
    self.multi_head_attention = MultiHeadAttention(num_heads, d_model, attention_size)
    self.ffn = FFNLayer(d_model, hidden_size)
    self.norm = nn.LayerNorm(d_model)
    self.pe = PositionalEncoding()

  def forward(self, x: torch.tensor, n_x: int):
    '''
    X: [batch_size, seq_len, d_model]
    '''
    x = self.pe(x)
    for _ in range(n_x):
      tmp = torch.clone(x)
      x = self.multi_head_attention(x) # todo: implement masked mha
      x = self.norm(x + tmp)
      tmp = torch.clone(x)
      x = self.multi_head_attention(x)
      x = self.norm(x + tmp)
      tmp = torch.clone(x)
      x = self.ffn(x)
      x = self.norm(x + tmp)

    return x

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
