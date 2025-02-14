import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self,dim_in,dim_out):
    super().__init__()
    self.w_q = nn.Linear(dim_in,dim_out,bias=False)
    self.w_k = nn.Linear(dim_in,dim_out,bias=False)
    self.w_v = nn.Linear(dim_in,dim_out,bias=False)

  def forward(self,x):
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)
    attention_scores = q @ k.transpose(2,1)
    attention_weights = attention_scores / k.shape[-1] ** 0.5
    attention_weights = torch.softmax(attention_weights,dim=-1)
    output = attention_weights @ v
    return output
  
class CausalAttention(nn.Module):
  def __init__(self,dim_in,dim_out,context_length,dropout_rate):
    super().__init__()
    self.w_q = nn.Linear(dim_in,dim_out)
    self.w_k = nn.Linear(dim_in,dim_out)
    self.w_v = nn.Linear(dim_in,dim_out)
    self.dropout = nn.Dropout(dropout_rate)
    self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1).bool())

  def forward(self,x):
    batch_size,context_length,dim_in = x.shape
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)
    attention_scores = q @ k.transpose(2,1)
    # print(attention_scores.shape)
    attention_scores.masked_fill_(self.mask[:context_length,:context_length],-torch.inf)
    attention_weights = torch.softmax(attention_scores/(k.shape[-1] ** 0.5),dim = -1)
    attention_weights = self.dropout(attention_weights)
    output = attention_weights @ v
    return output

class mh_attention(nn.Module):
  def __init__(self,num_heads,dim_in,dim_out,context_length,dropout_rate):
    super().__init__()
    self.heads  = nn.ModuleList([CausalAttention(dim_in,dim_out,context_length,dropout_rate) for _ in range(num_heads)])

  def forward(self,x):
    return torch.cat([head(x) for head in self.heads],dim=-1)
  

class MultiHeadAttention(nn.Module):
  def __init__(self,dim_in,dim_out,num_heads,context_length,dropout_rate):
    super().__init__()
    self.w_q = nn.Linear(dim_in,dim_out,bias=True)
    self.w_k = nn.Linear(dim_in,dim_out,bias=True)
    self.w_v = nn.Linear(dim_in,dim_out,bias=True)
    self.out_proj = nn.Linear(dim_out,dim_out)
    self.head_dim = dim_out // num_heads
    self.num_heads = num_heads
    self.dim_out = dim_out
    self.dropout = nn.Dropout(dropout_rate)
    self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1).bool())

  def forward(self,x):
    b,context_length,d_in = x.shape
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)
    # print(f'query shape : {q.shape}')
    # print(f'key shape : {k.shape}')
    # print(f'value shape : {v.shape}')
    q = q.view(b,context_length,self.num_heads,self.head_dim).transpose(1,2)
    k = k.view(b,context_length,self.num_heads,self.head_dim).transpose(1,2)
    v = v.view(b,context_length,self.num_heads,self.head_dim).transpose(1,2)
    # print(f'query shape after splitting into {self.num_heads} heads: {q.shape}')
    # print(f'key shape after splitting into {self.num_heads} heads: {k.shape}')
    # print(f'value shape after splitting into {self.num_heads} heads: {v.shape}')
    attention_scores = q @ k.transpose(3,2)
    attention_scores.masked_fill_(self.mask[:context_length,:context_length],-torch.inf)
    attention_weigths = torch.softmax(attention_scores/(k.shape[-1] ** 0.5),dim=-1)
    # print(f'attention weights shape : {attention_weigths.shape}')
    attention_weigths = self.dropout(attention_weigths)
    context_vector = (attention_weigths @ v).transpose(1,2)
    context_vector = context_vector.contiguous().view(b,context_length,self.dim_out)
    # print(f'context vector shape : {context_vector.shape}')
    output = self.out_proj(context_vector)
    return output
  
class LayerNorm(nn.Module):
  def __init__(self,emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self,x):
    mean = x.mean(dim=-1,keepdim=True)
    variance = x.var(dim=-1,keepdim=True,unbiased=False)
    normalized_x = (x-mean)/torch.sqrt(variance+self.eps)
    return self.scale * normalized_x + self.shift
  

class MLP(nn.Module):
  def __init__(self,emb_dim):
    super().__init__()
    self.layers = nn.Sequential(nn.Linear(emb_dim,emb_dim*4),
                                nn.GELU(),
                                nn.Linear(emb_dim*4,emb_dim))
  def forward(self,x):
    return self.layers(x)
  
class Transformer_Block(nn.Module):
  def __init__(self,emb_dim,num_heads,context_length,dropout_rate):
    super().__init__()
    self.ln1 = LayerNorm(emb_dim)
    self.ln2 = LayerNorm(emb_dim)
    self.attn = MultiHeadAttention(dim_in=emb_dim,dim_out=emb_dim,num_heads=num_heads,context_length=context_length,dropout_rate=dropout_rate)
    self.fcn = MLP(emb_dim)
    self.dropout = nn.Dropout(dropout_rate)
  def forward(self,x):
    x = x + self.dropout(self.attn(self.ln1(x)))
    x = x + self.dropout(self.fcn(self.ln2(x)))
    return x

class GPT(nn.Module):
  def __init__(self,vocab_size,emb_dim,num_heads,num_layers,context_length,dropout_rate):
    super().__init__()

    self.token_embedding = nn.Embedding(vocab_size,emb_dim)
    # context length is basically max_length here
    self.pos_embedding = nn.Embedding(context_length,emb_dim)
    self.dropout = nn.Dropout(dropout_rate)
    self.trf_blocks = nn.ModuleList([Transformer_Block(emb_dim,num_heads,context_length,dropout_rate) for _ in range(num_layers)])
    self.final_norm = LayerNorm(emb_dim)
    self.output = nn.Linear(emb_dim,vocab_size)

  def forward(self,x):
    batch_size,context_length = x.shape
    token_embeddings = self.token_embedding(x)
    pos_embeddings = self.pos_embedding(torch.arange(context_length,device=x.device))
    x = token_embeddings + pos_embeddings
    x = self.dropout(x)
    for block in self.trf_blocks:
        x = block(x)
    x = self.final_norm(x)
    logits = self.output(x)
    return logits