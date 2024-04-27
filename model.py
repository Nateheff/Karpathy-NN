import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass




max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

@dataclass
class Config:
    block_size: int = 32
    batch_size=32
    vocab_size: int = 300
    n_heads: int = 6
    n_embd: int = 192
    n_layer: int = 6
    head_size: int = n_embd // n_heads

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.head_size #The dimension of the key, query, and value. The dimensions of these variables are (batch_size, block_size, head_size). Impacts how much data each holds
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        affinities = q @ k.transpose(-2,-1)
        affinities = affinities * (self.head_size**-0.5)
        tril = torch.tril(torch.ones(T, T, device=device))

        affinities = affinities.masked_fill(tril == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.dropout(affinities)

        out = affinities @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.heads = nn.ModuleList([Head(config) for _ in range(self.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # head_size = config.n_embd // config.n_heads
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection
        x = x + self.ffwd(self.ln2(x)) # Residual Connection
        return x
    
class LayerNorm: #The EXACT same as BatchNorm1d implementation, but we change the dimension to calculate mean and var on x in _call_ from 0 -> 1
# and no need for any of the buffers and test vs training stuff
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps

        #trainable parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)


    def __call__(self, x):

        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
    
class BigramLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.config.block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    