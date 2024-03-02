import torch
import os 
import torch.nn as nn
from torch.nn import functional as F
from UIODataGenerator import getUIOTrainingVectors
import random

# hyperparameters
batch_size = 16 # how many independent sequences will we processed in parallel?
block_size = 6 # length of the UIO
vocab_size = 6 # number of possible values for each element of the UIO 
max_iters = 4000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
eval_iters = 100
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)


# Data generation

#def generate_vectors(n_vectors,length_vectors, start, end):
#    vectors = []
#    for _ in range(n_vectors):
#        vector = torch.sort(torch.randint(start, end+1, (length_vectors,)))[0]
#        vectors.append(vector)
#    return torch.stack(vectors)





# Train and test splits: UIO=[unit interval order,Stanley coeff]  
UIOin, UIOout = getUIOTrainingVectors(block_size)
n = int(0.9*len(UIOin)) # first 90% will be train, rest val
#Random shuffle
# Pair the elements of the two lists
paired = list(zip(UIOin, UIOout))
# Shuffle the pairs
random.shuffle(paired)
# Unzip the pairs
UIOin, UIOout = zip(*paired)
UIOintrain_data = UIOin[:n]
UIOouttrain_data = UIOout[:n]
UIOinval_data = UIOin[n:]
UIOoutval_data = UIOout[n:]


# data loading 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        datain = UIOintrain_data 
        dataout = UIOouttrain_data  
    else:
        datain = UIOinval_data 
        dataout = UIOoutval_data
    batch_indexes = torch.randint(len(datain), (batch_size,))
    x = torch.stack([datain[i] for i in batch_indexes])
    y = torch.stack([dataout[i] for i in batch_indexes])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            #print(X.shape,Y.shape)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# First attempt: multihead transformer encoder with embeddings. That is, instead of working with
# the UIO integer vectors directly, we first embed them into a continuous space of dimension n_embd.

# Version 1 (default) MSE loss on (B,1,1) tensor, where the training pair is (UIOVector, Coeff), so the network learns one 
#specific Stanley coefficient of lenth n=length of UIO. UIOVector has shape (B,T), Coeffs has size (B). 

class GPTStanleywithEmbed(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, 1)
        self.lm_head2 = nn.Linear(block_size,1)

    def forward(self, idx, targets=None): 
        B, T = idx.shape
        # idx is (B,T) tensor of integers, targets is (B,1) tensor of integers.
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,1)
        logitsfinal = self.lm_head2(logits.squeeze(-1)) # (B,1)

        if targets is None:
            loss = None
        else:
            mse_loss = nn.MSELoss()
            loss = mse_loss(logitsfinal, targets.float()) #both logitsfinal and targets tensors of size (B,1)
        return logitsfinal, loss

    

# Version 2: MSE loss on (B,T,1) tensor, where the training pair is (UIOVector, CoeffVector)
# Here CoeffVector[i]=StanleyCoeff(length i partition), so the network learns n coefficients simultaneously. 

class GPTStanleywithEmbed2(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, 1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,1)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T)
            targets = targets.view(B*T).float()
            mse_loss = nn.MSELoss()
            loss = mse_loss(logits, targets)
            

        return logits, loss


model = GPTStanleywithEmbed()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Print model parameters

#for name, param in model.named_parameters():
#    print(name, param)

# Print predicted (1,2,3) coefficients

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    #for i in range(len(UIOinval_data)-batch_size):
    #    xb = torch.stack(UIOinval_data[i:i+batch_size])
    #    yb = torch.stack(UIOoutval_data[i:i+batch_size])  
    xb, yb = get_batch('val')       
    logits, loss = model(xb,yb)  # Get the model's predictions
    paired = list(zip(logits.tolist(),yb.tolist()))
    for pair in paired:
        print('Predicted:', pair[0], 'Real value:', pair[1])
