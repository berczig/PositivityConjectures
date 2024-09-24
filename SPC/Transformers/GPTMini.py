# An adapted version of the small GPT model of Karpathy, which is a transformer model with a multihead self-attention mechanism.
# A linear layer is added to predict integer scalars.

# The model is trained on pairs (X, Y) where X=(a_1,...,a_n) is a list of integers between 0 and n, and Y is the predicted integer invairant, 
# e.g Stanley coefficient. 

# In this version we do not use tokenization, but we use embeddings. The input is a (a_1,...,a_n) is meant to be an integer vector, so the vocabulary size 
# is the number of unique integers in the input array. 

# The output is an integer y, so the model learns one specific group invariant y from other group invariants a_1,...,a_n.




import torch
import os 
import torch.nn as nn
from torch.nn import functional as F
from DataGenerator import getTrainingDataFromFile
import matplotlib.pyplot as plt
import random

# hyperparameters
filename = "SPC/Transformers/uio_data_n=9.csv" # file with training data
X, Y = getTrainingDataFromFile(filename, partition=(3,6)) # read in training data
batch_size = 16 # how many independent sequences will we processed in parallel?
block_size = X.shape[1] # length of the input array, that is the number of invariants from which we want to predict the # of subgroups
# vocab_size is the number of unique integers in X and Y
vocab_size = len(set(X.flatten().tolist()))
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
eval_iters = 300
n_embd = 84
n_head = 12
n_layer = 6
dropout = 0.0
# ------------

torch.manual_seed(1337)

# Print the parameters 

print('Hyperparameters:')
print('batch_size:', batch_size)
print('block_size:', block_size)
print('vocab_size:', vocab_size)
print('max_iters:', max_iters)
print('eval_interval:', eval_interval)
print('learning_rate:', learning_rate)
print('device:', device)
print('eval_iters:', eval_iters)
print('n_embd:', n_embd)
print('n_head:', n_head)
print('n_layer:', n_layer)
print('dropout:', dropout)


# Split the data into training and test sets
  
n = int(0.8*len(X)) # first 90% will be train, rest val
#Random shuffle
# Pair the elements of the two lists
paired = list(zip(X, Y))
# Shuffle the pairs
random.shuffle(paired)
# Unzip the pairs
X, Y = zip(*paired)
X_train = X[:n]
Y_train = Y[:n]
X_test = X[n:]
Y_test = Y[n:]

print('Training data:', len(X_train), 'Test data:', len(X_test))


# data loading 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        datain = X_train
        dataout = Y_train    
    else:
        datain = X_test
        dataout = Y_test
    batch_indexes = torch.randint(len(datain), (batch_size,))
    x = torch.stack([datain[i] for i in batch_indexes])
    y = torch.stack([dataout[i] for i in batch_indexes])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
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

# First model: multihead transformer encoder with embeddings. Input is a (a_1,...,a_n) output is scalar y.
# So the model learns one specific group invariant y from other group invariants a_1,...,a_n. 
# MSE loss on (B,1) tensor, input is (B,T) tensor of integers, output is (B,1) tensor of floats.


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

    

    # Version 2: The model learns n invariants simultaneously from the group invariants a_1,...,a_n.
    # The input is a (a_1,...,a_n) output is a vector y=(y_1,...,y_n).


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
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Plot all test points in the plane (predicted, real value)

import matplotlib.pyplot as plt
import torch


# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and real values
x_coords = []
y_coords = []

# Disable gradient calculation
with torch.no_grad():
    # Iterate over the entire test dataset
    for i in range(0, len(X_test), batch_size):
        xb = torch.stack(X_test[i:i+batch_size])
        yb = torch.stack(Y_test[i:i+batch_size])
        logits, loss = model(xb, yb)  # Get the model's predictions
        paired = list(zip(logits.tolist(), yb.tolist()))
        for pair in paired:
            x_coords.append(pair[0])
            y_coords.append(pair[1])

# Plot the points with smaller dots
plt.scatter(x_coords, y_coords, s=5)  # 's' parameter controls the size of the dots
plt.xlabel('Predicted')
plt.ylabel('Real value')
plt.title('Predicted vs Real Values')
plt.show()

# Alternative plotting method: histogram
plt.hist(x_coords, bins=30, alpha=0.5, label='Predicted')  # Histogram for predicted values
plt.hist(y_coords, bins=30, alpha=0.5, label='Real value')  # Histogram for real values
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted and Real Values')
plt.legend(loc='upper right')
plt.show()