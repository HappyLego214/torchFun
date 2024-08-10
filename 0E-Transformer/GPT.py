import torch
import torch.nn as nn
from torch.nn import functional as F

import time

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

global_time = time.time()

batch_size = 4
block_size = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
epochs = 5000
eval_iteration = 200
eval_interval = 500
n_embedding = 384
n_layer = 6
n_heads = 6
dropout = 0

torch.manual_seed(1337)

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creates dict with char as keys and int as value
stoi = {ch:i for i,ch in enumerate(chars)}
# Creates dict with int as keys and char as value
itos = dict(enumerate(chars))

# Iterates through stoi to select c's value (taking the integer by selecting char)
# converts string to a list of integers
encoder = lambda s: [stoi[c] for c in s]
# Iterates through itos to select i's value (taking the char by selecting integer)
# converts integers to list of characters and joins to create a string
decoder = lambda l: ''.join([itos[i] for i in l])

# Split data to train/eval sets
data = torch.tensor(encoder(text), dtype=torch.long)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    # We take a random selection from data and create a batch of batch_size
    # We minus block_size to ensure we don't go out of bounds
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # ix outputs a tensor of integer positions in the data
    # We start from these positions and grab the block_size characters thus creating a batch of block_size
    # We do i+block_size is the one that works on taking the characters.
    x = torch.stack([data[i:i+block_size] for i in ix])
    # We do the same here but we start from the next character as these are the targets. 
    # We do +1 to push the target ahead and i+block_size+1 to grab the last target.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads:int, head_size:int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # This last linear layer (self.proj) ensures that the output of the multi-head attention
        # has the same dimensions as the input, which is necessary for the residual connection.
        self.proj = nn.Linear(n_embedding, n_embedding)
        # Dropout can also be applied in the MHA - positioned at the very last after self.proj
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb:int):
        super().__init__()
        self.net = nn.Sequential(
            # In the original Transformer paper the n_emb of the feed forward layer is multiplied by the factor of 4
            # The feed forward multiplies it by 4 then returns it back to the original number at the end.
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            # This last linear layer ensures that the output of the feed-forward network has the same dimensions
            # as the input, which is necessary for the residual connection. We don't name it as self.proj because
            # it is inside a Sequential block.
            nn.Linear(4 * n_emb, n_emb),
            # Dropout is a regularization technique to reduce overfitting
            nn.Dropout(dropout)
        )
     
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emd:int, num_head:int):
        super().__init__()
        head_size = n_emd // num_head
        # head_size must split the n_embeddings equally per head. So if we have 4 heads then 
        # and assuming we have 32 for n_embeddings. We must share the 32 equally to the 4 heads. 
        # That is why head_size is n_embeddings // num_heads.
        self.sa = MultiheadAttention(num_heads=num_head, head_size=head_size) # n Heads of n-dimensional self-attention
        self.ffwd = FeedForward(n_emb=n_emd)
        # We create two normalization layers. In the original paper these are positioned after the MHA and FeedForward
        # But recent innovations have put these layers BEFORE the MHA and FeedForward layers. These are called pre-norm.
        self.ln1 = nn.LayerNorm(normalized_shape=n_emd)
        self.ln2 = nn.LayerNorm(normalized_shape=n_emd)

    def forward(self, x):
        # GPT model has a residual connection after a MHA and a FeedForward. So we can see here that
        # after x goes through MHA - x is added. The same is for the FeedForward.
        # Notice how the layer norms go first with operating on x before the MHA and FeedForward.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    def __init__(self, head_size:int):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size)
        self.query = nn.Linear(n_embedding, head_size)
        self.value = nn.Linear(n_embedding, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # This creates a tril of one's with matrix size of (block_size, block_size)

        # Dropout can also be applied in an individual head - positioned just after softmaxing and application
        # of weight @ value.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        key = self.key(x) # [B,T,C]
        query = self.query(x) # [B,T,C]

        weight = query @ key.transpose(-2,-1) / C**0.5  # [B,T,C] @ [B,C,T] = [B,T,T]
        # The above code scales the dot product of q and k by the inverse of the square root of the head size.
        # This scaling prevents the softmax function from producing extremely large values,
        # which would cause it to assign very high attention to a few tokens and very low attention to others.
        # By scaling, we ensure a more balanced distribution of attention across tokens.

        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight  = self.dropout(weight)
        value = self.value(x)
        output = weight @ value
        return output


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embedding)
        # nn.Embedding requires the actual number of vocab_size (characters being used) for the first argument
        # second argument is a hyperparameter that dictates the length of the embedding vector. We set that to vocab_size
        # just for simplicity. So the word embeddings would be length of vocab_size.
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embedding)
        # The same occurs for the positional embedding. But this time, it's block_size since we want
        # to get the positions of the tokens.
        self.blocks = nn.Sequential(*[Block(n_emd=n_embedding, num_head=n_heads) for _ in range(n_layer)])
        # Layer norm is also added just before the final linear layer - classifier. 
        self.l3_norm = nn.LayerNorm(normalized_shape=n_embedding)
        self.classifier = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx):
        B,T = idx.shape 
        # We pipeline is token_embedding + positional_embedding -> Linear Layer -> logits
        tok_emb = self.token_embedding_table(idx)  # [B,T,C] - C is num_emd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        
        inp_emb = tok_emb + pos_emb # (B,T,C) + (T,C) = (B,T,C) broadcasting occurs
        inp_emb = self.blocks(inp_emb) # Applying multiple heads of self-attention (B,T,C)
        inp_emd = self.l3_norm(inp_emb) # [B,T,C]
        logits = self.classifier(inp_emb) # [B,T,C] - C is vocab_size
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Ensures that idx selects the last block_size elements. Without the - , it would instead
            # select elements starting from block_size. -block_size ensures that the number of 
            # elements stays at block_size. 
            idx_cond = idx[:, -block_size:]
            # idx when inputted is in the shape of [Batch, Time, Channel] 
            logits = self(idx_cond)
            # Select last token in sequence [Batch, Time] to predict next token - transforms to [Batch, Channel]. We are currently
            # working with the batch of the last token in the sequence's logits.
            logits = logits[:,-1]
            # We get the probabilities with softmax on the logits
            probs = F.softmax(logits, dim=1)
            # We're not really 'predicting' since we have to use argmax for that.
            # What is happening is that we're just sampling from the distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # We concatenate the new token to idx - adding to the sequence [Batch, Time+1]
            # Don't get confused idx always stays as [Batch, Time, Channel] - Logit is the one that changes
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def estimate_loss():
    batch_loss = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iteration)
        for i in range(eval_iteration):
            X_eval, y_eval = get_batch(split)
            eval_logits = model(X_eval)
            B,T,C = eval_logits.shape
            eval_loss = loss_fn(eval_logits.view(B*T,C), y_eval.view(B*T))
            losses[i] = eval_loss.item()
        batch_loss[split] = losses.mean()
    return batch_loss

# Train/Test loop
def train(model, epochs, loss_fn, optim):

    for epoch in range(epochs):
        train_time = time.time()
        X_train, y_train = get_batch('train')

        model.train()
        train_logits = model(X_train)
        B,T,C = train_logits.shape
        train_loss = loss_fn(train_logits.view(B*T,C), y_train.view(B*T))

        optim.zero_grad()
        train_loss.backward()
        optim.step()

        model.eval()
        if epoch % eval_interval == 0:
            batch_loss = estimate_loss()
            print(f" Step: {epoch} | Train Loss: {batch_loss['train']:.4f} | Test Loss: {batch_loss['test']:.4f}")
            print("--- %s seconds ---" % (time.time() - train_time))

      
train(model, epochs, loss_fn, optim)
# Generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
# Transform data from tensor to list
print(decoder(model.generate(context, max_new_tokens=1000)[0].tolist()))
print("--- %s seconds ---" % (time.time() - global_time))
