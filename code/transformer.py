import torch
import torch.nn as nn
import torch.nn.functional as F

# add all  your Encoder and Decoder code here
# some extra hyperparameters for regularization

# Options explored
# ALiBi
use_linear_bias = False
linear_bias_m = 0.4

# DeBERTa
use_deberta = False
positional_encoding = None

dropout = 0.2
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, use_dropout = False, look_forward = False):
        super().__init__()
        self.head_size = head_size
        self.look_forward = look_forward
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        if not look_forward:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def compute_linear_bias(self, T):
        bias = torch.zeros(T, T)
        for i in range(T):
            for j in range(0, i+1):
                bias[i, j] = j - i
        return bias * linear_bias_m

    
    def forward(self, x):
        '''
        x: (Tensor): (B, T, n_embd) for classification T == block_size; 
        Return:
            attention: (B, block_size, block_size)
            results: (B, block_size, head_size)
        '''
        _, T, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x) * self.head_size**-0.5
        wei = query @ key.transpose(-1, -2)
        if use_linear_bias:
            wei = wei + self.compute_linear_bias(T)
            
        if not self.look_forward:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_M = F.softmax(wei, dim=-1) 

        attention = attention_M
        # it is common to drop out after softmax as well; drop some contributions etc
         # randomly prevent the tokens from communicating with each other 
        if self.dropout is not None:
            attention = self.dropout(attention_M)

        return (attention_M, attention @ value)
    
class DeBERTaHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, use_dropout = False):
        super().__init__()
        self.head_size = head_size
        self.content_key = nn.Linear(n_embd, head_size, bias=False)
        self.content_query = nn.Linear(n_embd, head_size, bias=False)
        self.position_key = nn.Linear(n_embd, head_size, bias=False)
        self.position_query = nn.Linear(n_embd, head_size, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        '''
        x: (Tensor): (B, T, n_embd) 
        Return:
            attention: (B, block_size, block_size)
            results: (B, block_size, head_size)
        '''
        _, T, _ = x.shape
        content = x
        position = positional_encoding
        content_key = self.content_key(content)
        content_query = self.content_query(content)

        position_key = self.position_key(position)
        position_query = self.position_query(position)
        value = self.value(content) * (3 * self.head_size)**-0.5

        wei = content_query @ content_key.transpose(-1, -2) + position_query @ content_key.transpose(-1, -2) + content_query @ position_key.transpose(-1, -2)   
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_M = F.softmax(wei, dim=-1) 

        attention = attention_M
        # it is common to drop out after softmax as well; drop some contributions etc
         # randomly prevent the tokens from communicating with each other 
        if self.dropout is not None:
            attention = self.dropout(attention)

        return (attention_M, attention @ value)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, use_dropout=False, look_forward=False):
        super().__init__()
        if use_deberta:
            self.heads = nn.ModuleList([DeBERTaHead(n_embd, head_size, block_size=block_size) for _ in range(n_head)])
        else:
            self.heads = nn.ModuleList([Head(n_embd, head_size, block_size=block_size, look_forward=look_forward) for _ in range(n_head)])

        # add a projection layer; because you need a projection layer before connecting back to the residual path way
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        '''
        x: (Tensor) (B, block_size, n_embd) or (B, T, 2, n_embd) for DeBERTa
        Return:
            attension_M (Tensor) (n_head, B, block_size, block_size)
            attension (Tensor) (B, block_size, n_embd)
        '''
        attention = [h(x) for h in self.heads]
        attention_M = [m for m, _ in attention] # n_head * (B, block_size, block_size)
        attention = [att for _ , att in attention]
        out = torch.concat(attention, dim=-1)
        out = self.proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.stack(attention_M), out
    
class FeedFoward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            # see the original Attention is All you Need Papar
            nn.Linear(n_embd, n_hidden), # the inner layer of the feedforward layer is 4 times the size
            nn.ReLU(),
            # add a projection layer; because you need a projection layer before connecting back to the residual path way
            nn.Linear(n_hidden, n_embd),
            # nn.Dropout(dropout) # it is common to add dropout layer right before the residual connection
        )
    def forward(self, x):
        '''
        x (Tensor) (B, block_size, n_embd)
        ''' 
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size, n_hidden, look_forward=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head=n_head, head_size=head_size, n_embd=n_embd, block_size=block_size, look_forward=look_forward)
        self.ffw = FeedFoward(n_embd=n_embd, n_hidden=n_hidden)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        '''
        x (Tensor) (B, block_size, n_embed)
        '''
        x = self.ln1(x)
        attention_M, attension = self.sa(x)
        x = x + attension
        x = x + self.ffw(self.ln2(x))
        return attention_M, x


class CLSModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, n_classes, n_hidden):
        super().__init__()
        print(f"embedding dimension: {n_embd}, block size: {block_size}")

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([Block(n_head=n_head, n_embd=n_embd, block_size=block_size, n_hidden=n_hidden, look_forward=True) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)

        # classifier
        self.proj = nn.Linear(n_embd, n_classes)

        self.Ms = None
    def forward(self, x):
        _, block_size = x.shape
        embedding = self.embedding(x) # (B, block_size, n_embd)
        positional_encoding = self.positional_encoding(torch.arange(block_size))
        x = embedding + positional_encoding
        Ms = [] # n_blocks * (n_head, B, block_size, block_size)
        for blk in self.blocks:
            M, res = blk(x)
            x = res
            Ms.append(M)

        x = self.ln(x) # (B, block_size, n_embd)
        x = torch.mean(x, dim=1) # (B, n_embd)
        x = self.proj(x) # (B, n_classes)

        self.Ms = torch.stack(Ms)
        return x # (B, n_classes)

        
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, n_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.blks = nn.ModuleList([Block(n_head, n_embd, block_size, n_hidden, look_forward=False) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        
        # self.fc = nn.Linear(n_embd, n_hidden)
      
        self.proj = nn.Linear(n_embd, vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.Ms = None
    def forward(self, x, y):
        B, T = x.shape
        embedding = self.embedding(x) # (B, T, n_embd)
        positional_encoding = self.positional_encoding(torch.arange(T))
        x = embedding + positional_encoding

        Ms = []
        for blk in self.blks:
            M, res = blk(x)
            x = res
            Ms.append(M)

        x = self.ln(x)
        x = self.proj(x) # (B, T, vocab_size)
        self.Ms = torch.stack(Ms)
        return self.criterion(x.view(B*T, -1), y.view(B*T))


class ALiBi(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, n_hidden):
        super().__init__()
        global use_linear_bias
        use_linear_bias = True

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blks = nn.ModuleList([Block(n_head, n_embd, block_size, n_hidden, look_forward=False) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        
        # self.fc = nn.Linear(n_embd, n_hidden)
      
        self.proj = nn.Linear(n_embd, vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.Ms = None
    def forward(self, x, y):
        B, T = x.shape
        x = self.embedding(x) # (B, T, n_embd)

        Ms = []
        for blk in self.blks:
            M, res = blk(x)
            x = res
            Ms.append(M)

        x = self.ln(x)
        x = self.proj(x) # (B, T, vocab_size)
        self.Ms = torch.stack(Ms)
        return self.criterion(x.view(B*T, -1), y.view(B*T))


class DeBERTa(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, n_hidden):
        super().__init__()
        global use_deberta
        use_deberta = True

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blks = nn.ModuleList([Block(n_head, n_embd, block_size, n_hidden, look_forward=False) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        
        # self.fc = nn.Linear(n_embd, n_hidden)
      
        self.proj = nn.Linear(n_embd, vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.Ms = None
    def forward(self, x, y):
        B, T = x.shape
        x = self.embedding(x) # (B, T, n_embd)
        global positional_encoding # Note: this will only work for sequential training!!! It does not work for GPU training
        positional_encoding = self.position_embedding(torch.arange(T))

        Ms = []
        for blk in self.blks:
            M, res = blk(x)
            x = res
            Ms.append(M)

        x = self.ln(x)
        x = self.proj(x) # (B, T, vocab_size)
        self.Ms = torch.stack(Ms)
        return self.criterion(x.view(B*T, -1), y.view(B*T))