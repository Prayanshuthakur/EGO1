import torch 
import torch.nn as nn 

class OptimizedAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self,inputs):
        batch,num_tokens,d_in=inputs.shape
        query=self.w_query(inputs) # Shape: (b, num_tokens, d_out)
        key=self.w_key(inputs) # Shape: (b, num_tokens, d_out)
        value=self.w_value(inputs) # Shape: (b, num_tokens, d_out)
    
        # now we have to reshape the query, key and value
        query=query.view(batch,num_tokens,self.num_heads,self.head_dim) # Shape: (b, num_tokens, num_heads, head_dim)
        key=key.view(batch,num_tokens,self.num_heads,self.head_dim) # Shape: (b, num_tokens, num_heads, head_dim)
        value=value.view(batch,num_tokens,self.num_heads,self.head_dim) # Shape: (b, num_tokens, num_heads, head_dim)
        
        # now transpose those query, key and value to group the head dimension with the token dimension
        query=query.transpose(1,2) # Shape: (b, num_heads, num_tokens, head_dim)
        key=key.transpose(1,2) # Shape: (b, num_heads, num_tokens, head_dim)
        value=value.transpose(1,2) # Shape: (b, num_heads, num_tokens, head_dim)
        
        # now calculate the attention scores
        attention_scores=query @ key.transpose(2,3) # Shape: (b, num_heads, num_tokens, num_tokens)
        
        # apply the mask
        mask=self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores=attention_scores.masked_fill(mask,-float("inf"))

        # now scale the attention scores
        attention_scores=attention_scores/(self.head_dim**0.5)
        # now apply softmax
        attention_weights=torch.softmax(attention_scores,dim=-1)
        # now apply dropout
        attention_weights=self.dropout(attention_weights) 
        # now calculate the context vector
        context_vector=attention_weights @ value # Shape: (b, num_heads, num_tokens, head_dim)
        # now reshape the context vector
        context_vector=context_vector.transpose(1,2) # Shape: (b, num_tokens, num_heads, head_dim)
        # now combine the context vector
        context_vector=context_vector.contiguous().view(batch,num_tokens,d_out) # Shape: (b, num_tokens, d_out)
        
        # output projection
        context_vector = self.out_proj(context_vector)
        context_vector = self.resid_dropout(context_vector)
        return context_vector
        

inputs=torch.tensor(
    [
        [0.43, 0.15, 0.89],        # you
        [0.55, 0.87, 0.66],        # journey
        [0.57, 0.85, 0.64]        # starts
    ]
)
d_in=inputs.shape[1]
d_out=6
context_length=3
num_heads=2
batch=torch.stack([inputs,inputs],dim=0)
dropout=0.0
optimized_attention=OptimizedAttention(d_in,d_out,context_length,num_heads,dropout)
context_vector=optimized_attention(batch)
print("your context vector is ",context_vector)
