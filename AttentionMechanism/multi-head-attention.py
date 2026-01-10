import torch 
import torch.nn as nn 
from Causal_Attention import CausalAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        """
        Args:
            d_in (int): The dimension of the input embeddings.
            d_out (int): The dimension of the output query/key/value vectors.
            context_length (int): The length of the context (number of tokens).
            dropout (float): The dropout rate.
            num_heads (int): The number of attention heads.
            qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.
        """
        super().__init__()
        self.heads=nn.ModuleList([
            CausalAttention(d_in,d_out,context_length,dropout,qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self,inputs):
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, seq_len, d_in)

        Returns:
            torch.Tensor: Context vectors of shape (batch, seq_len, d_out)
        """
        print("your context vector for this input is ",[head(inputs) for head in self.heads])
        return torch.cat([head(inputs) for head in self.heads],dim=-1)

inputs=torch.tensor(
    [
        [0.43, 0.15, 0.89],        # you
        [0.55, 0.87, 0.66],        # journey
        [0.57, 0.85, 0.64],        # starts
        [0.22, 0.58, 0.33],        # with
        [0.77, 0.22, 0.10],        # one
        [0.05, 0.80, 0.55]         # step
    ]
)
d_in=inputs.shape[1]
d_out=2
context_length=6
num_heads=2
batch=torch.stack([inputs,inputs],dim=0)

dropout=0.0
multi_head_attention=MultiHeadAttention(d_in,d_out,context_length,dropout,num_heads)
context_vector=multi_head_attention(batch)
print("your multi-head context vectors is ",context_vector)