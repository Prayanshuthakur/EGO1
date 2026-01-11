
import torch 
import torch.nn as nn
class CausalAttention(nn.Module):
    """
    this is the causal attention class 
    """
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        """
        Args:
            d_in (int): The dimension of the input embeddings.
            d_out (int): The dimension of the output query/key/value vectors.
            context_length (int): The length of the context (number of tokens).
            dropout (float): The dropout rate.
            qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.
        """
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias) # this will create a linear layer with d_in input features and d_out output features
        self.w_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self,inputs):
        """
        Args: 
            inputs (torch.Tensor): Input tensor of shape (batch, seq_len, d_in)

        Returns:
            torch.Tensor: Context vectors of shape (batch, seq_len, d_out)
        Implimentional Details:
        1. Projections: Convert the input embedding vectors into Query, Key, and Value vectors 
           by multiplying with their respective weight matrices.
           (Q = X @ W_q, K = X @ W_k, V = X @ W_v)
        2. Attention Scores: Calculate the raw attention scores by performing a dot product 
           between the Query and the transpose of the Key. This measures similarity/relevance.
           (Scores = Q @ K.T)
        3. Masking: Apply a mask to the attention scores to prevent the model from attending 
           to future tokens in the sequence.
           (Masked Scores = Scores * Mask)
        4. Scaling & Normalization: Scale the scores by the square root of the key dimension (d_out**0.5)
           to prevent vanishing gradients in the softmax. Then apply Softmax (dim=1) to obtain 
           normalized probability distributions (attention weights) that sum to 1 for each query.
           (Weights = softmax(Scores / sqrt(d_k)))
        5. Context Aggregation: Compute the final context vector by multiplying the attention weights 
           with the Value vectors. This aggregates information from relevant words based on the attention weights.
           (Context = Weights @ V)
        """
        B, T, _ = inputs.shape
        
        Q = self.w_query(inputs)  #Q=XWqT​+b
        K = self.w_key(inputs)    #K=XWKT​+b
        V = self.w_value(inputs)  #V=XWVT​+b

        scores = Q @ K.transpose(1, 2)
        scores = scores / (K.shape[-1] ** 0.5)

        mask = self.mask[:T, :T]
        # print("your mask is ",mask)
        scores = scores.masked_fill(mask.bool(), -float("inf"))
        # print("your scores is ",scores)
        weights = torch.softmax(scores, dim=-1)
        # print("your weights is ",weights)
        weights = self.dropout(weights)
        # print("your weights after dropout is ",weights)
        context = weights @ V
        return context



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
batch=torch.stack([inputs,inputs],dim=0)
print("your batch vector is ",batch)
print(batch.shape)
causal_attention=CausalAttention(d_in=3,d_out=2,context_length=6,dropout=0.1)
context_vector=causal_attention(batch)
print("your context vector is ",context_vector)
