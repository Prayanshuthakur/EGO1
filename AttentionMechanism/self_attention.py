import torch 
import torch.nn as nn 


def sample_self_attention(inputs):
    """
    this is the sample function for playing around before the actual class implimentation
    """
    x_2=inputs[1]
    d_in=inputs.shape[1]
    d_out=2 #in the gpt like models input and output dimension is same , 
    torch.manual_seed(123)
    w_query_2=nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    w_key_2=nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    w_value_2=nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
    print("your query weights matrix is ",w_query_2)
    print("your key weights matrix is ",w_key_2)
    print("your value weights matrix is ",w_value_2)
    query=inputs@w_query_2
    key=inputs@w_key_2
    value=inputs@w_value_2
    print("your query vector is ",query)
    print("your key vector is ",key)
    print("your value vector is ",value)
    # now we will calculate the attention scores
    query_2=query[1]
    atten_score_2=query_2@key.T # attention score for the given query and all other keys 
    atten_score_entire=query@key.T
    print("your attention score is ",atten_score_2)
    print("your entire attention score is ",atten_score_entire)
    # now we have to calculate the attention weights (which has to be normalized)
    # before the soft max we have to scale the attention scores with the square root of the dimension of the key
    # reason for this is to prevent the vanishing gradient problem when the dimension of the key is large
    atten_weights_2=torch.softmax(atten_score_2/d_out**0.5,dim=0)
    atten_weights_entire=torch.softmax(atten_score_entire/key.shape[-1]**0.5,dim=-1)
    print("your entire attention weights is ",atten_weights_entire)
    context_vector_2=atten_weights_2@value
    context_vector_entire=atten_weights_entire@value
    print("your context vector is ",context_vector_2)
    print("your entire context vector is ",context_vector_entire)

class SelfAttention(nn.Module):
    """
    Implements the Self-Attention mechanism using Query, Key, and Value matrices.
    """
    def __init__(self, d_input, d_out):
        """
        Initializes the learnable weight matrices for Query, Key, and Value projections.
        
        Conceptual Overview:
        In self-attention, we project the input vectors into three distinct spaces:
        1. Query (Q): Represents what the token is looking for.
        2. Key (K): Represents what the token offers/contains.
        3. Value (V): Represents the actual information content of the token.
        
        Args:
            d_input (int): The dimension of the input embeddings.
            d_out (int): The dimension of the output query/key/value vectors.
        """
        super().__init__()
        # Initialize learnable weights for Q, K, V
        torch.manual_seed(123)
        self.query = nn.Parameter(torch.rand(d_input, d_out))
        self.key = nn.Parameter(torch.rand(d_input, d_out))
        self.value = nn.Parameter(torch.rand(d_input, d_out))
        print("your query is ", self.query)
        print("your key is ", self.key)
        print("your value is ", self.value)

    def forward(self, inputs):
        """
        Performs the forward pass of the self-attention mechanism.

        Step-by-Step Logic:
        1. **Projections**: Convert the input embedding vectors into Query, Key, and Value vectors 
           by multiplying with their respective weight matrices.
           (Q = X @ W_q, K = X @ W_k, V = X @ W_v)
        
        2. **Attention Scores**: Calculate the raw attention scores by performing a dot product 
           between the Query and the transpose of the Key. This measures similarity/relevance.
           (Scores = Q @ K.T)
        
        3. **Scaling & Normalization**: Scale the scores by the square root of the key dimension (d_out**0.5)
           to prevent vanishing gradients in the softmax. Then apply Softmax (dim=1) to obtain 
           normalized probability distributions (attention weights) that sum to 1 for each query.
           (Weights = softmax(Scores / sqrt(d_k)))
        
        4. **Context Aggregation**: Compute the final context vector by multiplying the attention weights 
           with the Value vectors. This aggregates information from relevant words based on the attention weights.
           (Context = Weights @ V)

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, seq_len, d_input)

        Returns:
            torch.Tensor: Context vectors of shape (batch, seq_len, d_out)
        """
        # 1. Projections
        keys = inputs @ self.key
        values = inputs @ self.value
        queries = inputs @ self.query
        
        # 2. Attention Scores
        atten_scores = queries @ keys.T
        
        # 3. Scaling & Normalization
        d_out = keys.shape[-1]
        atten_weights = torch.softmax(atten_scores / d_out**0.5, dim=-1)
        
        # 4. Context Aggregation
        context_vector = atten_weights @ values
        return context_vector
inputs = torch.tensor(
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
print("your dimentions are",d_in,d_out)
sample_self_attention(inputs)
self_attention = SelfAttention(d_in, d_out)
context_vector = self_attention(inputs)
print("your context vector is ", context_vector)
