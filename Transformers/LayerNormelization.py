import torch 
import torch.nn as nn 

class LayerNorm(nn.Module):
    def __init__(self,normalized_shape):
        super().__init__()
        self.epsilon=1e-5
        self.scale=nn.Parameter(torch.ones(normalized_shape))
        self.shift=nn.Parameter(torch.zeros(normalized_shape))
    def forward(self,inputs):
        mean=torch.mean(inputs,dim=-1,keepdim=True)
        print("your mean is ",mean)
        var=torch.var(inputs,dim=-1,keepdim=True)
        print("your var is ",var)
        norm_x=(inputs-mean)/torch.sqrt(var+self.epsilon)
        print("your norm_x is ",norm_x)
        return self.scale*norm_x+self.shift
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
layer_norm=LayerNorm(d_in)
context_vector=layer_norm(inputs)
print("your layer norm context vector is ",context_vector)
