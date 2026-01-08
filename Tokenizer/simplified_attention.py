import torch

def attention_for_single_word():
    """
    n this we are going to implement the simplified attention 
    first we will start with a simple sentence 
    """
    sentence="your journey starts with one step"
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
    query=inputs[1]
    attention_scores=torch.empty(inputs.shape[0])
    for i,i_x in enumerate(inputs):
        torch_value=torch.dot(i_x,query)
        attention_scores[i]=torch_value
    print("your attension score is ",attention_scores)

    # now we have to do normalize the attention scores and there are two ways to do so 
    # one is the naive way and the other is the efficient way 
    def naive_soft_max(x):
        return torch.exp(x)/torch.exp(x).sum(dim=0)
    naive_softmax_weights=naive_soft_max(attention_scores)
    # print("your naive soft max is ",naive_softmax_weights)
    actual_softmax_weights=torch.softmax(attention_scores,dim=0)
    # print("your actual soft max is ",actual_softmax_weights)
    context_vector=torch.zeros(query.shape)
    for ind,emb_vector in enumerate(inputs):
        context_vector+=emb_vector*actual_softmax_weights[ind]
    print("your final context vector is ",context_vector)
    return context_vector

def attention_for_entire_sentence():
    """
    in this we are going to implement the attention for the entire sentence 
    """
    sentence="your journey starts with one step"
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
    attention_scores=inputs@inputs.T
    # print("your attension score is ",attention_scores)
    attention_weights=torch.softmax(attention_scores,dim=1)
    # print("your attention weights is ",attention_weights)
    # now calculate the context vectors
    context_vector=attention_weights@inputs
    print("your context vector is ",context_vector)
    return context_vector
        

context_vector_for_single_word=attention_for_single_word()
attention_for_entire_sentence()


