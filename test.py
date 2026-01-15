from utils.text_generation import generate_text_simple
from Model.ego_model import EgoModel
from config.ego_config import EGO_CONFIG_124M
from Tokenizer.bpe_tokenizer import BytePairEncoding
import torch

def text_to_token_ids(text):
    tokenizer=BytePairEncoding()
    encoded=tokenizer.encode(text)
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids):
    tokenizer=BytePairEncoding()
    text=tokenizer.decode(token_ids[0].tolist())
    return text

batch = []
txt1 = "How are you"
tokenizer=BytePairEncoding()
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch = torch.stack(batch, dim=0)
print(batch)
EGO_CONFIG_124M={
    "vocab_size": 50257,      # BPE tokenizer vocabulary size
    "context_length": 128,   # Maximum context window
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 8,            # Number of attention heads
    "n_layers": 8,           # Number of transformer layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # No bias in QKV projections (GPT-2 style)
}
print("your batch shape is",batch.shape)
model=EgoModel(EGO_CONFIG_124M)
output=generate_text_simple(model,batch,20,128)
print(output.shape)
# i don't want to include the input tokens in the output
print(output[:,batch.shape[1]:])
# now decode the ids to text 
output_text=token_ids_to_text(output[:,batch.shape[1]:])
print(output_text)