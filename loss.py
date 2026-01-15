
import torch.nn.functional as F
import torch
from Tokenizer.bpe_tokenizer import BytePairEncoding
from Model.ego_model import EgoModel
from Tokenizer.EgoDatasetLoader import create_dataloader

def cal_batch_loss(inputs,targets,model,device):
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss

def cal_loss(train_dataloader,model,device,num_batches=None):
    total_loss=0
    if len(train_dataloader)==0:
        return float('nan')
    if num_batches is None:
        num_batches=len(train_dataloader)
    else:
        num_batches=min(num_batches,len(train_dataloader))
    for i,(inputs,targets) in enumerate(train_dataloader):
        if i>=num_batches:
            break
        loss=cal_batch_loss(inputs,targets,model,device)
        total_loss+=loss.item()
    return total_loss/num_batches



def main():
    """Main function to calculate loss."""
    text_path = "test-dataset.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # train test split 
    train_ratio = 0.9
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    valid_text = text[split_idx:]
    # now create the dataloader
    tokenizer = BytePairEncoding()
    train_dataloader = create_dataloader(
        train_text,
        batch_size=2,
        max_length=256,
        stride=256,
        shuffle=False,
        tokenizer=tokenizer
    )
    valid_dataloader = create_dataloader(
        valid_text,
        batch_size=2,
        max_length=256,
        stride=256,
        shuffle=False,
        tokenizer=tokenizer
    )
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 2,
        "max_length": 256,
        "batch_size": 2,
        "drop_rate": 0.1,
        "qkv_bias":False
    }
    input_tokens=0
    for input_batch,target in train_dataloader:
        input_tokens+=input_batch.numel()
    print(input_tokens)
    target_tokens=0
    for input_batch,target in valid_dataloader:
        target_tokens+=target.numel()
    print(target_tokens)
    model = EgoModel(GPT_CONFIG_124M)
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loss=cal_loss(train_dataloader,model,device)
    valid_loss=cal_loss(valid_dataloader,model,device)
    print(f"Train Loss: {train_loss}")
    print(f"Valid Loss: {valid_loss}")
    
if __name__ == "__main__":
    main()