from .bpe_tokenizer import BytePairEncoding
import torch
from torch.utils.data import Dataset,DataLoader

class EgoDataset(Dataset):
    def __init__(self, text, max_length, stride, tokenizer=None) -> None:
        self.input_ids=[]
        self.target_ids=[] 
        
        # Use provided tokenizer or create new one
        if tokenizer is None:
            bpe_tokenizer = BytePairEncoding()
        else:
            bpe_tokenizer = tokenizer
        
        token_ids = bpe_tokenizer.encode(text) 
        
        # Sliding window approach
        for i in range(0, len(token_ids) - max_length, stride):
            context = token_ids[i : i + max_length]
            target = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(context)
            self.target_ids.append(target)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return (torch.tensor(self.input_ids[idx]),  
        torch.tensor(self.target_ids[idx])
        )

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, tokenizer=None):
    dataset = EgoDataset(txt, max_length, stride, tokenizer)
    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataset_loader


