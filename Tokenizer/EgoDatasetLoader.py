from .bpe_tokenizer import BytePairEncoding
import torch
from torch.utils.data import Dataset,DataLoader

class SlidingWindowDataset(Dataset):
    """
    Sliding Window Dataset for Autoregressive Training.
    
    This dataset converts a long text (sequence of integers) into many training samples.
    
    Mechanism: "Sliding Window"
    We take a window of size `max_length` + 1 from the text.
    - Input: The first `max_length` tokens.
    - Target: The shifted version (offset by 1), predicting the next token.
    
    Args:
        text (str): The full training corpus string.
        max_length (int): Context length (how many tokens the model sees).
        stride (int): How much to move the window for the next sample.
        tokenizer: Instance of BPETokenizer (defaults to BytePairEncoding).
    """
    def __init__(self, text, max_length, stride, tokenizer=None) -> None:
        self.input_ids=[]
        self.target_ids=[] 
        
        # Use provided tokenizer or create new one
        if tokenizer is None:
            bpe_tokenizer = BytePairEncoding()
        else:
            bpe_tokenizer = tokenizer
        
        # Convert full text to integer tokens
        token_ids = bpe_tokenizer.encode(text) 
        
        # Sliding window approach to create (context, target) pairs
        for i in range(0, len(token_ids) - max_length, stride):
            # Input (Context): Tokens [i : i + max_length]
            context = token_ids[i : i + max_length]
            
            # Target (Prediction): Tokens [i+1 : i + max_length + 1]
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
    """
    Helper function to create a PyTorch DataLoader.
    
    Wraps the SlidingWindowDataset in a standard DataLoader for batching and shuffling.
    """
    dataset = SlidingWindowDataset(txt, max_length, stride, tokenizer)
    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataset_loader


