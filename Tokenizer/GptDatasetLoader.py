from token import tok_name
from .bpe_tokenization import BytePairEncoding
import torch
from torch.utils.data import Dataset,DataLoader
class GptDataset(Dataset):
    def __init__(self,text,max_lenght,stride) -> None:
        self.input_ids=[]
        self.target_ids=[] 
        bpe_tokenizer=BytePairEncoding()

        token_ids=bpe_tokenizer.encode(text) 
        # now implement the sliding window approach to make the things done 
        for i in range (0,len(token_ids)-max_lenght,stride):
            context=token_ids[i:i+max_lenght]
            target=token_ids[i+1:i+max_lenght+1]
            self.input_ids.append(torch.tensor(context))
            # print("context tensor is ",torch.tensor(context))
            self.target_ids.append(torch.tensor(target))
            # print("target tensor is",torch.tensor(target))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader(txt,batch_size=4,max_length=256,stride=128,suffle=True,drop_last=True,num_workers=0):
    dataset=GptDataset(txt,max_length,stride)
    dataset_loader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=suffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataset_loader
