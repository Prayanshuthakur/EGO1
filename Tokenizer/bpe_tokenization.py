import tiktoken 

class BytePairEncoding:
    def __init__(self):
        self.tokenizer=tiktoken.get_encoding("gpt2")
    def encode(self,text):
        return self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
                                
    def decode(self,ids):
        return self.tokenizer.decode(ids)