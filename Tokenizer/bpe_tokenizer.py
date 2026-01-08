import tiktoken 

class BytePairEncoding:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.n_vocab = self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                                
    def decode(self, ids):
        return self.tokenizer.decode(ids)