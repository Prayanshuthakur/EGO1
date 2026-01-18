import tiktoken 

class BytePairEncoding:
    """
    Byte-Pair Encoding (BPE) Tokenizer
    
    This class wraps the `tiktoken` library, which provides a high-performance 
    implementation of the BPE algorithm used by GPT-2/3/4.
    
    Vocabulary Size: 50,257 tokens
    
    How it works:
        BPE iteratively merges the most frequent pair of bytes (or characters) 
        into a single new token. This allows the model to handle:
        1. Common words as single tokens (efficient)
        2. Rare words as sub-words (generalizable)
        3. Unknown characters as bytes (robust)
    """
    def __init__(self):
        # Load the pre-trained GPT-2 tokenizer from OpenAi's tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.n_vocab = self.tokenizer.n_vocab

    def encode(self, text):
        """
        Convert text string to list of integer token IDs.
        Example: "Hello world" -> [15496, 995]
        """
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                                
    def decode(self, ids):
        """
        Convert list of integer token IDs back to text string.
        Example: [15496, 995] -> "Hello world"
        """
        return self.tokenizer.decode(ids)