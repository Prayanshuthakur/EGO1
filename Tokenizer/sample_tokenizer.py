import re
class SimpleTokenizer:
    """
    this class is to test the functionality of a base simple tokenizer
    """
    def __init__(self,vocab) -> None:
        self.str_to_int=vocab 
        self.int_to_str={i:s for s,i in vocab.items()}

    def encode(self,text):
        all_words = re.split(r"(--\s|[.,!;:?_()\[\]\-]|\s+)", text)
        all_words=[item  for item in all_words if item.strip()]
        ids=[self.str_to_int[s] for s in all_words]
        return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        return text 

class SpecialContextTokens:
    """
    this class is to test the functionality of a base tokenizer with the special 
    context tokens which handles the out of vocabulary
    """
    def __init__(self,vocab) -> None:
        self.str_to_int=vocab 
        self.int_to_str={i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r"(--\s|[.,!;:?_()\[\]\-]|\s+)", text)
        preprocessed=[item  for item in preprocessed if item.strip()]
        preprocessed=[item if item in self.str_to_int else "unk" for item in preprocessed]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        return text