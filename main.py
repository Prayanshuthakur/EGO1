
from pydoc import text
from Tokenizer import sample_tokenizer,bpe_tokenizer
import re
with open("test-dataset.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
all_words = re.split(r"(--\s|[.,!;:?_()\[\]\-]|\s+)", raw_text)
# all_words=[item  for item in all_words if item.strip()]
# # to test the functionality of the normal tokenization... 
# vocab={token:integer for integer,token in enumerate(all_words)}
# tokenizer=sample_tokenizer.SimpleTokenizer(vocab)
# new_text="She glanced out almost timorously at the terrace where her husband, lounging in a hooded chair"
# result=tokenizer.encode(new_text)
# print("your token_id's are",result)
# text_back=tokenizer.decode(result)
# print("your actual text back is that",text_back)

# to test the functionality of the special context tokenization 
# new_tokens=all_words
# new_tokens.extend(("endoftext","unk"))
# special_token_vocab={token:integer for integer,token in enumerate(new_tokens)}
# print("your new special tokens lenght is",len(special_token_vocab))
# new_test_text="ramesh is very good boy and he works very well  rameh is the son of laxman he is from pebbair " # this is the example of the unknown words (out of context words)
# special_tokenizer=sample_tokenizer.SpecialContextTokens(special_token_vocab)
# special_ids=special_tokenizer.encode(new_test_text)
# print("your special text ids are",special_ids)


# test the functionality of the byte_pair encoding... 
bpe_tokenizer=bpe_tokenizer.BytePairEncoding()
text="Hello, Ramesh do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace. in a winter season"

bpe_ids=bpe_tokenizer.encode(text)
print("your bpe_ide's are",bpe_ids)
actual_string=bpe_tokenizer.decode(bpe_ids)
print("actual string",actual_string)