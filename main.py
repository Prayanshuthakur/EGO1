from pydoc import text
import token
import torch
from Tokenizer import sample_tokenizer,bpe_tokenization
import re
from Tokenizer.GptDatasetLoader import create_dataloader



def read_text(file_name):
    with open(file_name,"r",encoding="utf-8") as f:
        content_text=f.read()
    return content_text
def creat_tokens(book_content):
    all_words = re.split(r"(--\s|[.,!;:?_()\[\]\-]|\s+)", book_content)
    all_words=[item  for item in all_words if item.strip()]
    return all_words

def get_vocab(tokens):
    vocab={token:integer for integer,token in enumerate(tokens)}
    return vocab
def get_special_token_vocab(tokens):
    new_tokens=tokens
    new_tokens.extend(("endoftext","unk"))
    special_token_vocab={token:integer for integer,token in enumerate(new_tokens)} 
    return special_token_vocab

def test_sample_tokenizer(vocab):
    tokenizer=sample_tokenizer.SimpleTokenizer(vocab)
    new_text="She glanced out almost timorously at the terrace where her husband, lounging in a hooded chair"
    result=tokenizer.encode(new_text)
    print("your token_id's are",result)
    text_back=tokenizer.decode(result)
    print("your actual text back is that",text_back)

def test_special_context_tokens(special_token_vocab):
    print("your new special tokens lenght is",len(special_token_vocab))
    new_test_text="ramesh is very good boy and he works very well  rameh is the son of laxman he is from pebbair " # this is the example of the unknown words (out of context words)
    special_tokenizer=sample_tokenizer.SpecialContextTokens(special_token_vocab)
    special_ids=special_tokenizer.encode(new_test_text)
    print("your special text ids are",special_ids)
    actual_text=special_tokenizer.decode(special_ids)
    print("your actual text is ",actual_text)

def get_bpe_tokeizization_ids(raw_text):
    # test the functionality of the byte_pair encoding... 
    bpe_tokenizer=bpe_tokenization.BytePairEncoding()
    bpe_ids=bpe_tokenizer.encode(raw_text)
    actual_string=bpe_tokenizer.decode(bpe_ids)
    # print("actual string",actual_string[])
    print("your bpe id's are",bpe_ids[:10])
    return bpe_ids

def create_input_output_pairs(bpe_ids):
    context_length=4
    #prepring the input_output sequence
    bpe_tokenizer=bpe_tokenization.BytePairEncoding()
    for i in range(1,context_length+1):
        context=bpe_ids[:i]
        desired=bpe_ids[i] 
        print(context,"----->",desired)
        print(bpe_tokenizer.decode(context),"----->",bpe_tokenizer.decode([desired]))

def create_datataloaders(text_content):
    # use test the functionality of the datasetloaders
    dataloader=create_dataloader(text_content,batch_size=8,max_length=4,stride=1,suffle=True)
    data_iter=iter(dataloader)
    print("your data iter is ",data_iter)
    inputs,targets=next(data_iter)
    # print("inputs are",inputs)
    # print("target_id's are",targets)
    # print("inputs shapes are",inputs.shape) 
    # count=0
    # for item in data_iter:
    #     print("item is",item)
    #     if count>=10:
    #         break 
    #     count+=1
    return inputs,targets
def create_token_embeddings(input_ids,vocab_size=50257,model_dim=256):
    embedding_layer = torch.nn.Embedding(vocab_size,model_dim)
    token_embeddings=embedding_layer(input_ids)
    print("your shape of the token embeddings is ",token_embeddings.shape,"the emneddings are",token_embeddings)
    return token_embeddings
def get_positional_embeddings(max_length=4,output_dim=256):
    pos_embedding_layer=torch.nn.Embedding(max_length,output_dim)
    pos_ids = torch.arange(max_length)
    position_embeddings=pos_embedding_layer(pos_ids)
    return position_embeddings
def get_final_input_embeddings(token_embeddings,positional_embeddings):
    print("your token embddings size is",token_embeddings.shape)
    print("your positional embeddings size is",positional_embeddings.shape)
    return token_embeddings+positional_embeddings
if __name__=='__main__':
    text_file_name="test-dataset.txt"
    book_content=read_text(text_file_name)
    # tokens=creat_tokens(book_content)
    # vocab_dict=get_vocab(tokens)
    # special_vocab_dict=get_special_token_vocab(tokens)
    # test_sample_tokenizer(vocab_dict)
    # test_special_context_tokens(special_vocab_dict)
    # bpe_ids=get_bpe_tokeizization_ids(book_content)
    # create_input_output_pairs(bpe_ids[:10])
    inputs,target=create_datataloaders(book_content)
    token_embeddings = create_token_embeddings(inputs)
    pos_embeddings = get_positional_embeddings()
    final_input_embeddings=get_final_input_embeddings(token_embeddings,pos_embeddings)
    print("your final input_embeddings size is",final_input_embeddings.shape)






   