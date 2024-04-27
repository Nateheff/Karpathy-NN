import requests
import torch
from Tokenizer import Tokenizer
response = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
text = response.text
from model import Config


tok = Tokenizer(300)
tok.train(text)
tokens = torch.tensor(tok.encode(text=text))
config = Config()
n = int(len(tokens) * 0.9)
train_data = tokens[:n]
val_data = tokens[n:]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ids = torch.randint(high=299, size=(50,))
# tok.decode(ids.tolist())

def get_batch(split):
   
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size, ))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

get_batch('val')