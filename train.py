from model import BigramLanguageModel, Config
from Tokenizer import Tokenizer
from dataloading import get_batch, tok
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_iters = 10000
eval_iters = 200
eval_interval = 1000
lr = 1e-4


m = BigramLanguageModel(Config)
model = m.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    
    for iter in range(train_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        }, "model.pt")

# train()

def generate():
    checkpoint = torch.load("model.pt")
    model = BigramLanguageModel(Config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    xb,yb = get_batch('val')
    pred_tokens = model.generate(xb, 1000)
    pred_tokens = pred_tokens[0].tolist()
    txt = tok.decode(pred_tokens)
    print(txt)
generate()

