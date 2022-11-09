import gzip
import random
import wandb
import argparse
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from palm_pytorch.palm_flash_cosine_sim import PaLM_flash
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper

wandb.init(project="my-test-project")

# arguments

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda-kernel', default = False, action = 'store_true')
parser.add_argument('--use-float32', default = False, action = 'store_true')
parser.add_argument('--seq-len', default = 1024, type = int)
args = parser.parse_args()

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 10
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 8192
USE_AMP = not args.use_float32

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# instantiate GPT-like decoder model

model = PaLM_flash(num_tokens=256, dim=512, depth=8)

model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)
model.cuda()

# prepare enwik8 data

with gzip.open("./examples/enwik8_deepspeed/data/enwik8.gz") as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = GradScaler(enabled = USE_AMP)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()
    optim.zero_grad()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        with autocast(enabled = USE_AMP):
            loss = model(next(train_loader))
        scaler.scale(loss / GRADIENT_ACCUMULATE_EVERY).backward()

    wandb.log({"train loss": loss})
    print(f"training loss: {loss.item()}")

    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optim)
    scaler.update()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            wandb.log({"val loss": loss})
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)
