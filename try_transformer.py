#!/usr/bin/env python
# coding: utf-8
import os.path
# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pickle

# from toy_transformer import Transformer
from kvcached_transformer import Transformer
import torch

# In[3]:


import pandas as pd

# In[4]:


toydata = pd.read_csv('medium_articles/eng_de.csv', header=None, names=['en', 'de'])
toydata


# In[5]:


def gen_ufunc(nin, nout):
    return lambda f: np.frompyfunc(f, nin, nout)


# In[6]:


def reduce(f, arr, dims):
    r = arr
    for dim in dims:
        r = f.reduce(r, dim, keepdims=True)
    return r.squeeze()


# In[7]:


import numpy as np


# In[ ]:


# In[8]:


def gen_collect_func(f):
    @gen_ufunc(1, 1)
    def collect_cell(cell):
        return f(cell)

    return collect_cell


def collect_cell(df, f):
    res = reduce(gen_collect_func(f), df.to_numpy(), [0, 1])
    return res


# In[9]:


vocab = toydata.applymap(lambda x: x.split()).apply(np.sum).apply(set)

# In[10]:


src_vocab_size = len(vocab['en'])
target_vocab_size = len(vocab['de'])
num_layers = 6
seq_length = 12

# In[11]:


model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size,
                    target_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers=num_layers, expansion_factor=4, n_heads=8)
model

# In[13]:


# In[14]:


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

# In[15]:


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL[
    "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

TGT_LANGUAGE = 'de'
SRC_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

# In[16]:


token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# In[17]:


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for i, data_sample in data_iter:
        yield token_transform[language](data_sample[language])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_iter = toydata.iterrows()

    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


# In[18]:


# In[19]:


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# In[20]:


import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE

# In[21]:


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
model = Transformer(embed_dim=512, src_vocab_size=SRC_VOCAB_SIZE,
                    target_vocab_size=TGT_VOCAB_SIZE, seq_length=seq_length,
                    num_layers=3, expansion_factor=4, n_heads=8)
transformer = model

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# In[22]:


def pad_transform(tokens):
    return tokens + [PAD_IDX] * (seq_length - len(tokens))


# In[23]:


token_transform

# In[24]:


en_trans = token_transform['en']
en_voc_trans = vocab_transform['en']

# In[25]:


en_trans(toydata['en'].iloc[0])

# In[ ]:


# In[26]:


en_voc_trans(en_trans(toydata['en'].iloc[0]))

# In[27]:


pad_transform(en_voc_trans(en_trans(toydata['en'].iloc[0])))

# In[28]:


from torch.nn.utils.rnn import pad_sequence


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               # pad_transform,
                                               tensor_transform)  # Add BOS/EOS and create tensor


def pad_first_to_len(seqs, l):
    seqs[0] += [0] * (l - len(seqs[0]))


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    # pad_first_to_len(src_batch, seq_length)
    # pad_first_to_len(tgt_batch, seq_length)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# In[29]:


def ToyData(split='train'):
    if split == 'train':
        return toydata.to_numpy()[:20]
    elif split == 'valid':
        return toydata.to_numpy()[20:30]
    else:
        return toydata.to_numpy()[30:]


# In[30]:


from torch.utils.data import DataLoader


def generate_tokens(logits):
    tokens = torch.max(logits, dim=-1)[1]
    return tokens


def decode_tokens(tokens):
    words = vocab_transform[TGT_LANGUAGE].lookup_tokens(tokens.flatten().tolist())
    words = np.array(words).reshape(tokens.shape)
    return words


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_iter = ToyData(split='train')
    train_dataloader = DataLoader(train_iter, batch_size=20, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = model(src, tgt_input)
        # logits = model(src, tgt[:, :])
        tokens = generate_tokens(logits)

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]
        print(decode_tokens(tgt_out))
        print(decode_tokens(tokens))
        loss = loss_fn(logits.transpose(-1, -2), tgt_out)
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


# In[31]:


def evaluate(model):
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_iter = ToyData(split='valid')
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# In[33]:

continue_previous = True
# continue_previous = False
if continue_previous and os.path.exists('transformer.pickle'):
    transformer = pickle.load(open('transformer.pickle', 'rb'))


from timeit import default_timer as timer

NUM_EPOCHS = 180

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print(( f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

pickle.dump(transformer, open('transformer.pickle', 'wb'))


# In[ ]:


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        # tgt_mask = (generate_square_subsequent_mask(ys.size(0))
        #             .type(torch.bool)).to(DEVICE)
        # out = model.decode(ys, memory, tgt_mask)
        out = model.decode(ys, memory)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")

# In[ ]:


# In[ ]:
