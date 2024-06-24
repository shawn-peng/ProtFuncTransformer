#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import pandas as pd
import numpy as np
import Bio
import Bio.SeqIO
import re
import os
import pickle

from timeit import default_timer as timer
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[3]:


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# In[4]:


hyper_aa_regex = re.compile('[BXZJUO]')


# In[5]:


MAX_LEN = 20


# In[6]:


import torch
x = torch.tensor([1, 2, 3])
x.to('cuda')


# In[7]:


# seq_file = "data/uniprot_sprot.fasta"
seq_file = "data/debugging_sequence.fasta"


seqdb_pickle = 'states/seqdb.pickle'
if os.path.exists(seqdb_pickle):
    seqdb = pickle.load(open(seqdb_pickle, 'rb'))
else:
    seqdb = {}
    irregs = 0
    for record in Bio.SeqIO.parse(seq_file, "fasta"):
        if '|' in record.id:
            _, acc, geneid = record.id.split('|')
        else:
            acc = record.id
        if hyper_aa_regex.findall(str(record.seq)):
            irregs += 1
            continue
        if len(record.seq) > MAX_LEN:
            irregs += 1
            continue
        seqdb[acc] = record
    print('irregs', irregs)
    pickle.dump(seqdb, open(seqdb_pickle, 'wb'))
    seqdb = pickle.load(open(seqdb_pickle, 'rb'))


# In[8]:


len(seqdb)


# In[ ]:





# In[9]:


def build_datapoint(s):
    return ['>'] + list(s) + ['<']


# In[10]:


def build_dataset(seqdb):
    res = []
    for acc, rec in seqdb.items():
        res.append(build_datapoint(str(rec.seq)))
    return res


# In[11]:


for rec in seqdb.values():
    if 'B' in rec.seq:
        print(rec)
        break


# In[12]:


# ds = build_dataset(seqdb)
# ds


# In[13]:


# vocab = set(c for s in ds for c in s)
# vocab =


# In[14]:




# In[15]:




# In[16]:


from toy_transformer import *


# In[17]:


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['X', '_', '>', '<']


base_vocab_size = 5
# src_vocab_size = len(vocab) + len(special_symbols)
# target_vocab_size = len(vocab) + len(special_symbols)
src_vocab_size = base_vocab_size + len(special_symbols)
target_vocab_size = base_vocab_size + len(special_symbols)
# num_layers = 10
seq_length = 22


# In[18]:


torch.triu(torch.ones((5, 5), device=DEVICE))


# In[19]:


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
def generate_square_diagnal_mask(sz):
    mask = (torch.diag(torch.ones(sz, device=DEVICE)))
    # mask = (torch.diag(torch.zeros(sz, device=DEVICE)))
    mask = 1 - mask
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# In[20]:


generate_square_subsequent_mask(5)


# In[21]:


generate_square_diagnal_mask(5)


# In[22]:


def create_one_out_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def target_one_out_mask(tgt):
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_diagnal_mask(tgt_seq_len)

    return tgt_mask


# In[23]:


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List


# In[ ]:





# In[24]:


def yield_tokens(data_iter: Iterable) -> List[str]:
    for data_sample in data_iter:
        yield list(data_sample)
    


# In[25]:


# In[26]:


train_iter = seqdb.values()
vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter),
                                            min_freq=1,
                                            specials=special_symbols,
                                            special_first=True)


# In[27]:


vocab_transform.set_default_index(UNK_IDX)


# In[28]:


token_transform = list


# In[29]:


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# In[30]:


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))
def pad_transform(tokens):
    return tokens + [PAD_IDX] * (seq_length - len(tokens))

text_transform = sequential_transforms(token_transform,  # Tokenization
                                       vocab_transform,  # Numericalization
                                       # pad_transform,
                                       tensor_transform)


# In[31]:


model = Transformer(embed_dim=8, src_vocab_size=src_vocab_size,
                    target_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers=4, expansion_factor=4, n_heads=1,
                    target_mask_fn=target_one_out_mask)


# In[32]:


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_batch.append(text_transform(sample))
        tgt_batch.append(text_transform(sample))

    # pad_first_to_len(src_batch, seq_length)
    # pad_first_to_len(tgt_batch, seq_length)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch



# In[33]:


transformer = model


# In[34]:


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


# In[35]:


transformer = transformer.to(DEVICE)


# In[36]:


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# In[37]:


optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# In[38]:


token_transform(next(iter(seqdb.values())))


# In[39]:


def generate_tokens(logits):
    tokens = torch.max(logits, dim=-1)[1]
    return tokens


def decode_tokens(tokens):
    words = vocab_transform.lookup_tokens(tokens.flatten().tolist())
    words = np.array(words).reshape(tokens.shape)
    return words



# In[40]:


dataset = list(seqdb.values())


# In[41]:

BATCH_SIZE = 20


def evaluate(model):
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # val_iter = ToyData(split='valid')
    val_iter = dataset
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:, :-1]

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_one_out_mask(src, tgt_input)

        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# In[42]:


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_iter = dataset
    train_dataloader = DataLoader(train_iter, batch_size=20, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        print(src.shape, tgt.shape)
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        # tgt_input = tgt[:, :-1]

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_one_out_mask(src, tgt)

        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = model(src, tgt)
        # logits = model(src, tgt[:, :])
        tokens = generate_tokens(logits)

        optimizer.zero_grad()

        # tgt_out = tgt[:, 1:]
        tgt_out = tgt[:, :]
        print(decode_tokens(tgt_out))
        print(decode_tokens(tokens))
        loss = loss_fn(logits.transpose(-1, -2), tgt_out)
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))



# In[43]:


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[ ]:





# In[44]:


continue_previous = True
pickle_file = 'prot_seq_transformer.pickle'
# continue_previous = False
if continue_previous and os.path.exists(pickle_file):
    transformer = pickle.load(open(pickle_file, 'rb'))


from timeit import default_timer as timer

NUM_EPOCHS = 1800

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print(( f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

pickle.dump(transformer, open(pickle_file, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




