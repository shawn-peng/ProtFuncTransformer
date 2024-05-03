import pickle
import torch
import numpy as np
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# from toy_transformer import Transformer
from kvcached_transformer import Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TGT_LANGUAGE = 'de'
SRC_LANGUAGE = 'en'

toydata = pd.read_csv('medium_articles/eng_de.csv', header=None, names=['en', 'de'])

seq_length = 12

token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


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


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def pad_transform(tokens):
    return tokens + [PAD_IDX] * (seq_length - len(tokens))


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


# function to generate output sequence using greedy algorithm
'''
'''

from cache_session import CacheSession


def weighted_sample(weights):
    np.random.choice(np.arange(len(weights)), p=weights)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    cache = CacheSession()

    # memory = model.encode(src, src_mask)
    memory = model.encode(src)
    memory = memory.to(DEVICE)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        # tgt_mask = (generate_square_subsequent_mask(ys.size(0))
        #             .type(torch.bool)).to(DEVICE)
        cur_len = ys.shape[-1]
        # tgt_mask = (torch.triu(torch.ones(cur_len, cur_len))
        #             .type(torch.bool)).to(DEVICE)
        # out = model.decode(ys, memory, tgt_mask)
        out = model.decode(ys, memory, cache)
        # out = out.transpose(0, 1)
        # prob = model.generator(out[:, -1])
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1).T
    num_tokens = src.shape[1]
    src_mask = (torch.ones(num_tokens, num_tokens)).type(torch.bool)
    # tgt = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
    # model.decode(src.T, tgt.T)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")
    # return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt.cpu().numpy()))).replace("<bos>", "").replace(
    #     "<eos>", "")

testi = 32

transformer = pickle.load(open('transformer.pickle', 'rb'))
transformer.train(False)
print(toydata.iloc[testi])
out_s = translate(transformer, toydata.iloc[testi][SRC_LANGUAGE])

print(out_s)
