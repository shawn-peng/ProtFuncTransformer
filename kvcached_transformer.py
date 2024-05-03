# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt

from cache_session import CacheSession

warnings.simplefilter("ignore")
print(torch.__version__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DEBUGGING = True
DEBUGGING = False
if DEBUGGING:
    dbg_figs = ['encoder-decoder attention']
    figs = {k: plt.figure() for k in dbg_figs}
    plt.ion()
    plt.show()

dropout_rate = 0.1


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, name=None):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.name = name

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out


# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim, name=None):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim
        self.name = name

        pe = torch.zeros(max_seq_len, embed_model_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / self.embed_dim)))
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:seq_len, :], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, max_seqlen=64, name=None):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.name = name
        self.max_seqlen = max_seqlen

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim,
                                      bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def init_kvcache(self, key, value, cache: CacheSession):
        batch_size = key.size(0)
        seq_length = key.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads,
                       self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        k = self.key_matrix(key)  # (32x10x8x64)
        v = self.value_matrix(value)

        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        att_cache = torch.zeros((batch_size, self.n_heads, self.max_seqlen, self.single_head_dim))
        res_cache = torch.zeros((batch_size, self.max_seqlen, self.embed_dim))

        cache.new_cache(self.name, (k, v, att_cache, res_cache))

        # cache.new_cache(self.name, {
        #     'key': key,
        #     'value': value,
        #     'att_cache': att_cache,
        #     'res_cache': res_cache,
        # })

    def _forward_w_kvcache(self, query, kvcache, mask=None):
        assert not self.training
        k, v, att_cache, res_cache = kvcache
        # {k, v}.shape = batch_size x num_heads x seq_len x head_embedding_dim

        batch_size = k.size(0)
        dk = k.size(2)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        ind = seq_length_query - 1

        # 32x10x512
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        # k = self.key_matrix(key)
        vq = self.query_matrix(query[:, -1:, :, :])  # batch_size x 1 x num_heads x head_embedding_dim
        # v = self.value_matrix(value)

        vq = vq.transpose(1, 2)  # batch_size x 1 x num_heads x head_embedding_dim

        # dk = key.shape[-2]
        # attention.shape = batch_size x num_heads x seq_len_query x seq_len_key
        vatt = torch.matmul(vq, k.transpose(-1, -2))
        vatt /= math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        vatt = F.softmax(vatt, dim=-1)
        # vatt.shape = batch_sizse x num_heads x 1 x seq_len_key
        # dimension of cache may be bigger than the key, so update up to 'dk'
        att_cache[:, :, ind:ind + 1, :dk] = vatt
        scores = torch.matmul(vatt, v)  # (32 x 8 x 1 x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 1 x 64)

        # concatenated output (merge heads outputs)
        # (32x8x1x64) -> (32x1x8x64) -> (32,1,512)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, 1,
                                                          self.single_head_dim * self.n_heads)
        vout = self.out(concat)  # (32,1,512) -> (32,1,512)
        # result.shape = batch_size x seq_len_query x embedding_dim
        res_cache[:, ind:ind + 1, :] = vout
        return res_cache[:, :ind + 1, :]

    def forward(self, key, query, value, mask=None,
                cache: CacheSession = None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
           cache: cache session

        Returns:
           output vector from multihead attention
        """
        if cache is not None:
            if self.name not in cache:
                # init cache
                self.init_kvcache(key, value, cache)
            kvcache = cache.get_cache(self.name)
            return self._forward_w_kvcache(query, kvcache, mask=mask)

        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads,
                       self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  # (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)
        if DEBUGGING:
            if scores.shape[-1] != scores.shape[-2]:
                fig = figs['encoder-decoder attention']
                if not fig.axes:
                    fig.subplots(1, 1)
                ax = fig.axes[0]
                ax.imshow(scores[0, 0].detach())
                plt.pause(0.001)
                plt.ion()

        # mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                          self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, name=None):
        super(TransformerBlock, self).__init__()
        self.name = name

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads, name=f'{name}::attention_block')

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, key, query, value, cache: CacheSession = None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           cache: cache session
           norm2_out: output of transformer block

        """

        attention_out = self.attention(key, query, value, cache=cache)  # 32x10x512
        attention_residual_out = attention_out + query  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512

        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """

    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, name=None):
        super(TransformerEncoder, self).__init__()
        self.name = name

        self.embedding_layer = Embedding(vocab_size, embed_dim, name='src_embedding')
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim, name='src_positional_embedding')

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, expansion_factor, n_heads, f'{name}::encoder_transformer_block_{i}') for i in
             range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out  # 32x10x512


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, name=None):
        super(DecoderBlock, self).__init__()
        self.name = name

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8, name=f'{name}::causal_attention_block')
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads,
                                                  name=f'{name}::encoder_decoder_transformer_block')

    def _forward_w_cache(self, key, x, value, mask, cache):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """

        # we need to pass mask mask only to fst attention
        attention = self.attention(x, x, x, mask=mask, cache=cache)  # 32x10x512
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(key, query, value, cache=cache)

        return out

    def forward(self, key, x, value, mask, cache: CacheSession = None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """
        if cache is not None:
            return self._forward_w_cache(key, x, value, mask, cache)

        # we need to pass mask mask only to fst attention
        attention = self.attention(x, x, x, mask=mask)  # 32x10x512
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, name=None):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.name = name
        self.word_embedding = Embedding(target_vocab_size, embed_dim, name='tgt_embedding')
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim, name='tgt_positional_embedding')
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8, name=f'decoder_block_{i}')
                for i in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    def _forward_w_cache(self, x, enc_out, mask, cache):
        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask, cache)

        out = F.softmax(self.fc_out(x), dim=-1)

        return out

    def forward(self, x, enc_out, mask, cache=None):
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        if cache is not None:
            return self._forward_w_cache(x, enc_out, mask, cache)

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fc_out(x), dim=-1)

        return out


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8):
        super(Transformer, self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """

        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads, name='encoder')
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads, name='decoder')

    def encode(self, src):
        return self.encoder(src)

    def decode(self, trg, mem, cache: CacheSession = None):
        mask = self.make_trg_mask(trg)
        return self.decoder(trg, mem, mask, cache)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            1, 1, trg_len, trg_len
        )
        # returns the upper triangular part of matrix filled with ones
        # trg_mask = torch.triu(torch.ones((trg_len, trg_len))).expand(
        #     1, 1, trg_len, trg_len
        # )
        return trg_mask

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs
