import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

# importing required libraries
import math, copy, re
import warnings
# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

DEBUGGING = False
# DEBUGGING = True
if DEBUGGING:
    dbg_figs = ['encoder-decoder attention']
    figs = {k: plt.figure() for k in dbg_figs}
    plt.ion()
    plt.show()

dropout_rate = 0.1

key = jax.random.PRNGKey(0)


class Embedding(nn.Module):
    vocab_size: int
    embed_dim: int
    embedding: Optional[jnp.ndarray] = None

    def setup(self):

        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        if self.embedding is not None:
            # assert embedding.shape == (vocab_size, embed_dim)
            self.embed = nn.Embed(self.vocab_size, self.embed_dim, embedding_init=lambda *args: self.embedding)
            assert True
        else:
            self.embed = nn.Embed(self.vocab_size, self.embed_dim)

    def __call__(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out

    def hidden_states(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: {'': embedding vector}
        """
        out = self.embed(x)
        return {'': out}


class PositionalEmbedding(nn.Module):
    max_seq_len: int
    embed_model_dim: int
    def setup(self):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        # super(PositionalEmbedding, self).__init__()
        self.embed_dim = self.embed_model_dim

        pe = np.zeros((self.max_seq_len, self.embed_model_dim))
        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / self.embed_dim)))
        # pe = pe.unsqueeze(0)
        # self.pe = jnp.array(pe)
        self.pe = jax.device_put(pe)
        # self.register_buffer('pe', pe)

    def __call__(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :]
        return x

    def hidden_states(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :]
        return {'': x}


class MultiHeadAttention(nn.Module):
    embed_dim: int=512
    n_heads: int=8
    def setup(self):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Dense(self.single_head_dim)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Dense(self.single_head_dim)
        self.value_matrix = nn.Dense(self.single_head_dim)
        # self.out = nn.Dense(self.n_heads * self.single_head_dim, self.embed_dim)
        self.out = nn.Dense(self.embed_dim)

    def __call__(self, key, query, value, mask=None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder

        Returns:
           output vector from multihead attention
        """
        batch_size = key.shape[0]
        seq_length = key.shape[1]

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.shape[1]

        # 32x10x512 => (32x10x8x64)
        key = key.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)
        value = value.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        # 10 -> seq_len
        k_adjusted = k.transpose(0, 1, 3, 2)  # (batch_size, n_heads, single_head_dim, seq_len)  #(32 x 8 x 64 x 10)
        product = q @ k_adjusted  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        # product = jnp.dot(q, k_adjusted)
        print(q.shape, k_adjusted.shape, product.shape)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            # product = product.masked_fill(mask == 0, float("-1e20"))
            product = jnp.where(mask == 0, float("-1e20"), product)

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = nn.softmax(product, axis=-1)
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
        # scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)
        scores = scores @ v

        # concatenated output
        concat = scores.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length_query, self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output

    def hidden_states(self, key, query, value,
                      mask=None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        """
        batch_size = key.shape[0]
        seq_length = key.shape[1]

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.shape[1]


        # 32x10x512 => (32x10x8x64)
        key = key.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)
        value = value.reshape(batch_size, seq_length, self.n_heads, self.single_head_dim)


        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        states = {'key': key, 'query': query, 'value': value,
                    'k': k,       'q': q,         'v': v}

        q = q.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(0, 1, 3, 2)  # (batch_size, n_heads, single_head_dim, seq_len)  #(32 x 8 x 64 x 10)
        product = q @ k_adjusted  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        # product = jnp.dot(q, k_adjusted)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            # product = product.masked_fill(mask == 0, float("-1e20"))
            product = jnp.where(mask == 0, float("-1e20"), product)

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        states['attention'] = product

        # applying softmax
        scores = nn.softmax(product, axis=-1)
        if DEBUGGING:
            if scores.shape[-1] != scores.shape[-2]:
                fig = figs['encoder-decoder attention']
                if not fig.axes:
                    fig.subplots(1, 1)
                ax = fig.axes[0]
                ax.imshow(scores[0, 0].detach())
                plt.pause(0.001)
                plt.ion()

        states['attention_softmax'] = scores

        # mutiply with value matrix
        # scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)
        scores = scores @ v

        states['v_extracted'] = scores

        # concatenated output
        concat = scores.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length_query, self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)
        states['out'] = output

        states[''] = output
        return states


class TransformerBlock(nn.Module):
    embed_dim: int
    expansion_factor: int=4
    n_heads: int=8
    def setup(self):
        super(TransformerBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(self.embed_dim, self.n_heads)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        self.feed_forward = nn.Sequential([
            nn.Dense(self.expansion_factor * self.embed_dim),
            nn.relu,
            nn.sigmoid,
            nn.Dense(self.embed_dim)
        ])

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def __call__(self, key, query, value, train=True):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """
        attention_out = self.attention(key, query, value)  # 32x10x512
        attention_residual_out = attention_out + query  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out), deterministic=not train)  # 32x10x512

        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out), deterministic=not train)  # 32x10x512

        return norm2_out

    def hidden_states(self, key, query, value):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        states = {}
        extend_states(states, 'attention',
                      self.attention.hidden_states(key, query, value)  # 32x10x512
                      )
        attention_residual_out = states['attention'] + query  # 32x10x512
        states['attention_residual'] = attention_residual_out
        states['norm1'] = self.norm1(attention_residual_out)  # 32x10x512
        states['dropout1'] = self.dropout1(states['norm1'], deterministic=False)

        states['feed_fwd'] = self.feed_forward(states['dropout1'])  # 32x10x512 -> #32x10x2048 -> 32x10x512
        states['feed_fwd_residual'] = states['feed_fwd'] + states['norm1']  # 32x10x512
        states['norm2'] = self.norm2(states['feed_fwd_residual'])
        states['dropout2'] = self.dropout2(states['norm2'], deterministic=False)

        states[''] = states['dropout2']
        return states


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
    seq_len: int
    vocab_size: int
    embed_dim: int
    num_layers: int=2
    expansion_factor: int=4
    n_heads: int=8
    word_embedding: Optional[jnp.ndarray]=None

    def setup(self):
        self.embedding_layer = Embedding(self.vocab_size, self.embed_dim, self.word_embedding)
        self.positional_encoder = PositionalEmbedding(self.seq_len, self.embed_dim)

        self.layers = [
            TransformerBlock(
                self.embed_dim,
                self.expansion_factor,
                self.n_heads
            ) for i in range(self.num_layers)
        ]

    def __call__(self, x, train=True):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out, train)

        return out  # 32x10x512

    def hidden_states(self, x, train=True):
        states = {}
        extend_states(states, 'embedding_layer',
                      self.embedding_layer.hidden_states(x)
                      )
        extend_states(states, 'positional_encoder',
                      self.positional_encoder.hidden_states(states['embedding_layer'])
                      )
        out = states['positional_encoder']
        for i, layer in enumerate(self.layers):
            out = extend_states(states, f'layer.{i}',
                                layer.hidden_states(out, out, out, train))

        states[''] = out
        return states


class DecoderBlock(nn.Module):
    embed_dim: int
    expansion_factor: int=4
    n_heads: int=8
    residue_link: bool=False
    def setup(self):
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(self.embed_dim, n_heads=self.n_heads)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_block = TransformerBlock(self.embed_dim, self.expansion_factor, self.n_heads)

    def __call__(self, key, x, value, mask, train=True):
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
        attention = self.attention(x, x, x, mask=mask)  # 32x10x512
        if self.residue_link:
            attention = attention + x
        query = self.dropout(self.norm(attention), deterministic=not train)

        out = self.transformer_block(key, query, value)

        return out

    def hidden_states(self, key, x, value, mask, train=True):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """

        states = {}
        # we need to pass mask mask only to fst attention
        attention = extend_states(states, 'attention',
                                  self.attention.hidden_states(x, x, x, mask=mask))  # 32x10x512
        if self.residue_link:
            attention = states['with_residue'] = attention + x
        norm = states['norm'] = self.norm(attention)
        query = states['dropout'] = self.dropout(norm, deterministic=not train)

        out = extend_states(states, 'transformer_block',
                            self.transformer_block.hidden_states(key, query, value))

        states[''] = out
        return states


class TransformerDecoder(nn.Module):
    vocab_size: int
    embed_dim: int
    seq_len: int
    num_layers: int = 2
    expansion_factor: int = 4
    n_heads: int = 8
    word_embedding: Optional[jnp.ndarray]=None
    residue_links: bool=False
    def setup(self):
        """
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        print(self.word_embedding)
        self.embedding_layer = Embedding(self.vocab_size, self.embed_dim, self.word_embedding)
        # self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(self.seq_len, self.embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = [
            DecoderBlock(self.embed_dim, expansion_factor=self.expansion_factor,
                         n_heads=self.n_heads, residue_link=self.residue_links)
            for _ in range(self.num_layers)
        ]
        self.fc_out = nn.Dense(self.vocab_size)

    def __call__(self, x, enc_out, mask, train=True):
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        print(x.shape)
        x = self.embedding_layer(x)  # 32x10x512
        # x = self.word_embedding(x)  # 32x10x512
        print(x.shape)
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x, deterministic=not train)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask, train)

        out = nn.softmax(self.fc_out(x), axis=-1)

        return out

    def hidden_states(self, x, enc_out, mask, train=True):
        states = {}
        x = extend_states(states, 'embedding_layer',
                          self.embedding_layer.hidden_states(x))  # 32x10x512

        # x = self.word_embedding(x)  # 32x10x512
        x = states['position_embedding'] = self.position_embedding(x)  # 32x10x512
        x = states['dropout'] = self.dropout(x, deterministic=not train)

        for i, layer in enumerate(self.layers):
            x = extend_states(states, f'layer.{i}', layer.hidden_states(enc_out, x, enc_out, mask))

        x = states['fc_out'] = self.fc_out(x)
        out = states['softmax'] = nn.softmax(x, axis=-1)

        states[''] = out
        return states


class Transformer(nn.Module):
    embed_dim: int
    src_vocab_size: int
    target_vocab_size: int
    source_seq_length: int
    target_seq_length: int
    num_layers: int=2
    expansion_factor: int=4
    n_heads: int=8
    target_mask_fn: Optional[callable]=None
    source_embedding: Optional[jnp.ndarray]=None
    target_embedding: Optional[jnp.ndarray]=None
    decoder_residue_links: bool=True
    def __post_init__(self):
        super().__post_init__()
        if self.target_mask_fn is None:
            self.target_mask_fn = self.make_trg_mask
    def setup(self):
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

        self.encoder = TransformerEncoder(self.source_seq_length, self.src_vocab_size, self.embed_dim, num_layers=self.num_layers,
                                          expansion_factor=self.expansion_factor, n_heads=self.n_heads,
                                          word_embedding=self.source_embedding)
        self.decoder = TransformerDecoder(self.target_vocab_size, self.embed_dim, self.target_seq_length, num_layers=self.num_layers,
                                          expansion_factor=self.expansion_factor, n_heads=self.n_heads,
                                          word_embedding=self.target_embedding, residue_links=self.decoder_residue_links)



    def encode(self, src):
        return self.encoder(src)

    def decode(self, trg, mem):
        mask = self.make_trg_mask(trg)
        return self.decoder(trg, mem, mask)

    @staticmethod
    def make_trg_mask(trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = jnp.tril(jnp.ones((trg_len, trg_len))).reshape(
            1, 1, trg_len, trg_len
        )
        # returns the upper triangular part of matrix filled with ones
        # trg_mask = torch.triu(torch.ones((trg_len, trg_len))).expand(
        #     1, 1, trg_len, trg_len
        # )
        return trg_mask

    def __call__(self, src, trg, train=True):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        batch_size, trg_len = trg.shape
        trg_mask = self.target_mask_fn(trg).reshape(1, 1, trg_len, trg_len)
        enc_out = self.encoder(src, train)

        outputs = self.decoder(trg, enc_out, trg_mask, train)
        return outputs

    def hidden_states(self, src, trg, train=True):
        batch_size, trg_len = trg.shape
        trg_mask = self.target_mask_fn(trg).expand(1, 1, trg_len, trg_len)
        states = {}
        extend_states(states, 'encoder', self.encoder.hidden_states(src, train))

        extend_states(states, 'decoder',
                      self.decoder.hidden_states(trg, states['encoder'], trg_mask, train))
        states[''] = states['decoder']
        return states


def extend_states(states, mod_name, mod_states):
    for k, state in mod_states.items():
        if k:
            states[f'{mod_name}.{k}'] = state
        else:
            states[f'{mod_name}'] = state
    return mod_states['']



