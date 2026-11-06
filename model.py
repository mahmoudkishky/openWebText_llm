import configs
import tensorflow as tf
from tensorflow.keras import Model
from keras import initializers
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, Dropout
from tensorflow.nn import softmax


def is_legacy_checkpoint(path):
    import h5py

    with h5py.File(path, "r") as checkpoint_file:
        has_final_norm = "final_norm" in checkpoint_file
        if not has_final_norm and "layers" in checkpoint_file:
            has_final_norm = "final_norm" in checkpoint_file["layers"]
    return not has_final_norm


def load_checkpoint_weights(model, path):
    if is_legacy_checkpoint(path):
        model.load_weights(path, by_name=True, skip_mismatch=True)
    else:
        model.load_weights(path)


class SelfAttention(Model):

    def __init__(self, d_model):

        super().__init__()

        self.wq = Dense(d_model, use_bias=False)
        self.wk = Dense(d_model, use_bias=False)
        self.wv = Dense(d_model, use_bias=False)

    def call(self, x):

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scale = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        att_score = tf.matmul(q, k, transpose_b=True) / scale

        causal_mask = tf.linalg.band_part(tf.ones((tf.shape(x)[1], tf.shape(x)[1]), dtype=tf.float32), -1, 0)
        att_score = att_score + 1e9 * (causal_mask - 1)

        att_weights = softmax(att_score, axis=-1)
        output = tf.matmul(att_weights, v)

        return output
    

def stateless_dropout(x, rate, seed):
    if seed is None or rate <= 0.0:
        return x
    return tf.nn.experimental.stateless_dropout(x, rate=rate, seed=seed)


class DenseWithLoRA(Dense):

    def __init__(self, units, rank, alpha=None, **kwargs):
        super().__init__(units, **kwargs)
        self.rank = rank
        self.scale = (alpha or rank) / rank

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel.trainable = False
        if self.bias is not None:
            self.bias.trainable = False
        self.lora_a = self.add_weight(name="lora_a", shape=(input_shape[-1], self.rank), initializer="he_uniform")
        self.lora_b = self.add_weight(name="lora_b", shape=(self.rank, self.units), initializer="zeros")

    def call(self, inputs):
        # y = activation(xW + b + scale * xAB)
        output = tf.matmul(inputs, self.kernel) + self.scale * tf.matmul(tf.matmul(inputs, self.lora_a), self.lora_b)
        if self.bias is not None:
            output += self.bias
        return self.activation(output)


class MultiHeadAttention(Model):

    def __init__(self, d_model, num_heads, lora_rank=0, lora_alpha=None):

        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        def dense():
            return DenseWithLoRA(d_model, lora_rank, lora_alpha, use_bias=False) if lora_rank else Dense(d_model, use_bias=False)

        self.wq = dense()
        self.wk = dense()
        self.wv = dense()
        self.wo = dense()

    def split_heads(self, x):
        # x: (B, T, d_model)
        B, T, _ = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (B, T, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, D)

    def call(self, x, seed=None):
        B, T, _ = tf.unstack(tf.shape(x))

        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        att = tf.matmul(q, k, transpose_b=True) / scale  # (B, H, T, T)

        causal_mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        causal_mask = tf.cast(causal_mask, att.dtype)
        att = att + (1e9 * (causal_mask - 1))

        att_weights = tf.nn.softmax(att, axis=-1)
        att_weights = stateless_dropout(att_weights, configs.ATTENTION_DROPOUT, seed)

        out = tf.matmul(att_weights, v)  # (B, H, T, D)

        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (B, T, self.num_heads * self.head_dim))

        return self.wo(out)
    

class PositionalEncoding(Model):
    
    def __init__(self, max_seq_length, model_dim):

        super().__init__()

        pos = tf.range(max_seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(model_dim, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(model_dim, tf.float32))
        angle_rads = pos * angle_rates

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]

    def call(self, x):

        seq_len = tf.shape(x)[1]
        pe  = self.pos_encoding[:, :seq_len, :]
        return x + tf.cast(pe, x.dtype)


class TransformerDecoder(Model):
    
    def __init__(self, d_model, num_attention_heads, ffn_dim, lora_rank=0, lora_alpha=None):

        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_attention_heads, lora_rank, lora_alpha)
        def dense(units, **kwargs):
            return DenseWithLoRA(units, lora_rank, lora_alpha, **kwargs) if lora_rank else Dense(units, **kwargs)

        self.ffn = tf.keras.Sequential([dense(ffn_dim, activation='gelu'),
                                        dense(d_model),
                                        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, seed=None):
        s_att, s_res1, s_res2 = (None, None, None) if seed is None else (seed, seed + 1, seed + 2)
        x = x + stateless_dropout(self.attention(self.norm1(x), seed=s_att), configs.RESIDUAL_DROPOUT, s_res1)
        x = x + stateless_dropout(self.ffn(self.norm2(x)), configs.RESIDUAL_DROPOUT, s_res2)
        return x
    
class LLM(Model):
    
    def __init__(self, tokenzier_vocab_size, model_dim, num_attention_heads, num_transformer_layers = 4, use_tying_embedding=True, lora_rank=0, lora_alpha=None):

        super().__init__()
        self.embedding = Embedding(tokenzier_vocab_size, model_dim, embeddings_initializer = initializers.RandomNormal(stddev=0.02))
        self.model_dim = model_dim
        self.positional_encoding = PositionalEncoding(configs.MAX_SEQ_LENGTH, model_dim)
        self.transformers = [TransformerDecoder(model_dim, num_attention_heads, configs.FFN_DIM, lora_rank, lora_alpha) for _ in range(num_transformer_layers)]
        self.use_tying_embedding = use_tying_embedding
        self.final_norm = LayerNormalization(epsilon=1e-6)

        if self.use_tying_embedding:
            self.lm_bias = self.add_weight(name="lm_bias", shape=(tokenzier_vocab_size,), initializer="zeros", trainable=True)
        else:
            self.head = Dense(tokenzier_vocab_size)

    @property
    def lora_variables(self):
        variables = []
        for block in self.transformers:
            for layer in [block.attention.wq, block.attention.wk, block.attention.wv, block.attention.wo, *block.ffn.layers]:
                if isinstance(layer, DenseWithLoRA):
                    variables += [layer.lora_a, layer.lora_b]
        return variables


    def call(self, input, training=False):

        seeds = tf.random.uniform((len(self.transformers) + 1, 2), maxval=2**30, dtype=tf.int32) if training else None

        # scale embeddings so they are not dominated by the sinusoidal encoding
        x = self.embedding(input) * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x = self.positional_encoding(x)
        x = stateless_dropout(x, configs.EMBEDDING_DROPOUT, None if seeds is None else seeds[-1])

        for i, transformer in enumerate(self.transformers):
            # Use gradient checkpointing for memory efficiency by recomputing the activations during backpropagation 
            # instead of storing them all in memory. This allows us to train deeper models without running out of memory.
            # we need stateless dropout inside the transformer, so we pass the seed to reproduce the same mask
            # when we apply recompute_grad
            if seeds is None:
                x = transformer(x)
            else:
                x = tf.recompute_grad(transformer)(x, seeds[i])

        x = self.final_norm(x)

        if self.use_tying_embedding:

            # use tying embedding to avoid very large output layer instead of Dense(tokenzier_vocab_size)
            # the embedding weights project the input to embedding space
            # the output layer projects back to token space
            # so we can reuse the embedding weights for output layer to do the projection (tying embedding) to reduce the number of parameters

            E = self.embedding.embeddings  # shape: (vocab_size, model_dim)
            # flatten to 2D so E is not broadcast over the batch (its gradient would otherwise be a (B, vocab, dim) tensor)
            B, T = tf.shape(x)[0], tf.shape(x)[1]
            x2d = tf.reshape(x, (-1, self.model_dim))
            logits = tf.matmul(x2d, tf.cast(E, x2d.dtype), transpose_b=True) + tf.cast(self.lm_bias, x2d.dtype)
            logits = tf.reshape(logits, (B, T, -1))

        else:

            logits = self.head(x)

        # cast logits to float32 for gradient stability with mixed precision
        return tf.cast(logits, tf.float32)
