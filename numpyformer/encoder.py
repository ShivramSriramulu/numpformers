import numpy as np
from .utils import softmax, layer_norm

class NumpyFormerEncoder:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_head)
        return np.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x):
        B, H, T, Dh = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(B, T, H * Dh)

    def attention(self, Q, K, V):
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(self.d_head)
        attn = softmax(scores, axis=-1)
        output = np.matmul(attn, V)
        return output, attn

    def forward(self, x):
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_out, attn_weights = self.attention(Q, K, V)
        attn_out = self.combine_heads(attn_out)
        attn_out = np.matmul(attn_out, self.W_o)

        x = layer_norm(x + attn_out)
        return x, attn_weights 