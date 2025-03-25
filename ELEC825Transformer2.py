from torch import triu, ones, arange, exp, zeros, float32, log, tensor, sin, cos
from torch.nn import Embedding, Linear, Module, MultiheadAttention, LayerNorm, Sequential, ReLU, ModuleList
from torch.nn.init import normal_


class Encoder(Module):
    def __init__(self, d_model, d_ffn, num_heads):
        super().__init__()
        # attention
        self.attention = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attention_norm = LayerNorm(d_model)
        # ffn
        self.ffn = Sequential(
            Linear(d_model, d_ffn),
            ReLU(),
            Linear(d_ffn, d_model))
        self.ffn_norm = LayerNorm(d_model)

    # so: Tensor (batch, length, d_model)
    def forward(self, so, padding_mask):
        # attention
        s_norm = self.attention_norm(so)
        a = self.attention(s_norm, s_norm, s_norm,
                           key_padding_mask=padding_mask)[0]
        a = so + a
        # ffn
        a_norm = self.ffn_norm(a)
        o = self.ffn(a_norm)
        o = a + o
        return o


class Decoder(Module):
    def __init__(self, d_model, d_ffn, num_heads):
        super().__init__()
        # self attention
        self.self_attention = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attention_norm = LayerNorm(d_model)
        # cross attention
        self.cross_attention = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attention_norm = LayerNorm(d_model)
        # ffn
        self.ffn = Sequential(
            Linear(d_model, d_ffn),
            ReLU(),
            Linear(d_ffn, d_model))
        self.ffn_norm = LayerNorm(d_model)

    # so: Tensor (batch, length, d_model)
    # to: Tensor (batch, length, d_model)
    def forward(self, so, to, src_padding_mask, tgt_padding_mask):
        batch, length, d_model = to.shape
        # self attention
        t_norm = self.self_attention_norm(to)
        sa = self.self_attention(t_norm, t_norm, t_norm,
                                 key_padding_mask=tgt_padding_mask,
                                 attn_mask=triu(ones((length, length), device=to.device).bool(), diagonal=1))[0]
        sa = to + sa
        # cross attention
        sa_norm = self.cross_attention_norm(sa)
        ca = self.cross_attention(sa_norm, so, so,
                                  key_padding_mask=src_padding_mask)[0]
        ca = sa + ca
        # ffn
        ca_norm = self.ffn_norm(ca)
        o = self.ffn(ca_norm)
        o = ca + o
        return o


class PositionalEncoding(Module):
    def __init__(self, d_model, max_length) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

    def forward(self, x):
        batch, length, d_model = x.shape
        if not hasattr(self, 'pe'):
            self.register_pe(self.d_model, self.max_length, x.device)
        pe = self.get_buffer('pe')[:length, :].unsqueeze(0).expand(batch, -1, -1)
        return x + pe

    def register_pe(self, d_model, max_length, device):
        pe = zeros((max_length, d_model), device=device)
        position = arange(max_length, dtype=float32, device=device).unsqueeze(1)
        num_terms = (d_model + 1) // 2

        div_term = exp(
            arange(num_terms, dtype=float32, device=device) *
            (-log(tensor(10000.0, device=device)) / d_model) * 2)

        angles = position * div_term

        pe[:, 0::2] = sin(angles[:, :num_terms])

        if d_model % 2 == 0:
            pe[:, 1::2] = cos(angles)
        else:
            pe[:, 1::2] = cos(angles[:, :-1])
        self.register_buffer('pe', pe)


class Transformer(Module):
    def __init__(self, src_size, tgt_size, d_model, d_ffn, num_heads, num_layers, max_length):
        super().__init__()
        self.d_model = d_model
        self.positioning = PositionalEncoding(d_model, max_length)
        # encoder
        self.src_embedding = Embedding(src_size, d_model)
        self.encoders = ModuleList(Encoder(d_model, d_ffn, num_heads) for x in range(num_layers))
        # decoder
        self.tgt_embedding = Embedding(tgt_size, d_model)
        self.decoders = ModuleList(Decoder(d_model, d_ffn, num_heads) for x in range(num_layers))
        # prediction
        self.linear_out = Linear(d_model, tgt_size)

        self._init()

    def _init(self):
        std = 1.0 / (self.d_model ** 0.5)
        # normal_(self.src_positioning.weight, mean=0.0, std=std)
        normal_(self.src_embedding.weight, mean=0.0, std=std)
        # normal_(self.tgt_positioning.weight, mean=0.0, std=std)
        normal_(self.tgt_embedding.weight, mean=0.0, std=std)

    # x: Tensor (batch, length)
    # y: Tensor (batch, length)
    def forward(self, x, y):
        # encoder
        s = self.src_embedding(x) * self.d_model ** 0.5
        s = self.positioning(s)
        so = s
        for i in range(len(self.encoders)):
            so = self.encoders[i](so, x == 0)
        # decoder
        t = self.tgt_embedding(y) * self.d_model ** 0.5
        t = self.positioning(t)
        to = t
        for i in range(len(self.decoders)):
            to = self.decoders[i](so, to, x == 0, y == 0)
        # prediction
        return self.linear_out(to)
