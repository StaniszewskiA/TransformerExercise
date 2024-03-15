import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

from tqdm import tqdm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_queue = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_output = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)

        return output

    def split_heads(self, x):
        batch_size, sequence_length, d_model = x.size()

        return x.view(batch_size, sequence_length, self.num_heads,
                      self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, sequence_length, d_k = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length,
                                                   self.d_model)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.w_queue(q))
        K = self.split_heads(self.w_key(k))
        V = self.split_heads(self.w_value(v))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.w_output(self.combine_heads(attention_output))

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length) -> None:
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, encoder_output, encoder_output,
                                                src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, d_model,
                 num_heads, num_layers, d_ff, max_seq_length, dropout) -> None:
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask

        return src_mask, target_mask

    def forward(self, src, target):
        src_mask, target_mask = self.generate_mask(src, target)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask,
                                           target_mask)

        output = self.fc(decoder_output)
        return output


if __name__ == "__main__":
    src_vocab_size = 5000
    target_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size=src_vocab_size,
                              target_vocab_size=target_vocab_size,
                              d_model=d_model,
                              num_heads=num_heads,
                              num_layers=num_layers,
                              d_ff=d_ff,
                              max_seq_length=max_seq_length,
                              dropout=dropout)

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
    target_data = torch.randint(1, target_vocab_size, (64, max_seq_length))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        output = transformer(src_data, target_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, target_vocab_size), target_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
