#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

"""
Reimplement Andrej karpathy's gpt.py from scratch.


Usage:
python gpt_b.py input.txt
"""

from typing import Optional

import fire
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """Single head of self-attention."""

    def __init__(self, emb_dim: int, head_dim: int, max_position_embeddings: int, dropout_rate: float):
        super().__init__()

        # w/o bias
        self.key = nn.Linear(emb_dim, head_dim, bias=False)
        self.query = nn.Linear(emb_dim, head_dim, bias=False)
        self.value = nn.Linear(emb_dim, head_dim, bias=False)

        # todo: should be more flexible
        self.register_buffer("tril", torch.tril(torch.ones(max_position_embeddings, max_position_embeddings)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        # input size: B x T x emb_dim

        B, T, C = x.shape

        # k/q/v projection
        # B x T x head_dim
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # compute attention scores ("affinities")
        # (B x T x head_dim) @ (B x head_dim x T) ->  B x T x T
        weight = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # DON'T FORGET "DIM"
        weight = F.softmax(weight, dim=-1)
        # dropout randomly prevent some of the nodes communicating
        weight = self.dropout(weight)

        # perform the weighted aggregation of the values
        # (B x T x T) @ (B x T x head_dim) -> (B x T x head_dim)
        out = weight @ value

        return out


# use convolution group instead of a single large convolution
# helps create multiple indepedent channel communicaiton
# gathering different types of information
class MultiAttentionHead(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, emb_dim: int, num_heads: int, max_position_embeddings: int, dropout_rate: float):
        super().__init__()

        head_dim = emb_dim // num_heads
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(
                    emb_dim=emb_dim,
                    head_dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_heads)
            ]
        )

        # projection layer
        self.proj = nn.Linear(head_dim * num_heads, emb_dim)

        # dropout right before get residual connection back to the original pathway
        # prevent overfitting when scaling up the model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        # B x T x emb_dim
        # DON'T FORGET "DIM"
        out = torch.cat([attention_head(x) for attention_head in self.attention_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# think on what they've founnd on the other tokens
# self-attention is a communication, gathering information
# mlp is on per-token level, all tokens thinking on itself individually
class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, emb_dim: int, dropout_rate: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),  # NB
            # projection layer
            nn.Linear(emb_dim * 4, emb_dim),
            # dropout right before get residual connection back to the original pathway
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, emb_dim: int, num_heads: int, max_position_embeddings: int, dropout_rate: float):
        super().__init__()

        self.attn = MultiAttentionHead(
            emb_dim=emb_dim, num_heads=num_heads, max_position_embeddings=max_position_embeddings, dropout_rate=dropout_rate
        )
        self.ffwd = FeedForward(emb_dim=emb_dim, dropout_rate=dropout_rate)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        # help with the depth: residual, layernorm (pre-LN)

        # residual
        # fork off, do some communication (attention), and come back
        # with the residual, one can go directly from supervision (loss) all the way to the input
        # attention/mlp blocks contribute very little in the beginning (as if they are not there)
        # but come online over time during optimization and start to contribute
        # this dramatically help with the optimization

        x = self.attn(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x


class GPTLanguageModel(nn.Module):
    """GPT Language model for casual LM."""

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        vocab_size: int,
        max_position_embeddings: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(max_position_embeddings, emb_dim)

        self.blocks = nn.Sequential(
            *[
                Block(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    max_position_embeddings=max_position_embeddings,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        # final layer norm
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape

        # B x T x emb_dim
        token_embedding = self.token_embedding_table(input_ids)
        # T x emb_dim
        position_embedding = self.position_embedding_table(torch.arange(T, device=input_ids.device))
        # B x T x emb_dim
        x = token_embedding + position_embedding

        x = self.blocks(x)

        # additional LN before final prediction
        x = self.ln_f(x)
        # B x T x vocab_size
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.modules.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return dict(
            logits=logits,
            loss=loss,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        # input_ids is B x T
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            input_ids_cond = input_ids[:, -self.max_position_embeddings :]
            # forward
            # B x T x C
            logits = self(input_ids_cond)["logits"]
            # focus only on the last time step
            # B x C
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            # B x C
            scores = F.softmax(logits, dim=-1)
            # sample from the distribution
            # B x 1
            next_token_ids = torch.multinomial(scores, num_samples=1)
            # append sampled index to the running sequence
            # B x (T+1)
            input_ids = torch.cat((input_ids, next_token_ids), dim=1)

        return input_ids


def main(
    input_file: str,
    # hyperparameters
    batch_size=64,  # how many independent sequences will we process in parallel?
    block_size=256,  # what is the maximum context length for predictions?
    max_iters=5000,
    eval_interval=500,
    # NB: bring down LR when model gets bigger
    learning_rate=3e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_iters=200,
    emb_dim=384,
    num_heads=6,
    num_layers=6,
    dropout_rate=0.2,
):
    torch.manual_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(input_file) as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabs: {chars}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    model = GPTLanguageModel(
        num_layers=num_layers,
        emb_dim=emb_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                model_outputs = model(X, Y)
                loss = model_outputs["loss"]
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        model_outputs = model(xb, yb)
        loss = model_outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training is done!")

    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Generation:")
    print(decode(model.generate(decoder_input_ids, max_new_tokens=512)[0].tolist()))


if __name__ == "__main__":
    fire.Fire(main)
