#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

"""
Reimplement Andrej karpathy's bigram.py from scratch.

Usage:
python bigram_b.py input.txt
"""

from typing import Optional

import fire
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # input_ids is B x T

        # B x T x C
        logits = self.embedding_layer(input_ids)

        # B x T
        loss = None
        if labels is not None:
            loss_fct = nn.modules.CrossEntropyLoss()
            # expect initial loss to be ~4.17, which is -ln(1/65), 65 is vocab_size
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return dict(
            logits=logits,
            loss=loss,
        )

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512):
        # input_ids is B x T
        for _ in range(max_new_tokens):
            # feed all current sequence, educational purpose
            # B x T x C
            logits = self(input_ids)["logits"]
            # B x C
            next_token_logits = logits[:, -1, :]
            # B x C
            # DON'T FORGET "DIM"
            next_token_scores = F.softmax(next_token_logits, dim=-1)
            # B x 1
            next_tokens = torch.multinomial(next_token_scores, num_samples=1)
            # B x (T + 1)
            input_ids = torch.cat((input_ids, next_tokens), dim=1)

        return input_ids


def main(
    input_file: str,
    batch_size=32,  # how many independent sequences will we process in parallel?
    block_size=16,  # what is the maximum context length for predictions?
    max_iters=3000,
    eval_interval=300,
    learning_rate=1e-2,
    eval_iters=200,
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

    model = BigramLanguageModel(vocab_size=vocab_size)
    model = model.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        # set to eval mode, disabling dropout, batch_norm
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                model_outputs = model(X, Y)
                loss = model_outputs["loss"]
                losses[k] = loss.item()
            out[split] = losses.mean()
        # set back to train mode
        model.train()
        return out

    # create a PyTorch optimizer
    # good lr is roughly 3e-4, but for set higher for smaller models
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
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
    # tolist() automatically moved to the CPU first if necessary
    print(decode(model.generate(decoder_input_ids, max_new_tokens=512)[0].tolist()))


if __name__ == "__main__":
    fire.Fire(main)
