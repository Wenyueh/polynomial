import torch
import torch.nn as nn
import math
import argparse
from transformers import BartForConditionalGeneration, BartConfig
from data import train_bart_tokenizer
import torch


def construct_model(args):
    config = BartConfig.from_pretrained("facebook/bart-base")
    config.vocab_size = args.vocab_size
    config.d_model = args.d_model

    config.decoder_layers = args.layer_number
    config.decoder_attention_heads = 2
    config.decoder_ffn_dim = args.d_model * 4

    config.encoder_layers = args.layer_number
    config.encoder_attention_heads = 2
    config.encoder_ffn_dim = args.d_model * 4

    model = BartForConditionalGeneration(config)

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of parameters in the model is {}".format(number_of_parameters))

    assert number_of_parameters < 5000000

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=40)
    parser.add_argument("--d_model", type=int, default=150)
    parser.add_argument("--layer_number", type=int, default=6)
    args = parser.parse_args()

    """
    embedder = Embedder(100, 10)
    x = embedder(torch.tensor([[1, 2]]))

    positionalencoder = PositionalEncoder(10, 200)
    a = positionalencoder(x)

    print(a)
    """

    model = construct_model(args)

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert number_of_parameters < 5000000

    model.eval()

    tokenizer = train_bart_tokenizer()

    inputs = ["111111", "222+()**2"]

    for one_input in inputs:
        input_ids = torch.tensor([tokenizer.encode(one_input).ids])
        attention_mask = torch.ones_like(input_ids)
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask)[0]
        out = out.tolist()
        print(out)
        o = tokenizer.decode(out)
        print(o)
        o = o.replace(" ", "")
        print(o)
        print(o)
