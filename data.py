import json
from torch.utils.data import Dataset, DataLoader
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
import os
import argparse
import torch


def load_data(args):
    with open(args.train_data_dir, "r") as f:
        train = json.load(f)
    with open(args.dev_data_dir, "r") as f:
        dev = json.load(f)

    if args.toy:
        train = train[:10]
        dev = train[:10]

    return train, dev


def train_bart_tokenizer():
    if not os.path.isfile("tokenizer.json"):
        with open("train.txt", "r") as f:
            data = f.read()

        data = data.split("\n")
        all_data = []
        for d in data:
            all_data += d.split("=")

        with open("traintokenizer.txt", "w") as f:
            for d in all_data:
                f.write(d)
                f.write("\n")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=40
        )

        tokenizer.train(["traintokenizer.txt"], trainer)

        tokenizer.save("tokenizer.json")

    tokenizer = Tokenizer.from_file("tokenizer.json")

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    return tokenizer


class InputDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.input_data = [d[0] for d in data]
        self.output_data = [d[1] for d in data]
        self.tokenized_input = self.tokenizer.encode_batch(self.input_data)
        self.tokenized_output = self.tokenizer.encode_batch(self.output_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.tokenized_input[index].ids),
            torch.tensor(self.tokenized_input[index].attention_mask),
            torch.tensor(self.tokenized_output[index].ids),
            torch.tensor(self.tokenized_output[index].attention_mask),
        )


def load_dataloaders(args):
    tokenizer = train_bart_tokenizer()
    train, dev = load_data(args)
    num_train_examples = len(train)
    TrainDataset = InputDataset(train, tokenizer)
    DevDataset = InputDataset(dev, tokenizer)

    train_loader = DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DevDataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, dev_loader, num_train_examples, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=10)

    parser.add_argument("--train_data_dir", type=str, default="train.json")
    parser.add_argument("--dev_data_dir", type=str, default="dev.json")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    train_loader, dev_loader, num_train_examples, tokenizer = load_dataloaders(args)

    for batch in train_loader:
        print(batch[0].tolist())
        print(batch[2].tolist())
        print(tokenizer.decode_batch(batch[0].tolist()))
        print(tokenizer.decode_batch(batch[2].tolist()))
        break
