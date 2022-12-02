import torch
from data import load_dataloaders
from model import construct_model
import argparse
import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import sys
import os


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class Logger(object):
    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += "+"

    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, "a") as logf:
                logf.write(string)
                if newline:
                    logf.write("\n")

            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    num_train_steps = int(
        num_train_examples
        / args.batch_size
        / args.gradient_accumulation_steps
        * args.epochs
    )
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    return optimizer, scheduler


def post_processing(ids, tokenizer):
    ids = ids.tolist()
    outs = tokenizer.decode_batch(ids)
    withoutwhite = []
    for out in outs:
        withoutwhite.append(out.replace(" ", ""))
    return outs


def evaluation(prediction, gold):
    n = 0
    for p, g in zip(prediction, gold):
        # print((p, g))
        if p == g:
            n += 1

    return n


def main(args):
    set_seeds(args)
    logger = Logger(args.model + ".log", True)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(str(args))

    logger.log("loading data")
    train_loader, dev_loader, num_train_examples, tokenizer = load_dataloaders(args)
    logger.log("set up model")
    model = construct_model(args)
    model.load_state_dict(torch.load("polynomial.pt"))
    model.cuda()
    optimizer, scheduler = configure_optimizer(args, model, num_train_examples)

    logger.log("start training")
    step_num = 0
    logging_steps = 0
    total_loss = 0
    logging_loss = 0
    best_EM = 0
    model.zero_grad()
    for e in range(args.epochs):
        """
        model.train()
        logger.log("trainig epoch {}".format(e))
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            output_ids = batch[2].cuda()

            loss = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=output_ids
            ).loss

            total_loss += loss
            logging_loss += loss.item()
            step_num += 1

            if step_num % args.gradient_accumulation_steps == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                scheduler.step()
                total_loss = 0

                model.zero_grad()
                logging_steps += 1
                if logging_steps % args.logging_steps == 0:
                    logger.log(
                        "the accumulated loss for epoch {} from {} steps to {} steps is {}".format(
                            e,
                            logging_steps - args.logging_steps,
                            logging_steps,
                            loss.item(),
                        )
                    )
                    logging_loss = 0
        """
        logger.log("evaluation for epoch {}".format(e))
        model.eval()
        correct = 0
        total = 0
        for batch in dev_loader:
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            output_ids = batch[2]

            predictions = model.generate(
                input_ids=input_ids, attention_mask=attention_mask
            )

            total += output_ids.size(0)
            correct += evaluation(
                post_processing(predictions, tokenizer),
                post_processing(output_ids, tokenizer),
            )
        EM = correct / total
        logger.log("exact match after {} epoch is {}".format(e, EM))

        if EM > best_EM:
            torch.save(model.state_dict(), args.model + ".pt")
            logger.log("exact match {} ----> {} on epoch {}".format(best_EM, EM, e))
            best_EM = EM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed [%(default)d]"
    )
    # model structure
    parser.add_argument("--vocab_size", type=int, default=40)
    parser.add_argument("--d_model", type=int, default=50)
    parser.add_argument("--layer_number", type=int, default=10)

    # data dir
    parser.add_argument("--train_data_dir", type=str, default="train.json")
    parser.add_argument("--dev_data_dir", type=str, default="dev.json")
    parser.add_argument("--batch_size", type=int, default=8)

    # trainning hyperparameters
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)

    # save model
    parser.add_argument("--model", type=str, default="polynomial")
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)
