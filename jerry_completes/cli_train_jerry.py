import argparse
import time
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange
from transformers import (
    AdamW, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
)

from jerry_completes.data_reader import get_dataset
from jerry_completes.trainer import train_epoch, valid_epoch


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-epochs', '--epochs', type=int, default=1,
        help='Choose the number of epochs for training.'
    )
    ap.add_argument(
        '-batch_size', '--batch_size', type=int, default=1,
        help='Batch size.'
    )
    ap.add_argument(
        '-block_size', '--block_size', type=int, default=512,
        help='Sequence block size.'
    )
    ap.add_argument(
        '-learning_rate', '--learning_rate', type=int, default=5e-5,
        help='Learning rate.'
    )

    return vars(ap.parse_args())


def collate(examples):
    """
    Pad a list of variable length Tensors with '0'. Output will be B x T x *, where `B`
    is batch size and `T` is length of the longest sequence.
    """
    return pad_sequence(examples, batch_first=True)


def fine_tune_gpt2():
    args = argument_parser()

    num_epochs = int(args['epochs'])
    batch_size = int(args['batch_size'])
    block_size = int(args['block_size'])
    learning_rate = float(args['learning_rate'])

    print()
    if torch.cuda.is_available():
        device = 'cuda'
        print('Training on GPU.')
    else:
        device = 'cpu'
        print('Training on CPU.')

    # Get dataset
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_dataset, valid_dataset = get_dataset(tokenizer, block_size=block_size)

    # Sample data randomly from a shuffled dataset
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)

    # Combine a dataset and a sampler, and provide an iterable over the given dataset.
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate
    )
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=batch_size, collate_fn=collate
    )
    print(f'Batch size is {batch_size}.')
    print(
        f'Training on {len(train_dataloader)} sequences and '
        f'validating on {len(valid_dataloader)} sequences.\n'
    )

    # Get the pre-trained GPT2 Model transformer with a language modeling head on top
    # (linear layer with weights tied to the input embeddings).
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    # Adam algorithm with weight decay fix
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    # Schedule learning rate to decrease linearly after linearly increasing during a
    # warmup period
    optimization_steps = len(train_dataloader) // num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=optimization_steps
    )

    save_dir = 'trained_model'
    if not os.path.isdir(f'{save_dir}'):
        os.makedirs(f'{save_dir}')

    best_valid_loss = float('inf')
    start_time = time.time()
    for epoch in trange(num_epochs, desc="Epoch"):
        train_loss, train_lr = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        valid_loss = valid_epoch(model, valid_dataloader, device)

        if valid_loss < best_valid_loss:
            print(
                f'Valid loss {valid_loss:.3f} is less than best valid loss '
                f'{best_valid_loss:.3f}!'
            )
            best_valid_loss = valid_loss

            print(f'Saving model to {save_dir:s}')
            model.save_pretrained(save_dir)
        else:
            print('Valid loss ({valid_loss:.3f}) did not improve!')

        elapsed_time = time.time() - start_time
        print(
            f'Epoch: {epoch + 1} (of {num_epochs}); '
            f'elapsed time: {elapsed_time / 60:.1f}m')
        print(f'Train loss: {train_loss:.3f}; Valid loss: {valid_loss:.3f}')
        print(f'Train lr: {train_lr:.3E}\n')


def main():
    """
    The setuptools entry point.
    """
    fine_tune_gpt2()