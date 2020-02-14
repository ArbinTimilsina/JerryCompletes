import csv
import os

import torch
from torch.utils.data import Dataset, random_split

THIS_DIR = os.path.dirname(__file__)
EOT = "<|endoftext|>"


def get_dataset(tokenizer, block_size=512):
    training_file_path = get_training_file()
    dataset = TextDataset(
        tokenizer=tokenizer, file_path=training_file_path, block_size=block_size
    )

    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset


def get_training_file():
    """
    Get all the dialogues by Jerry in the CSV file and write it to a txt file.
    """
    seinfeld_scripts = os.path.join(
        THIS_DIR, '..', 'seinfeld_scripts', 'complete_seinfeld_scripts.csv'
    )
    training_file = os.path.join(os.path.dirname(seinfeld_scripts), 'training.txt')

    if not os.path.isfile(training_file):
        with open(seinfeld_scripts) as input_file:
            with open(training_file, 'w') as out_file:
                input_data = csv.DictReader(input_file)
                for row in input_data:
                    if row['Character'] == 'JERRY':
                        out_file.write(row['Dialogue'] + EOT)
    return training_file


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i: i + block_size]
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
