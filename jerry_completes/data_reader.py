import csv
import os

import torch
import nltk
from ftfy import fix_text
from nltk import sent_tokenize
from torch.utils.data import Dataset, random_split

nltk.download('punkt')

THIS_DIR = os.path.dirname(__file__)


def get_dataset(tokenizer, output_dir, block_size=128):
    training_file_path = get_training_file(output_dir)
    dataset = TextDataset(
        tokenizer=tokenizer, file_path=training_file_path, block_size=block_size
    )

    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset


def get_training_file(output_dir):
    """
    Get all the dialogues by Jerry in the CSV file and write it to a text file.
    """
    # Note: dataset is obtained from https://www.kaggle.com/thec03u5/seinfeld-chronicles
    seinfeld_scripts = os.path.join(
        THIS_DIR, '..', 'seinfeld_scripts', 'complete_seinfeld_scripts.csv'
    )
    training_file = os.path.join(output_dir, 'seinfeld_input.txt')

    min_length = 8
    with open(seinfeld_scripts) as input_file:
        with open(training_file, 'w') as out_file:
            input_data = csv.DictReader(input_file)
            for row in input_data:
                if row['Character'] == 'JERRY':
                    dialogue = row['Dialogue']
                    for sentence in sent_tokenize(dialogue):
                        if len(sentence.encode('utf-8')) > min_length:
                            out_file.write(fix_text(sentence) + '\n')
    return training_file


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as file:
            lines = [line for line in file.read().splitlines()]

        self.examples = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=block_size
        )['input_ids']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
