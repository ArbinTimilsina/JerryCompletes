from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch

from jerry_completes.server import complete_this

app = Flask(__name__)

if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('\nServing on GPU.')
else:
    DEVICE = 'cpu'
    print('\nServing on CPU.')

MODEL = GPT2LMHeadModel.from_pretrained('output')
MODEL.to(DEVICE)
TOKENIZER = GPT2Tokenizer.from_pretrained('output')


@app.route('/')
def get_seed():
    return render_template('for_seed.html')


@app.route('/completed', methods=['POST'])
def make_completions():
    if request.method == 'POST':
        seed_sequence = request.form['seed']
        completion = complete_this(MODEL, TOKENIZER, DEVICE, seed_sequence)

        return render_template(
            'render_this.html', seed=seed_sequence, jerry_completes=completion
            )


def _main(host="localhost", port=5050):
    app.run(host=host, port=port, debug=False)


def main():
    """
    The setuptools entry point.
    """
    _main()
