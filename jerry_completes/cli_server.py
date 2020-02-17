from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel

from jerry_completes.server import complete_this

app = Flask(__name__)

DEVICE = 'cpu'
MODEL = GPT2LMHeadModel.from_pretrained('output')
MODEL.to(DEVICE)


@app.route('/')
def get_seed():
    return render_template('for_seed.html')


@app.route('/completed', methods=['POST'])
def make_completions():
    if request.method == 'POST':
        seed_sequence = request.form['seed']
        completion = complete_this(MODEL, DEVICE, seed_sequence)

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
