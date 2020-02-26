from ftfy import fix_text
from nltk import sent_tokenize


def complete_this(
        model, tokenizer, device, seed_sequence, max_length=40, temperature=1.0, num_sent=2
):
    encoded_seed_sequence = tokenizer.encode(
        seed_sequence, add_special_tokens=False, return_tensors="pt"
    )
    encoded_seed_sequence = encoded_seed_sequence.to(device)

    output_sequences = model.generate(
        input_ids=encoded_seed_sequence,
        max_length=max_length,
        temperature=temperature,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True
    )

    generated_sequence = output_sequences[0].tolist()
    decoded_generated_sequence = tokenizer.decode(
        generated_sequence, clean_up_tokenization_spaces=True
    )
    decoded_generated_sequence = fix_text(decoded_generated_sequence)
    print(decoded_generated_sequence)
    decoded_generated_sequence = decoded_generated_sequence.replace('\n', ' ')

    sentences = sent_tokenize(decoded_generated_sequence)
    output = ''
    if len(sentences) < num_sent:
        num_sent = len(sentences)
    for i in range(num_sent):
        output = output + ' ' + sentences[i]
    return output
