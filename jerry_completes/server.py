from transformers import GPT2Tokenizer
import truecase


def complete_this(
        model, device, seed_sequence, max_length=25, temperature=1.0,
        stop_token="<|endoftext|>"
):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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
        do_sample=True,
    )

    # Batch size == 1. to add more examples please use num_return_sequences > 1
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(stop_token) if stop_token else None]

    return truecase.get_true_case(text)
