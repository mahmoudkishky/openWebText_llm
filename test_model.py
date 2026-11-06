import tensorflow as tf
import numpy as np
import argparse
from tokenizers import Tokenizer
from model import LLM, load_checkpoint_weights
import configs
import os
from transformers import PreTrainedTokenizerFast


os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # Use CPU for testing

def stream_tokens(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    top_k=50,
    top_p=0.9,
    temperature=1.0,
    sampling="top_p",
    repetition_penalty=1.0,
    prompt_ids=None,
):
    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt)
    input_ids = tf.constant([prompt_ids], dtype=tf.int32)

    vocab_size = len(tokenizer.get_vocab())

    for _ in range(max_new_tokens):

        logits = model(input_ids[:, -configs.MAX_SEQ_LENGTH:])
        next_token_logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            # CTRL-style penalty: shrink positive logits and amplify negative ones for tokens already in the context
            seen = np.unique(input_ids.numpy()[0, -configs.MAX_SEQ_LENGTH:])
            row = next_token_logits.numpy()[0]
            row[seen] = np.where(row[seen] > 0, row[seen] / repetition_penalty, row[seen] * repetition_penalty)
            next_token_logits = tf.constant(row[np.newaxis, :])

        # sampling: "top_k", "top_p" (nucleus) or "both" (top-k first, then top-p)
        use_k = sampling in ("top_k", "both")
        use_p = sampling in ("top_p", "both")

        k = min(top_k, vocab_size) if use_k else vocab_size
        values, indices = tf.math.top_k(next_token_logits / temperature, k=k)  # sorted descending
        probs = tf.nn.softmax(values)[0].numpy()
        indices = indices[0].numpy()

        if use_p and top_p < 1.0:
            # keep the smallest set of tokens (sorted by prob) whose cumulative prob reaches top_p
            keep = np.cumsum(probs) - probs < top_p
            probs, indices = probs[keep], indices[keep]
            probs = probs / probs.sum()

        next_token = np.random.choice(indices, p=probs)

        input_ids = tf.concat([input_ids, [[next_token]]], axis=1)
        yield int(next_token)

        if next_token == tokenizer.eos_token_id:
            break


def generate_text(model, tokenizer, prompt, **kwargs):
    new_ids = list(stream_tokens(model, tokenizer, prompt, **kwargs))
    return tokenizer.decode(tokenizer.encode(prompt) + new_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--sampling", choices=["top_k", "top_p", "both"], default="top_p", help="Which truncation to use when sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Used when --sampling is top_k or both")
    parser.add_argument("--top_p", type=float, default=0.9, help="Used when --sampling is top_p or both")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="1.0 disables; 1.1-1.3 reduces repeated tokens")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file, e.g. checkpoints/step_500.weights.h5")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json", unk_token="[UNK]", bos_token="[BOS]", pad_token="[PAD]", eos_token="[EOS]")
    vocab_size = len(tokenizer.get_vocab())

    model = LLM(
        tokenzier_vocab_size=vocab_size,
        model_dim=configs.EMBEDDING_DIM,
        num_attention_heads=configs.NUM_ATTENTION_HEADS,
        num_transformer_layers=configs.NUM_TRANSFORMER_LAYERS
    )

    # Dummy input to build model
    _ = model(tf.zeros((1, configs.MAX_SEQ_LENGTH), dtype=tf.int32))

    print(f"Loading weights from: {args.checkpoint}")
    load_checkpoint_weights(model, args.checkpoint)

    print("Generating text...")
    output_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        sampling=args.sampling,
        repetition_penalty=args.repetition_penalty
    )

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated Text ===")
    print(output_text)
    print("======================================")


if __name__ == "__main__":
    main()
