import argparse
import math

import numpy as np
import tensorflow as tf

import configs
from model import LLM, load_checkpoint_weights


def build_model(tokenizer, checkpoint, rank, alpha):
    args = dict(
        tokenzier_vocab_size=tokenizer.get_vocab_size(),
        model_dim=configs.EMBEDDING_DIM,
        num_attention_heads=configs.NUM_ATTENTION_HEADS,
        num_transformer_layers=configs.NUM_TRANSFORMER_LAYERS,
    )
    sample = tf.zeros((1, 1), dtype=tf.int32)

    base_model = LLM(**args)
    base_model(sample)
    load_checkpoint_weights(base_model, checkpoint)

    lora_model = LLM(**args, lora_rank=rank, lora_alpha=alpha)
    lora_model(sample)

    # Copy the frozen base weights; adapter variables are the only extra ones
    adapter_ids = {id(v) for v in lora_model.lora_variables}
    base_weights = [v for v in lora_model.weights if id(v) not in adapter_ids]
    for source, target in zip(base_model.weights, base_weights):
        target.assign(source)
    return lora_model


def encode_coqa_turn(example, turn_index, tokenizer, max_seq_length, pad_id, eos_id):
    questions = example["questions"]
    answers = example["answers"]["input_text"]
    if len(questions) != len(answers):
        raise ValueError(
            f"CoQA example has {len(questions)} questions but {len(answers)} answers"
        )
    if not 0 <= turn_index < len(questions):
        raise IndexError(f"Invalid CoQA turn index: {turn_index}")

    encode = lambda text: tokenizer.encode(text).ids
    passage_header = encode("### Passage:\n")
    conversation_header = encode("\n\n### Conversation:\n")
    question_header = encode("### Question:\n")
    answer_header = encode("\n### Answer:\n")
    question_body = encode(questions[turn_index])

    answer_tokens = encode(answers[turn_index])
    target_budget = max_seq_length - len(passage_header) - len(conversation_header) - len(question_header) - len(answer_header) - 1
    if target_budget < 1:
        raise ValueError("Sequence length is too short for the CoQA prompt format")
    target_tokens = answer_tokens[:target_budget - 1] + [eos_id]

    question_budget = (
        max_seq_length
        - len(target_tokens)
        - len(passage_header)
        - len(conversation_header)
        - len(question_header)
        - len(answer_header)
    )
    question_tokens = question_header + question_body[:max(0, question_budget)] + answer_header

    history_text = "".join(
        f"### Question:\n{questions[index]}\n"
        f"### Answer:\n{answers[index]}\n\n"
        for index in range(turn_index)
    )
    history_tokens = encode(history_text)
    story_tokens = encode(example["story"])

    context_budget = max_seq_length - len(target_tokens) - len(question_tokens) - len(passage_header) - len(conversation_header)
    story_budget = min(len(story_tokens), (context_budget * 2) // 3)
    history_budget = min(len(history_tokens), context_budget - story_budget)
    remaining = context_budget - story_budget - history_budget
    extra_story = min(len(story_tokens) - story_budget, remaining)
    story_budget += extra_story
    remaining -= extra_story
    history_budget += min(len(history_tokens) - history_budget, remaining)

    prompt_tokens = passage_header + story_tokens[:story_budget] + conversation_header
    if history_budget:
        prompt_tokens += history_tokens[-history_budget:]
    prompt_tokens += question_tokens
    token_ids = prompt_tokens + target_tokens
    if len(token_ids) > max_seq_length:
        raise ValueError("Encoded CoQA turn exceeds the configured sequence length")

    target_start = len(prompt_tokens) - 1
    labels_mask = np.zeros((max_seq_length - 1,), dtype=np.float32)
    labels_mask[target_start:target_start + len(target_tokens)] = 1.0
    token_ids.extend([pad_id] * (max_seq_length - len(token_ids)))
    return np.asarray(token_ids, dtype=np.int32), labels_mask


def main():
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Fine-tune the project TensorFlow model with LoRA adapters.")
    parser.add_argument("--checkpoint", default="checkpoints/step_9500.weights.h5")
    parser.add_argument("--tokenizer", default="bpe_tokenizer.json")
    parser.add_argument("--output", default="coqa-lora.npz")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")

    model = build_model(tokenizer, args.checkpoint, args.rank, args.alpha)
    adapter_variables = model.lora_variables
    dataset = load_dataset("stanfordnlp/coqa", split="train", token=False)
    num_turns = sum(len(example["questions"]) for example in dataset)
    steps_per_epoch = math.ceil(num_turns / args.batch_size)

    def encoded_examples():
        for example in dataset:
            for turn_index in range(len(example["questions"])):
                yield encode_coqa_turn(
                    example,
                    turn_index,
                    tokenizer,
                    configs.MAX_SEQ_LENGTH,
                    pad_id,
                    eos_id,
                )

    training_data = tf.data.Dataset.from_generator(
        encoded_examples,
        output_signature=(
            tf.TensorSpec((configs.MAX_SEQ_LENGTH,), dtype=tf.int32),
            tf.TensorSpec((configs.MAX_SEQ_LENGTH - 1,), dtype=tf.float32),
        ),
    )
    training_data = training_data.shuffle(10_000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    token_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    for epoch in range(args.epochs):
        progress = tqdm(training_data, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for tokens, mask in progress:
            with tf.GradientTape() as tape:
                logits = model(tokens[:, :-1], training=True)
                losses = token_loss(tokens[:, 1:], logits)
                loss = tf.reduce_sum(losses * mask) / tf.reduce_sum(mask)
            gradients = tape.gradient(loss, adapter_variables)
            optimizer.apply_gradients(zip(gradients, adapter_variables))
            progress.set_postfix(loss=f"{float(loss):.4f}")

    np.savez(args.output, **{f"lora_{i}": v.numpy() for i, v in enumerate(adapter_variables)})
    print(f"Saved {len(adapter_variables)} LoRA tensors to {args.output}")


if __name__ == "__main__":
    main()
