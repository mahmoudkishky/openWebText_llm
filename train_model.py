import argparse
import configs
import tensorflow as tf
from model import LLM
from trainer import Trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a checkpoint, e.g. checkpoints/step_5000.weights.h5")
    parser.add_argument("--source-layers", type=int, default=None, help="Number of transformer blocks in --checkpoint when it is shallower than the current model; copies them into the first blocks of the deeper model")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    try:
        from data_generator import OpenWebTextDataGenerator, loadOpenWebTextDataset
    except ImportError as error:
        raise ImportError(
            "train_model.py requires additional packages from data_generator.py "
            "(including 'datasets', 'tokenizers', and 'tqdm'). "
            "Install the training dependencies before running."
        ) from error
    ds, tokenizer, train_doc_ids, test_doc_ids = loadOpenWebTextDataset()
    training_generator = OpenWebTextDataGenerator(ds, tokenizer, train_doc_ids, isTraining=True)
    val_generator = OpenWebTextDataGenerator(ds, tokenizer, test_doc_ids, isTraining=False)

    llm = LLM(training_generator.get_vocab_size(), configs.EMBEDDING_DIM, configs.NUM_ATTENTION_HEADS, configs.NUM_TRANSFORMER_LAYERS)
    trainer = Trainer(llm, training_generator, val_generator, checkpoint=args.checkpoint, source_layers=args.source_layers)

    trainer.train()

if __name__ == "__main__":

    main()