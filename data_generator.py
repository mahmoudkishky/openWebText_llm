import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.utils import Sequence
from datasets import load_dataset, Dataset, concatenate_datasets
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import glob
from tqdm import tqdm
import configs

# use parallism for faster tokenization
def memmap_append(path, new_data):
    new_data = np.asarray(new_data, dtype=np.int32)

    if not os.path.exists(path):
        fp = np.memmap(path, dtype=np.int32, mode="w+", shape=new_data.shape)
        fp[:] = new_data
        fp.flush()
        return

    itemsize = 4
    old_bytes = os.path.getsize(path)
    old_items = old_bytes // itemsize

    new_items = new_data.shape[0]
    new_bytes = (old_items + new_items) * itemsize

    # Resize file
    with open(path, "r+b") as f:
        f.seek(new_bytes - 1)
        f.write(b"\0")

    fp = np.memmap(path, dtype=np.int32, mode="r+",
                   shape=(old_items + new_items,))
    fp[old_items:] = new_data
    fp.flush()


def load_openwebtext():

    shards = sorted(glob.glob(os.path.join(os.path.expanduser(configs.OPENWEBTEXT_CACHE_DIR), "**", "openwebtext-train-*.arrow"), recursive=True))

    if shards:
        # from_file memory-maps the shards in place (no copy, unlike the "arrow" builder)
        return {"train": concatenate_datasets([Dataset.from_file(f) for f in shards])}

    print("No local OpenWebText cache found, downloading parquet conversion from the Hub")
    return load_dataset("Skylion007/openwebtext", revision="refs/convert/parquet")


def loadOpenWebTextDataset():

    ds = load_openwebtext()

    if not os.path.exists("bpe_tokenizer.json"):

        print("Train BPE tokenizer on OpenWebText dataset")

        web_txt = (item["text"] for item in ds["train"])
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        bpe_trainer = trainers.BpeTrainer(vocab_size=configs.VOCAB_SIZE, special_tokens=["[UNK]", "[BOS]", "[PAD]", "[EOS]"])
        tokenizer.train_from_iterator(web_txt, trainer=bpe_trainer)
        tokenizer.save("bpe_tokenizer.json")

        print("Tokenizer trained and saved to bpe_tokenizer.json")

    else:

        print("Load existing BPE tokenizer from bpe_tokenizer.json")
        tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
    
    # contiguous split over the first MAX_DOCS documents (random access over millions of docs is slow)
    num_docs = min(configs.MAX_DOCS, len(ds['train'])) if configs.MAX_DOCS else len(ds['train'])
    num_val = max(1, int(num_docs * configs.VAL_SET_RATIO))
    train_doc_ids = range(0, num_docs - num_val)
    test_doc_ids = range(num_docs - num_val, num_docs)

    return ds, tokenizer, train_doc_ids, test_doc_ids


class OpenWebTextDataGenerator(Sequence):

    def __init__(self, ds, tokenizer, doc_ids, isTraining, cache_prefix="openwebtext"):

        self.batch_size = configs.MICRO_BATCH_SIZE if isTraining else configs.VAL_BATCH_SIZE
        self.max_seq_length = configs.MAX_SEQ_LENGTH
        self.total_training_steps = int(configs.TOTAL_TRAINING_STEPS) if isTraining else int(configs.TOTAL_TRAINING_STEPS * configs.VAL_SET_RATIO)
        self.tokenizer = tokenizer
        self.isTraining = isTraining

        os.makedirs("cache", exist_ok=True)
        tokens_file = f"cache/{cache_prefix}_train_tokens.bin" if isTraining else f"cache/{cache_prefix}_val_tokens.bin"

        # create memmap file for train/val tokens if not exists
        if not os.path.exists(tokens_file):

            self.__build_token_file__(ds["train"].select(doc_ids), tokens_file)

        self.docs_tokens = np.memmap(tokens_file, dtype=np.int32, mode="r")

    def __build_token_file__(self, docs, tokens_file, chunk=2000):

        eos = self.tokenizer.token_to_id("[EOS]")
        tmp_file = tokens_file + ".tmp"

        # encode_batch tokenizes in parallel in Rust; write to a tmp file so an interrupted run leaves no partial cache
        with open(tmp_file, "wb") as f:
            for start in tqdm(range(0, len(docs), chunk), desc=f"Tokenizing {os.path.basename(tokens_file)}"):
                texts = docs[start:start + chunk]["text"]
                ids = [enc.ids + [eos] for enc in self.tokenizer.encode_batch(texts)]
                np.concatenate(ids).astype(np.int32).tofile(f)

        os.replace(tmp_file, tokens_file)

    def __len__(self):
        return self.total_training_steps

    def __getitem__(self, index):

        X = np.empty((self.batch_size, self.max_seq_length), dtype=np.int32)
        Y = np.empty((self.batch_size, self.max_seq_length), dtype=np.int32)

        for i in range(self.batch_size):

            if self.isTraining:
                # random offset: chunks overlap naturally
                off = random.randint(0, len(self.docs_tokens) - self.max_seq_length - 1)
            else:
                off = ((index * self.batch_size + i) * self.max_seq_length) % (len(self.docs_tokens) - self.max_seq_length - 1)

            X[i] = self.docs_tokens[off : off + self.max_seq_length]
            # Y is X shifted by 1
            Y[i] = self.docs_tokens[off + 1 : off + self.max_seq_length + 1]

        #print("Generated batch X:", self.tokenizer.decode(X[0][0:10].tolist()))
        #print("Generated batch Y:", self.tokenizer.decode(Y[0][0:10].tolist()))
        #print("#########################################################")
        return {"X": X, "Y": Y}


    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
