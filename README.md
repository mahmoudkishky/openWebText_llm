# OpenWebText LLM: foundation model, QA and RAG

This project builds a local question-answering application in three stages:

1. **Foundation model:** train a decoder-only Transformer to predict the next
   token in OpenWebText.
2. **Question answering:** fine-tune low-rank adapters on the CoQA
   conversational question-answering dataset.
3. **RAG:** retrieve relevant passages from uploaded text files and provide them
   to the fine-tuned model when answering a question.


## Setup

Run commands from the project root in a Python environment with TensorFlow,
NumPy, h5py, Hugging Face `datasets`, `tokenizers` and `transformers`, tqdm,
ChromaDB, HTTPX and Streamlit installed. 

## 1. Train the foundation model

The foundation model in `model.py` is a TensorFlow decoder-only Transformer
with causal self-attention. It learns next-token prediction, not conversational
question answering. Current defaults in `configs.py` include 12 Transformer
layers, 12 attention heads, a 768-dimensional embedding and a 1,024-token
context window.

1. Review `configs.py`, including batch sizes, training steps, validation
   split, `MAX_DOCS` and the OpenWebText cache location.
2. Start training:

   ```sh
   python train_model.py
   ```

3. `data_generator.py` loads cached OpenWebText Arrow shards if available;
   otherwise, it downloads the `Skylion007/openwebtext` parquet conversion
   from Hugging Face. It loads `bpe_tokenizer.json` if present or trains and
   saves a byte-level BPE tokenizer if it is absent. It tokenizes the
   training and validation splits into files in `cache/`.
4. `trainer.py` trains with gradient accumulation and periodically writes
   model weights such as `checkpoints/step_9500.weights.h5` and a matching
   optimizer state file.

To resume from a checkpoint:

```sh
python train_model.py --checkpoint checkpoints/step_9500.weights.h5
```

Keep the tokenizer, model configuration and checkpoint together: they must
match. Token caches are reused when present; if the dataset split or tokenizer
changes, rebuild the corresponding cache files before training.

You can test text generation from a foundation checkpoint with:

```sh
python test_model.py --checkpoint checkpoints/step_9500.weights.h5 --prompt "Once upon a time"
```

Replace the example checkpoint with one you actually trained or have available.

## 2. Fine-tune for conversational QA

`lora.py` loads a foundation checkpoint and trains low-rank adapters on the
CoQA training split. Each training example combines a passage, earlier
questions and answers, and the current question. Loss is calculated on the
current answer rather than on the whole prompt. The optimizer updates the LoRA
variables, and the resulting `.npz` file contains adapter tensors, not a
complete standalone model.

For example:

```sh
python lora.py \
  --checkpoint checkpoints/step_9500.weights.h5 \
  --tokenizer bpe_tokenizer.json \
  --output checkpoints/coqa-lora.npz \
  --rank 8 --alpha 16 \
  --epochs 2 --batch-size 4 --learning-rate 0.0002
```

The default rank is 8 and alpha is 16. The adapter contribution is scaled by
`alpha / rank`. The file does **not** record alpha or the foundation checkpoint:
record both and use the same values when loading the adapter. Rank is inferred
from the adapter file. Use the same tokenizer and model configuration as the
foundation checkpoint.

## 3. Build and use RAG

`rag_backend.py` implements the knowledge base and prompt construction.
Uploaded UTF-8 `.txt` files are split into overlapping character chunks
(currently 200 characters with 25 characters of overlap). ChromaDB embeds
those chunks with the local ONNX `all-MiniLM-L6-v2` model and stores them in
`chroma_db/`. Queries are embedded using the same model and relevant chunks are
retrieved by similarity. Document text and queries are embedded locally, not
sent to OpenAI.

To use it:

1. Ensure `bpe_tokenizer.json`, a foundation checkpoint under `checkpoints/`
   named `step_*.weights.h5`, and the corresponding QA adapter are available.
2. Launch the application:

   ```sh
   streamlit run chat_app.py
   ```

3. In the sidebar, select the foundation checkpoint and the adapter trained
   from it. Set **LoRA alpha** to the value used during training.
4. Enable **Use knowledge base**, then upload one or more `.txt` files using
   the **+** button in the chat input. You may upload without a question or
   upload and ask in the same submission. Files must be UTF-8 encoded and no
   larger than 5 MiB each. `rag_docs/` contains example documents.
5. Ask a question. The application retrieves passages, constructs a prompt in
   the CoQA training format, and streams an answer from the selected local
   model. Expand **Retrieved passages** to inspect the sources.

You can adjust the number of passages, generation length, temperature,
sampling and repetition penalty in the sidebar. Passage and optional chat
history are limited to the model's context window, with room reserved for
generation. The foundation model can also be selected without an adapter, but
it is not fine-tuned for CoQA-style answers.

The database is shared by all users of the app, persists across restarts and
chat clearing, and should not be committed. Treat uploads as visible to anyone
who can use the app. Re-uploading an unchanged filename does nothing;
uploading different content with the same filename replaces its passages.
Changing chunk settings does not automatically re-index existing uploads: reset
the knowledge base and upload them again.

To remove uploaded passages, select **Reset knowledge base** in the sidebar
and confirm. This affects every user and cannot be undone. It leaves original
files, other ChromaDB collections and existing chat history unchanged. If a
reset fails, some passages may already have been removed; retry to finish.
Do not delete or move database files while the app is running.

Retrieval can miss relevant passages or return irrelevant ones, and the model
may produce inaccurate or unsupported answers. Inspect the sources and verify
important claims.

The NASA starter project inspired the overlapping chunking, persistent
ChromaDB collection and source-labelled retrieval. Its OpenAI embeddings and
existing NASA collection are not imported because they use a different
embedding model. The previous `knowledge_base.sqlite3` keyword index is not
automatically imported or deleted; re-upload its original `.txt` files.


## Demo


https://github.com/user-attachments/assets/0b922d75-78fd-4ec4-b70f-f8c5776a94f7

