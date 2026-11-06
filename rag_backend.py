"""Local, persistent text retrieval and prompt construction for chat."""

import hashlib
import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
import httpx
import numpy as np
from chromadb.errors import ChromaError
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CHUNK_SIZE = 200
CHUNK_OVERLAP = 25
MAX_FILE_BYTES = 5 * 1024 * 1024
BATCH_SIZE = 128
logger = logging.getLogger(__name__)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


class KnowledgeBase:
    def __init__(self, path, embedding_function=None):
        self.path = str(path)
        with sqlite3.connect(":memory:") as connection:
            try:
                connection.execute("CREATE VIRTUAL TABLE fts_check USING fts5(text)")
            except sqlite3.OperationalError as error:
                if "no such module: fts5" not in str(error):
                    raise
                raise RuntimeError(
                    "SQLite does not support FTS5. Restart Streamlit using the project "
                    "environment and ensure SQLite is imported before TensorFlow. "
                    "See the README for troubleshooting."
                ) from error
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            name="uploaded_text_minilm_v1",
            embedding_function=embedding_function or ONNXMiniLM_L6_V2(),
            metadata={"hnsw:space": "cosine"},
        )

    def add_file(self, name, data):
        source = Path(name).name
        if not source or not source.lower().endswith(".txt"):
            raise ValueError("Only .txt files can be added")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("Files must be 5 MB or smaller")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("Text files must be UTF-8 encoded") from error
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("The file contains no text")
        digest = hashlib.sha256(data).hexdigest()
        existing = self.collection.get(where={"source": source}, include=["metadatas"])
        if existing["ids"] and all(meta["digest"] == digest for meta in existing["metadatas"]):
            return 0
        source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
        ids = [f"{source_id}-{digest}-{index}" for index in range(len(chunks))]
        metadata = [{"source": source, "digest": digest, "chunk": index + 1}
                    for index in range(len(chunks))]
        try:
            for start in range(0, len(chunks), BATCH_SIZE):
                end = start + BATCH_SIZE
                self.collection.upsert(
                    ids=ids[start:end], documents=chunks[start:end], metadatas=metadata[start:end]
                )
        except (ChromaError, httpx.HTTPError, sqlite3.Error, OSError, ValueError, RuntimeError):
            existing_ids = set(existing["ids"])
            new_ids = [item for item in ids if item not in existing_ids]
            for start in range(0, len(new_ids), BATCH_SIZE):
                try:
                    self.collection.delete(ids=new_ids[start:start + BATCH_SIZE])
                except (ChromaError, httpx.HTTPError, sqlite3.Error, OSError, ValueError, RuntimeError):
                    logger.exception("Could not clean up incomplete upload for %s", source)
            raise
        current_ids = set(ids)
        obsolete = [item for item in existing["ids"] if item not in current_ids]
        for start in range(0, len(obsolete), BATCH_SIZE):
            self.collection.delete(ids=obsolete[start:start + BATCH_SIZE])
        return len(chunks)

    def search(self, query, limit=3):
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if not query.strip():
            return []
        if not self.collection.count():
            return []
        results = self.collection.query(
            query_texts=[query], n_results=limit, include=["documents", "metadatas"]
        )
        return [
            {"source": meta["source"], "chunk": int(meta["chunk"]), "text": text}
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def clear(self):
        deleted = 0
        while True:
            ids = self.collection.get(limit=BATCH_SIZE, include=[])["ids"]
            if not ids:
                return deleted
            self.collection.delete(ids=ids)
            deleted += len(ids)


def adapter_rank(path):
    with np.load(path, allow_pickle=False) as weights:
        return int(weights["lora_0"].shape[1])


def load_adapter_weights(model, path):
    with np.load(path, allow_pickle=False) as weights:
        if len(weights.files) != len(model.lora_variables):
            raise ValueError(f"{path} has {len(weights.files)} tensors; expected {len(model.lora_variables)}")
        for index, variable in enumerate(model.lora_variables):
            value = weights[f"lora_{index}"]
            if value.shape != tuple(variable.shape):
                raise ValueError(f"lora_{index} has shape {value.shape}; expected {tuple(variable.shape)}")
            variable.assign(value)


def build_rag_prompt(tokenizer, question, documents, history, max_tokens):
    """Encode the prompt in the same sections as encode_coqa_turn."""
    encode = lambda text: tokenizer.encode(text).ids
    passage_header = encode("### Passage:\n")
    conversation_header = encode("\n\n### Conversation:\n")
    question_header = encode("### Question:\n")
    question_tokens = encode(question)
    answer_header = encode("\n### Answer:\n")

    def prompt_ids(passage, conversation):
        return (
            passage_header + encode(passage) + conversation_header
            + (encode(conversation) if conversation else [])
            + question_header + question_tokens + answer_header
        )

    if len(prompt_ids("", "")) > max_tokens:
        raise ValueError("The question is too long to fit in the model's context window")

    passage = ""
    used = []
    for document in documents:
        prefix = passage + ("\n\n" if passage else "")
        text = document["text"]
        if len(prompt_ids(prefix + text, "")) > max_tokens:
            low, high = 0, len(text)
            while low < high:
                mid = (low + high + 1) // 2
                if len(prompt_ids(prefix + text[:mid], "")) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            text = text[:low]
        if text.strip() and len(prompt_ids(prefix + text, "")) <= max_tokens:
            passage = prefix + text
            used.append({**document, "text": text})

    conversation = ""
    for message in reversed(history):
        label = "Question" if message["role"] == "user" else "Answer"
        separator = "\n" if message["role"] == "user" else "\n\n"
        entry = f"### {label}:\n{message['content']}{separator}"
        if len(prompt_ids(passage, entry + conversation)) > max_tokens:
            break
        conversation = entry + conversation
    return prompt_ids(passage, conversation), used


# ---------------------------------------------------------------------------
# Tests: run with `python -m unittest rag_backend`
# ---------------------------------------------------------------------------

class CharacterTokenizer:
    def encode(self, text):
        class Encoding:
            ids = list(text)
        return Encoding()


class FakeEmbeddings(EmbeddingFunction):
    def __init__(self):
        pass

    @staticmethod
    def name():
        return "test_embeddings"

    def get_config(self):
        return {}

    def __call__(self, input):
        return [
            [float("orion" in text.lower()) + 0.01,
             float("apollo" in text.lower() or "lunar" in text.lower()) + 0.01,
             1.0]
            for text in input
        ]


class KnowledgeBaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "chroma")
        self.knowledge = KnowledgeBase(self.path, embedding_function=FakeEmbeddings())

    def test_chunks_overlap_without_redundant_tail(self):
        self.assertEqual(chunk_text("abcdefghijk", 5, 2), ["abcde", "defgh", "ghijk"])
        self.assertEqual(chunk_text("abcde", 5, 2), ["abcde"])
        with self.assertRaises(ValueError):
            chunk_text("text", 5, 5)

    def test_upload_search_persistence_duplicate_and_replacement(self):
        self.assertEqual(self.knowledge.add_file("notes.txt", b"Orion spacecraft mission"), 1)
        self.assertEqual(self.knowledge.add_file("notes.txt", b"Orion spacecraft mission"), 0)
        reopened = KnowledgeBase(self.path, embedding_function=FakeEmbeddings())
        results = reopened.search('What is "Orion" OR spacecraft?')
        self.assertEqual([(result["source"], result["chunk"]) for result in results], [("notes.txt", 1)])
        self.assertEqual(reopened.add_file("notes.txt", b"Apollo lunar landing"), 1)
        self.assertEqual(reopened.search("lunar")[0]["text"], "Apollo lunar landing")
        self.assertEqual(reopened.collection.count(), 1)

    def test_invalid_upload_does_not_replace_existing_file(self):
        self.knowledge.add_file("notes.txt", b"existing content")
        for data in (b"", b"\xff"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.knowledge.add_file("notes.txt", data)
        with self.assertRaises(ValueError):
            self.knowledge.add_file("notes.pdf", b"text")
        with self.assertRaises(ValueError):
            self.knowledge.add_file("notes.txt", b"x" * (5 * 1024 * 1024 + 1))
        self.assertEqual(self.knowledge.search("existing")[0]["text"], "existing content")

    def test_empty_query_and_ordering(self):
        self.knowledge.add_file("first.txt", b"mars rover mars rover")
        self.knowledge.add_file("second.txt", b"mars")
        self.assertEqual(self.knowledge.search(" "), [])
        self.assertEqual(len(self.knowledge.search("mars", limit=1)), 1)
        with self.assertRaises(ValueError):
            self.knowledge.search("mars", limit=0)

    def test_failed_write_rolls_back_replacement(self):
        self.knowledge.add_file("notes.txt", b"original")
        real_upsert = self.knowledge.collection.upsert
        calls = 0

        def fail_second_batch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated write failure")
            return real_upsert(*args, **kwargs)

        with patch(f"{__name__}.BATCH_SIZE", 1), patch.object(
            self.knowledge.collection, "upsert", side_effect=fail_second_batch
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.knowledge.add_file("notes.txt", b"replacement " * 200)
        self.assertEqual(self.knowledge.search("original")[0]["text"], "original")
        self.assertEqual(self.knowledge.collection.count(), 1)

    def test_overlapping_chunks_are_searchable(self):
        self.knowledge.add_file("large.txt", ("a" * 900 + " beacon " + "b" * 900).encode())
        self.assertGreaterEqual(len(self.knowledge.search("beacon", limit=10)), 2)

    def test_reset_removes_only_uploaded_collection_and_allows_reupload(self):
        self.knowledge.add_file("notes.txt", b"Orion spacecraft mission")
        self.knowledge.add_file("other.txt", b"Apollo lunar landing")
        other = self.knowledge.client.get_or_create_collection(
            name="unrelated_collection", embedding_function=FakeEmbeddings()
        )
        other.add(ids=["other"], documents=["Do not delete"])
        with patch(f"{__name__}.BATCH_SIZE", 1):
            self.assertEqual(self.knowledge.clear(), 2)
        self.assertEqual(self.knowledge.clear(), 0)
        self.assertEqual(other.count(), 1)
        reopened = KnowledgeBase(self.path, embedding_function=FakeEmbeddings())
        self.assertEqual(reopened.search("Orion"), [])
        self.assertEqual(reopened.add_file("notes.txt", b"Orion spacecraft mission"), 1)
        self.assertEqual(self.knowledge.search("Orion")[0]["source"], "notes.txt")

    def test_reset_surfaces_failure(self):
        self.knowledge.add_file("notes.txt", b"original")
        with patch.object(self.knowledge.collection, "delete", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.knowledge.clear()
        self.assertEqual(self.knowledge.collection.count(), 1)

    def test_missing_fts5_explains_how_to_resolve(self):
        class NoFTS5:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, query):
                raise sqlite3.OperationalError("no such module: fts5")

        with patch(f"{__name__}.sqlite3.connect", return_value=NoFTS5()):
            with self.assertRaisesRegex(RuntimeError, "Restart Streamlit"):
                KnowledgeBase(self.path, embedding_function=FakeEmbeddings())


class PromptTest(unittest.TestCase):
    def test_prompt_preserves_question_and_limits_context(self):
        documents = [{"source": "one.txt", "chunk": 1, "text": "x" * 2000}]
        ids, used = build_rag_prompt(CharacterTokenizer(), "Where?", documents, [], 300)
        prompt = "".join(ids)
        self.assertLessEqual(len(prompt), 300)
        self.assertTrue(prompt.endswith("### Question:\nWhere?\n### Answer:\n"))
        self.assertEqual(len(used), 1)
        self.assertIn(used[0]["text"], prompt)
        self.assertLess(len(used[0]["text"]), len(documents[0]["text"]))

    def test_history_and_source_order(self):
        documents = [{"source": "one.txt", "chunk": 2, "text": "Moon"}]
        history = [{"role": "user", "content": "Earlier?"}, {"role": "assistant", "content": "Yes."}]
        ids, used = build_rag_prompt(CharacterTokenizer(), "Now?", documents, history, 500)
        prompt = "".join(ids)
        self.assertEqual(used, documents)
        self.assertIn("### Passage:\nMoon", prompt)
        self.assertNotIn("one.txt", prompt)
        self.assertIn("### Question:\nEarlier?\n### Answer:\nYes.\n\n### Question:\nNow?", prompt)

    def test_long_question_raises(self):
        with self.assertRaisesRegex(ValueError, "question is too long"):
            build_rag_prompt(CharacterTokenizer(), "x" * 500, [], [], 100)

    def test_real_tokenizer_matches_training_prompt(self):
        import numpy as np
        from tokenizers import Tokenizer
        from lora import encode_coqa_turn

        tokenizer_path = os.path.join(os.path.dirname(__file__), "bpe_tokenizer.json")
        raw = Tokenizer.from_file(tokenizer_path)
        example = {
            "story": "Marie Curie was born in Warsaw in 1867.",
            "questions": ["Where was Marie Curie born?", "When?"],
            "answers": {"input_text": ["Warsaw", "1867"]},
        }
        documents = [{"source": "Marie.txt", "chunk": 1, "text": example["story"]}]
        for turn, history in (
            (0, []),
            (1, [{"role": "user", "content": example["questions"][0]},
                 {"role": "assistant", "content": example["answers"]["input_text"][0]}]),
        ):
            with self.subTest(turn=turn):
                encoded, mask = encode_coqa_turn(
                    example, turn, raw, 1024, raw.token_to_id("[PAD]"), raw.token_to_id("[EOS]")
                )
                target_start = int(np.flatnonzero(mask)[0]) + 1
                ids, used = build_rag_prompt(raw, example["questions"][turn], documents, history, 924)
                self.assertEqual(ids, encoded[:target_start].tolist())
                self.assertEqual(used, documents)


class StubTokenizer:
    eos_token_id = 4

    def __init__(self):
        self.encoded = []

    def encode(self, text):
        self.encoded.append(text)
        return [1, 2]

    def get_vocab(self):
        return {str(index): index for index in range(5)}


class StubModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, input_ids):
        import numpy as np
        import tensorflow as tf

        self.inputs.append(input_ids.numpy().tolist())
        logits = np.zeros((*input_ids.shape, 5), dtype=np.float32)
        logits[:, -1, 4] = 100
        return tf.constant(logits)


class StreamTokensTest(unittest.TestCase):
    def test_normal_prompt_is_encoded(self):
        from test_model import stream_tokens

        tokenizer = StubTokenizer()
        model = StubModel()
        self.assertEqual(list(stream_tokens(model, tokenizer, "hello", top_k=1, sampling="top_k")), [4])
        self.assertEqual(tokenizer.encoded, ["hello"])
        self.assertEqual(model.inputs, [[[1, 2]]])

    def test_rag_prompt_ids_are_passed_without_reencoding(self):
        from test_model import stream_tokens

        tokenizer = StubTokenizer()
        model = StubModel()
        prompt_ids, sources = build_rag_prompt(
            CharacterOrdTokenizer(), "question",
            [{"source": "doc.txt", "chunk": 1, "text": "passage"}], [], 500,
        )
        self.assertEqual(len(sources), 1)
        tokens = list(stream_tokens(model, tokenizer, "question", top_k=1, sampling="top_k", prompt_ids=prompt_ids))
        self.assertEqual(tokens, [4])
        self.assertEqual(tokenizer.encoded, [])
        self.assertEqual(model.inputs, [[prompt_ids]])


class CharacterOrdTokenizer:
    def encode(self, text):
        class Encoded:
            ids = [ord(character) for character in text]

        return Encoded()


if __name__ == "__main__":
    unittest.main()
