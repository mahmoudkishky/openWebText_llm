import glob
import os
import sqlite3
from pathlib import Path

import streamlit as st
import tensorflow as tf
from chromadb.errors import ChromaError
from httpx import HTTPError
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import configs
from lora import build_model
from model import LLM
from rag_backend import KnowledgeBase, adapter_rank, build_rag_prompt, load_adapter_weights
from test_model import stream_tokens  # also forces CPU so training can keep the GPU

st.set_page_config(page_title="OpenWebText LLM", page_icon="💬")

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "chroma_db"


@st.cache_resource
def load_knowledge_base():
    return KnowledgeBase(KNOWLEDGE_BASE_PATH)


@st.cache_resource
def load_tokenizer():
    return PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json", unk_token="[UNK]", bos_token="[BOS]", pad_token="[PAD]", eos_token="[EOS]")


@st.cache_resource
def load_prompt_tokenizer():
    return Tokenizer.from_file("bpe_tokenizer.json")


@st.cache_resource
def load_model(checkpoint, checkpoint_version, adapter, adapter_version, alpha):
    tokenizer = load_tokenizer()
    if adapter:
        rank = adapter_rank(adapter)
        model = build_model(tokenizer.backend_tokenizer, checkpoint, rank, alpha)
        load_adapter_weights(model, adapter)
    else:
        model = LLM(len(tokenizer.get_vocab()), configs.EMBEDDING_DIM, configs.NUM_ATTENTION_HEADS, configs.NUM_TRANSFORMER_LAYERS)
        model(tf.zeros((1, configs.MAX_SEQ_LENGTH), dtype=tf.int32))
        model.load_weights(checkpoint)
    return model


def file_version(path):
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def checkpoint_step(path):
    name = os.path.basename(path)
    return int(name.split("_")[1].split(".")[0])


with st.sidebar:
    st.header("Settings")

    checkpoints = sorted(glob.glob("checkpoints/step_*.weights.h5"), key=checkpoint_step, reverse=True)
    if st.button("Refresh checkpoints"):
        st.rerun()
    checkpoint = st.selectbox("Checkpoint", checkpoints, help="Newest first")

    adapters = sorted(
        path for path in set(glob.glob("*.npz") + glob.glob("checkpoints/*.npz"))
        if not path.endswith(".optimizer.npz")
    )
    adapter = st.selectbox(
        "LoRA adapter",
        [None] + adapters,
        format_func=lambda path: path if path else "None (foundation model only)",
        help="Choose the adapter trained from the selected foundation checkpoint.",
    )
    alpha = st.number_input(
        "LoRA alpha",
        min_value=0.01,
        value=16.0,
        step=1.0,
        disabled=adapter is None,
        help="Use the alpha supplied during adapter training. The adapter file does not record it.",
    )
    if adapter:
        st.caption("Select the same foundation checkpoint used to train this adapter.")
    else:
        st.warning("No LoRA adapter selected: the foundation model is not fine-tuned to answer CoQA questions.")

    max_new_tokens = st.slider("Max new tokens", 10, 500, 100)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    sampling = st.radio("Sampling", ["top_p", "top_k", "both"], horizontal=True,
                        help="top_p: nucleus sampling. top_k: fixed number of candidates. both: top-k first, then top-p.")
    top_k = st.slider("Top-k", 1, 200, 40, disabled=sampling == "top_p")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05, disabled=sampling == "top_k")
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05, help="1.0 disables. Higher values discourage reusing tokens already in the text.")
    use_history = st.checkbox("Use chat history as context", value=False,
                              help="Prepends previous turns to the prompt. This is a base language model, not instruction-tuned.")
    use_rag = st.checkbox(
        "Use knowledge base", value=False,
        help="Search uploaded text files locally and include matching passages in the prompt.",
    )
    n_documents = st.slider("Passages to retrieve", 1, 10, 3, disabled=not use_rag)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Knowledge base")
    if "knowledge_reset_message" in st.session_state:
        st.success(st.session_state.pop("knowledge_reset_message"))
    if st.button("Reset knowledge base"):
        st.session_state.confirm_knowledge_reset = True
    if st.session_state.get("confirm_knowledge_reset", False):
        st.warning(
            "This permanently removes all uploaded passages for every user. "
            "It does not clear chat history or original files."
        )
        if st.button("Confirm reset", type="primary"):
            try:
                deleted = load_knowledge_base().clear()
            except (ChromaError, HTTPError, OSError, RuntimeError, sqlite3.Error, ValueError) as error:
                st.error(f"Could not reset the knowledge base; some passages may already have been deleted: {error}")
            else:
                st.session_state.confirm_knowledge_reset = False
                st.session_state.knowledge_reset_message = f"Knowledge base reset ({deleted} passages removed)."
                st.rerun()
        if st.button("Cancel reset"):
            st.session_state.confirm_knowledge_reset = False
            st.rerun()

st.title("OpenWebText LLM")

if not checkpoint:
    st.warning("No checkpoints found in checkpoints/. Train the model first.")
    st.stop()

tokenizer = load_tokenizer()
try:
    model = load_model(
        checkpoint,
        file_version(checkpoint),
        adapter,
        file_version(adapter) if adapter else None,
        alpha if adapter else None,
    )
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the selected model weights: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved passages"):
                for source in message["sources"]:
                    st.caption(f"{source['source']} — chunk {source['chunk']}")
                    st.text(source["text"])

submission = st.chat_input("Start a sentence or upload .txt files...", accept_file="multiple", file_type=["txt"])
if submission:
    prompt = submission.text.strip()
    failed_upload = False
    knowledge_base = None
    if submission.files or (prompt and use_rag):
        try:
            knowledge_base = load_knowledge_base()
        except (ChromaError, HTTPError, OSError, RuntimeError, sqlite3.Error, ValueError) as error:
            st.error(f"Could not open the knowledge base: {error}")
            st.stop()

    for uploaded_file in submission.files:
        try:
            count = knowledge_base.add_file(uploaded_file.name, uploaded_file.getvalue())
        except (ChromaError, HTTPError, ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not add {uploaded_file.name}: {error}")
            failed_upload = True
        else:
            if count:
                st.success(f"Added {uploaded_file.name} to the knowledge base ({count} passages).")
            else:
                st.info(f"{uploaded_file.name} is already in the knowledge base.")
    if submission.files and not use_rag:
        st.info("Enable 'Use knowledge base' in Settings to use uploaded files in answers.")
    if failed_upload or not prompt:
        st.stop()

    sources = []
    prompt_ids = None
    if use_rag:
        try:
            documents = knowledge_base.search(prompt, n_documents)
            if not documents:
                st.info("No matching passages found; generating without knowledge-base context.")
            prompt_ids, sources = build_rag_prompt(
                load_prompt_tokenizer(), prompt, documents,
                st.session_state.messages if use_history else [],
                configs.MAX_SEQ_LENGTH - max_new_tokens,
            )
            context = prompt
            if documents and not sources:
                st.warning("Matching passages could not fit in the model's context window.")
        except (ChromaError, HTTPError, ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not retrieve knowledge-base context: {error}")
            st.stop()
    elif use_history:
        context = "\n\n".join(m["content"] for m in st.session_state.messages + [{"content": prompt}])
    else:
        context = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if sources:
            with st.expander("Retrieved passages"):
                for source in sources:
                    st.caption(f"{source['source']} — chunk {source['chunk']}")
                    st.text(source["text"])
        placeholder = st.empty()
        new_ids = []
        for token_id in stream_tokens(model, tokenizer, context, max_new_tokens=max_new_tokens,
                                      top_k=top_k, top_p=top_p, temperature=temperature, sampling=sampling,
                                      repetition_penalty=repetition_penalty, prompt_ids=prompt_ids):
            new_ids.append(token_id)
            placeholder.markdown(tokenizer.decode(new_ids, skip_special_tokens=True))
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
