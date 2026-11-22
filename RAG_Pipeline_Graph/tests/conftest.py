"""
Shared fixtures. Everything runs offline: hash embeddings (no model download), the extractive
generator or a scripted LLM double, and sample data copied into a temp folder.
"""
import importlib.util

import pytest

from graph_rag import load_config
from graph_rag.demo_data import seed_demo_data
from graph_rag.embeddings import HashEmbedder
from graph_rag.index import build_graph_index

HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
HAS_PDFPLUMBER = importlib.util.find_spec("pdfplumber") is not None


class ScriptedLLM:
    """Returns a fixed answer and records the prompts it was given (plain `generate(prompt)` LLM)."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def make_config(tmp_path, **overrides):
    cfg = load_config(use_env=False)
    cfg.update(seed_demo_data(str(tmp_path / "data")))
    cfg.update({"index_dir": str(tmp_path / "idx"), "embedding_backend": "hash", "use_spacy": False,
                "vector_backend": "memory"})
    cfg.update(overrides)
    return cfg


@pytest.fixture()
def cfg(tmp_path):
    return make_config(tmp_path)


@pytest.fixture()
def index(cfg):
    return build_graph_index(cfg, embedding_fn=HashEmbedder())


@pytest.fixture()
def env(cfg, index):
    """(config, quad store, vector store) - the shape the original test-suite used."""
    return cfg, index.store, index.vstore
