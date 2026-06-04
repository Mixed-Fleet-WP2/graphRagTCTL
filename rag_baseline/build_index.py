#!/usr/bin/env python3
"""
Build the flat RAG vector index from knowledge_base.json.

Each chunk (pattern, operator, example) is embedded using OpenAI's
text-embedding-3-small model and stored as a numpy matrix alongside
a metadata JSON file for lookup.

Run once before using pipeline_rag.py:
    python build_index.py

Output:
    rag_index/embeddings.npy   - shape (N, 1536) float32
    rag_index/metadata.json    - list of N chunk metadata dicts
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
KB_PATH = Path(__file__).parent / "knowledge_base.json"
INDEX_DIR = Path(__file__).parent / "rag_index"
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536-dim, cheap, fast
BATCH_SIZE = 50                               # OpenAI allows up to 2048 inputs/call


def load_chunks(kb_path: Path) -> List[Dict[str, Any]]:
    """Load all chunks (patterns + operators + examples) from knowledge base."""
    with open(kb_path) as f:
        kb = json.load(f)

    chunks: List[Dict[str, Any]] = []

    for pat in kb.get("patterns", []):
        chunks.append({
            "id": pat["id"],
            "type": "pattern",
            "name": pat["name"],
            "template": pat.get("template", ""),
            "when_to_use": pat.get("when_to_use", ""),
            "text": pat["text"],
        })

    for op in kb.get("operators", []):
        chunks.append({
            "id": op["id"],
            "type": "operator",
            "name": op["name"],
            "when_to_use": op.get("when_to_use", ""),
            "when_not_to_use": op.get("when_not_to_use", ""),
            "text": op["text"],
        })

    for ex in kb.get("examples", []):
        chunks.append({
            "id": ex["id"],
            "type": "example",
            "nl": ex["nl"],
            "uppaal": ex["uppaal"],
            "pattern": ex["pattern"],
            "text": ex["text"],
        })

    print(f"Loaded {len(chunks)} chunks  "
          f"({sum(1 for c in chunks if c['type']=='pattern')} patterns, "
          f"{sum(1 for c in chunks if c['type']=='operator')} operators, "
          f"{sum(1 for c in chunks if c['type']=='example')} examples)")
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Embed a list of texts in batches; return (N, dim) float32 array."""
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded batch {i // BATCH_SIZE + 1} "
              f"({min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks)")
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.3)   # minor rate-limit courtesy

    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    print("─" * 60)
    print("RAG INDEX BUILDER")
    print("─" * 60)

    # 1. Load chunks
    chunks = load_chunks(KB_PATH)

    # 2. Embed
    texts = [c["text"] for c in chunks]
    print(f"\nEmbedding {len(texts)} chunks with {EMBEDDING_MODEL} ...")
    embeddings = embed_texts(client, texts)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # 3. Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "metadata.json"

    np.save(str(emb_path), embeddings)
    with open(meta_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\n✓ Saved embeddings → {emb_path}")
    print(f"✓ Saved metadata   → {meta_path}")
    print("─" * 60)
    print("Index ready. You can now run pipeline_rag.py")
    print("─" * 60)


if __name__ == "__main__":
    main()
