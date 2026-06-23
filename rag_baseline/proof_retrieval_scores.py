#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute top-k cosine retrieval scores as proof")
    parser.add_argument("constraint", type=str)
    parser.add_argument("--expected-pattern", type=str, default="safety_immediate_response")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default="rag_baseline/retrieval_score_proof.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    meta_path = root / "rag_baseline" / "rag_index" / "metadata.json"
    emb_path = root / "rag_baseline" / "rag_index" / "embeddings.npy"

    metadata_bytes = meta_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    embeddings = np.load(str(emb_path)).astype(np.float32)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=[args.constraint])
    query = np.array(resp.data[0].embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    scores = embeddings_norm @ query_norm
    top_idx = np.argsort(scores)[::-1][: args.k]

    top = []
    for rank, idx in enumerate(top_idx, start=1):
        chunk = metadata[int(idx)]
        top.append(
            {
                "rank": rank,
                "id": chunk.get("id"),
                "type": chunk.get("type"),
                "pattern": chunk.get("pattern", chunk.get("name")),
                "score": round(float(scores[int(idx)]), 4),
                "nl": chunk.get("nl"),
            }
        )

    out = {
        "constraint": args.constraint,
        "expected_pattern": args.expected_pattern,
        "embedding_model": "text-embedding-3-small",
        "index_files": {
            "metadata": str(meta_path),
            "embeddings": str(emb_path),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "embedding_shape": list(embeddings.shape),
        },
        "top_k": args.k,
        "top_chunks": top,
    }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
