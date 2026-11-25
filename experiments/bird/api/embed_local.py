"""Local MiniLM encoder for the dense stages of the API methods.

Same cached weights as the workspace retrievers.dense_retriever._embed
(paraphrase-multilingual-MiniLM-L12-v2 @ revision
e8f8c211226b894fcb81acc59f3b34ba3efd5f42), loaded from the snapshot path with
local_files_only=True because sentence_transformers is broken in the pinned
environment. Pooling math is IDENTICAL (mean pool over the attention mask,
L2 normalization, 256-token truncation, batch 64, in = out order), so the
embeddings equal the workspace ones; only the loading path differs.
"""
import numpy as np

from experiments.bird.api.config import MINILM_SNAPSHOT

BATCH_SIZE = 64
MAX_TOKENS = 256
_MODEL = None
_TOKENIZER = None


def _get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _TOKENIZER = AutoTokenizer.from_pretrained(MINILM_SNAPSHOT,
                                                   local_files_only=True)
        _MODEL = AutoModel.from_pretrained(MINILM_SNAPSHOT,
                                           local_files_only=True)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def _embed(texts, batch_size=BATCH_SIZE):
    """Batch-encode texts; identical math to dense_retriever._embed."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    import torch
    tok, model = _get_model()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.numpy())
    result = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / (norms + 1e-10)
