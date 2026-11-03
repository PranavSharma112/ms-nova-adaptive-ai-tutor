"""Session-local, source-traceable hybrid retrieval for Ms. Nova.

- BM25 always available, no network needed.
- Optional SentenceTransformer dense search + reciprocal rank fusion (RRF).
- Optional CrossEncoder reranking of the fused candidates.

Optional models are only loaded when first used. They may download weights from a
model registry; set NOVA_RAG_EMBEDDINGS=off / NOVA_RAG_RERANK=off for offline use.
Never send uploaded text to the embedding/cross-encoder services: all model
inference runs locally after installation.
"""
from __future__ import annotations
import importlib.util
import io
import math
import os
import re
import threading
from collections import Counter
from typing import List, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_MODEL_LOCK = threading.RLock()
_MODEL_CACHE = {}

def _tokens(text: str):
    return _TOKEN.findall(text.lower())

def _flag(name: str):
    return os.environ.get(name, 'auto').strip().lower()

def _enabled(name: str):
    flag = _flag(name)
    return flag not in ('0', 'off', 'false', 'no')

def _optional_available():
    return importlib.util.find_spec('sentence_transformers') is not None

def _load_model(kind: str):
    """Models cached across sessions; on explicit or auto opt-in if installed."""
    if not _optional_available():
        return None
    env = 'NOVA_RAG_EMBEDDINGS' if kind == 'embedding' else 'NOVA_RAG_RERANK'
    if not _enabled(env):
        return None
    model_id = os.environ.get('NOVA_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2') if kind == 'embedding' else os.environ.get('NOVA_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    with _MODEL_LOCK:
        key = (kind, model_id)
        if key not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            cls = SentenceTransformer if kind == 'embedding' else CrossEncoder
            _MODEL_CACHE[key] = cls(model_id)
        return _MODEL_CACHE[key]

class RagIndex:
    def __init__(self):
        self.chunks: List[dict] = []
        self.df: Counter = Counter()
        self._vector_cache = None
        self._cache_size = -1
        self.last_search = {'mode': 'bm25', 'reranked': False, 'dense_error': '', 'rerank_error': ''}

    def __len__(self):
        return len(self.chunks)

    def add_text(self, text: str, source: str, chunk_words: int = 180, overlap: int = 30):
        words = text.split()
        step = max(1, chunk_words - overlap)
        for i in range(0, len(words), step):
            piece = ' '.join(words[i:i + chunk_words]).strip()
            if len(piece) < 40:
                continue
            toks = _tokens(piece)
            self.df.update(set(toks))
            self.chunks.append({'id': len(self.chunks) + 1, 'source': source,
                                'text': piece, 'tf': Counter(toks), 'len': len(toks)})
        self._vector_cache, self._cache_size = None, -1

    def add_file(self, name: str, data: bytes) -> int:
        before = len(self.chunks)
        if name.lower().endswith('.pdf'):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                for n, page in enumerate(reader.pages, 1):
                    self.add_text(page.extract_text() or '', f'{name} p.{n}')
            except Exception as e:
                raise RuntimeError(f"Couldn't read PDF '{name}': {e}")
        else:
            self.add_text(data.decode('utf-8', errors='ignore'), name)
        return len(self.chunks) - before

    def _lexical_scores(self, query: str):
        n = len(self.chunks)
        if not n:
            return []
        avgdl = sum(c['len'] for c in self.chunks) / n
        scores = []
        for c in self.chunks:
            score = 0.0
            for token in set(_tokens(query)):
                f = c['tf'].get(token, 0)
                if f:
                    idf = math.log(1 + (n - self.df[token] + 0.5) / (self.df[token] + 0.5))
                    score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * c['len'] / max(avgdl, 1)))
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 5, candidate_multiplier: int = 5):
        """Return source-id, source, text, score, method and reranking metadata."""
        query = (query or '').strip()[:400]
        k = max(1, min(20, int(k)))
        if not query or not self.chunks:
            self.last_search = {'mode': 'bm25', 'reranked': False, 'dense_error': '', 'rerank_error': ''}
            return []
        lex = self._lexical_scores(query)
        lexical_order = sorted((i for i, s in enumerate(lex) if s > 0), key=lambda i: (-lex[i], i))
        dense_order, dense_scores = [], {}
        dense_error = ''
        try:
            model = _load_model('embedding')
            if model is not None:
                if self._vector_cache is None or self._cache_size != len(self.chunks):
                    self._vector_cache = np.asarray(model.encode([c['text'] for c in self.chunks],
                                                                normalize_embeddings=True, show_progress_bar=False), dtype=float)
                    self._cache_size = len(self.chunks)
                qv = np.asarray(model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=float)
                sims = np.asarray(self._vector_cache @ qv, dtype=float)
                dense_scores = {i: float(v) for i, v in enumerate(sims) if np.isfinite(v)}
                dense_order = sorted(dense_scores, key=lambda i: (-dense_scores[i], i))
        except Exception as exc:
            dense_error = str(exc)[:160]
        mode = 'hybrid' if dense_order else 'bm25'
        fused = {}
        for order in (lexical_order, dense_order):
            for rank, idx in enumerate(order):
                fused[idx] = fused.get(idx, 0.) + 1. / (60 + rank + 1)
        candidates = sorted(fused, key=lambda i: (-fused[i], i))[:max(k, k * candidate_multiplier)]
        reranked = False
        rerank_scores = {}
        rerank_error = ''
        if candidates:
            try:
                cross_encoder = _load_model('rerank') if dense_order else None
                if cross_encoder is not None:
                    raw = cross_encoder.predict([(query, self.chunks[i]['text']) for i in candidates])
                    rerank_scores = {i: float(score) for i, score in zip(candidates, raw)}
                    if len(rerank_scores) != len(candidates) or not all(np.isfinite(x) for x in rerank_scores.values()):
                        raise ValueError('Reranker returned incomplete scores')
                    candidates.sort(key=lambda i: (-rerank_scores[i], -fused[i], i))
                    reranked = True
            except Exception as exc:
                rerank_error = str(exc)[:160]
                rerank_scores = {}
        self.last_search = {'mode': mode, 'reranked': reranked, 'dense_error': dense_error, 'rerank_error': rerank_error}
        results = []
        for i in candidates[:k]:
            c = self.chunks[i]
            results.append({'id': c['id'], 'source': c['source'], 'text': c['text'],
                            'score': round(rerank_scores.get(i, fused[i]), 6),
                            'bm25_score': round(lex[i], 6),
                            'semantic_score': round(dense_scores[i], 6) if i in dense_scores else None,
                            'rerank_score': round(rerank_scores[i], 6) if i in rerank_scores else None})
        return results

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, str, str]]:
        """Compatibility with the existing teacher and Learning Lab interfaces."""
        return [(h['id'], h['source'], h['text']) for h in self.search(query, k=k)]

    def status(self):
        return {'chunks': len(self.chunks), 'last_search': dict(self.last_search),
                'sentence_transformers_installed': _optional_available(),
                'embedding_setting': _flag('NOVA_RAG_EMBEDDINGS'),
                'reranker_setting': _flag('NOVA_RAG_RERANK'),
                'embedding_model': os.environ.get('NOVA_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                'reranker_model': os.environ.get('NOVA_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')}

def format_context(hits) -> str:
    return '\n\n'.join(f'[S{i}] ({src}) {txt}' for i, src, txt in hits)
