"""Run small reproducible BM25 vs optional hybrid search evaluations.

python evaluate_retrieval.py --dataset examples/sample_rag_eval.json --k 3
python evaluate_retrieval.py --dataset my_eval.json --k 5

Input format: {"documents": [{"source":"x.txt", "text":"..."}],
               "queries": [{"query":"...", "relevant_sources":["x.txt"]}]}

This tests retrieval ONLY. It does not test the factual accuracy of Gemini outputs.
"""
import argparse
import json
import os
from rag import RagIndex


def evaluate(index: RagIndex, queries, k: int):
    total = 0
    hits = 0
    reciprocal_ranks = []
    modes = set()
    for item in queries:
        relevant = set(item['relevant_sources'])
        if not relevant:
            continue
        results = index.search(item['query'], k=k)
        modes.add(index.last_search['mode'] + ('+rerank' if index.last_search['reranked'] else ''))
        sources = [r['source'] for r in results]
        ranks = [j + 1 for j, src in enumerate(sources) if src in relevant]
        hits += bool(ranks)
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        total += 1
    return {'num_queries': total, 'hit_at_k': round(hits / total, 4) if total else 0,
            'mrr_at_k': round(sum(reciprocal_ranks) / total, 4) if total else 0,
            'actual_modes': sorted(modes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='examples/sample_rag_eval.json')
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args()
    dataset = json.load(open(args.dataset, encoding='utf-8'))
    idx = RagIndex()
    for doc in dataset['documents']:
        idx.add_text(doc['text'], doc['source'])
    old_embedding, old_rerank = os.environ.get('NOVA_RAG_EMBEDDINGS'), os.environ.get('NOVA_RAG_RERANK')
    try:
        os.environ['NOVA_RAG_EMBEDDINGS'] = 'off'
        os.environ['NOVA_RAG_RERANK'] = 'off'
        bm25 = evaluate(idx, dataset['queries'], args.k)
        os.environ['NOVA_RAG_EMBEDDINGS'] = old_embedding or 'auto'
        os.environ['NOVA_RAG_RERANK'] = old_rerank or 'auto'
        combined = evaluate(idx, dataset['queries'], args.k)
    finally:
        if old_embedding is None: os.environ.pop('NOVA_RAG_EMBEDDINGS', None)
        else: os.environ['NOVA_RAG_EMBEDDINGS'] = old_embedding
        if old_rerank is None: os.environ.pop('NOVA_RAG_RERANK', None)
        else: os.environ['NOVA_RAG_RERANK'] = old_rerank
    print(json.dumps({'benchmark': os.path.basename(args.dataset), 'note': 'Retrieval-only evaluation; sample queries are illustrative.',
                      'bm25': bm25, 'configured_retriever': combined}, indent=2))


if __name__ == '__main__':
    main()
