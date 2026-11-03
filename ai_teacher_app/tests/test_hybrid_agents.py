"""Offline tests for source-safe retrieval and orchestration; no external keys."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['NOVA_RAG_EMBEDDINGS'] = 'off'
os.environ['NOVA_RAG_RERANK'] = 'off'

import numpy as np
import pytest
import rag
import multi_agent as agents
from tutor_service import TutorSession


@pytest.fixture
def idx():
    r = rag.RagIndex()
    r.add_text('Photosynthesis converts light energy into chemical energy in plants using chlorophyll and water.', 'biology.txt')
    r.add_text('Mitosis divides one parent cell into daughter cells in the cell cycle.', 'cells.txt')
    return r


def test_bm25_stable_ids_and_sources(idx):
    hits = idx.search('chlorophyll and sunlight', k=4)
    assert hits[0]['source'] == 'biology.txt'
    assert hits[0]['id'] == 1
    assert idx.status()['last_search']['mode'] == 'bm25'
    assert idx.retrieve('chlorophyll', k=2)[0][:2] == (1, 'biology.txt')
    assert '[S1]' in rag.format_context(idx.retrieve('chlorophyll', k=2))


def test_hybrid_and_reranker_with_fake_models(monkeypatch, idx):
    class FakeEmbed:
        def encode(self, texts, **kwargs):
            out = []
            for t in texts:
                t = t.lower()
                out.append([1., 0.] if ('photosynthesis' in t or 'plant food' in t) else [0., 1.])
            return np.array(out)

    class FakeRerank:
        def predict(self, pairs):
            return np.array([20. if 'photosynthesis' in text.lower() else -2. for _, text in pairs])

    monkeypatch.setattr(rag, '_load_model', lambda kind: FakeEmbed() if kind == 'embedding' else FakeRerank())
    results = idx.search('plant food', k=2)
    assert idx.status()['last_search']['mode'] == 'hybrid'
    assert idx.status()['last_search']['reranked'] is True
    assert results[0]['source'] == 'biology.txt'
    assert results[0]['semantic_score'] is not None
    assert results[0]['rerank_score'] is not None


def test_dense_fails_over_to_bm25(monkeypatch, idx):
    def load(kind):
        raise RuntimeError('model files unavailable')
    monkeypatch.setattr(rag, '_load_model', load)
    results = idx.search('mitosis parent cell', k=2)
    assert results[0]['source'] == 'cells.txt'
    assert idx.last_search['mode'] == 'bm25'
    assert 'unavailable' in idx.last_search['dense_error']


def test_no_uploaded_sources_abstains():
    session = TutorSession(api_key='test')
    out = agents.run(session, 'Explain neural networks')
    assert out['status'] == 'no_sources'
    assert out['answer'] == ''


def test_agents_model_reviewed_with_true_sources(monkeypatch):
    s = TutorSession(api_key='fake-test-key')
    s.subject = 'Biology'
    s.rag.add_text('Photosynthesis converts light energy into chemical energy inside chloroplasts.', 'biology.txt')
    monkeypatch.setattr(rag, '_load_model', lambda kind: None)
    def mock_gen(prompt, cls, key, **kw):
        if cls is agents.LessonPlan:
            return agents.LessonPlan(retrieval_query='photosynthesis chemical energy', focus='energy transfer', method='step_by_step')
        if cls is agents.TeachingDraft:
            return agents.TeachingDraft(answer='Photosynthesis converts light energy into chemical energy [S1].', source_ids=[1], check_question='What form of energy is produced?')
        if cls is agents.CriticReview:
            return agents.CriticReview(supported=True, follows_instruction=True)
        raise ValueError(cls)
    monkeypatch.setattr(agents.te, '_generate_json', mock_gen)
    out = agents.run(s, 'Explain photosynthesis')
    assert out['status'] == 'model_reviewed'
    assert out['evidence'][0]['id'] == 1
    assert [t['stage'] for t in out['trace']] == ['Planner', 'Retriever', 'Teacher', 'Critic']
    assert '[S1]' in out['answer']


def test_agents_reject_fabricated_source_id(monkeypatch):
    s = TutorSession(api_key='fake-test-key')
    s.rag.add_text('Photosynthesis converts light into chemical energy in chloroplasts.', 'biology.txt')
    monkeypatch.setattr(rag, '_load_model', lambda kind: None)
    def mock_gen(prompt, cls, key, **kw):
        if cls is agents.LessonPlan:
            return agents.LessonPlan(retrieval_query='photosynthesis', focus='basics', method='summary')
        if cls is agents.TeachingDraft:
            return agents.TeachingDraft(answer='An unsupported claim supposedly from [S999].', source_ids=[999])
        raise AssertionError('Critic should never be asked when citations are fabricated')
    monkeypatch.setattr(agents.te, '_generate_json', mock_gen)
    out = agents.run(s, 'Explain photosynthesis')
    assert out['status'] == 'unverified'
    assert not out['answer']


def test_agents_reject_unsupportable_answer(monkeypatch):
    s = TutorSession(api_key='fake-test-key')
    s.rag.add_text('Photosynthesis converts light into chemical energy in chloroplasts.', 'biology.txt')
    monkeypatch.setattr(rag, '_load_model', lambda kind: None)
    def mock_gen(prompt, cls, key, **kw):
        if cls is agents.LessonPlan:
            return agents.LessonPlan(retrieval_query='photosynthesis', focus='basics', method='summary')
        if cls is agents.TeachingDraft:
            return agents.TeachingDraft(answer='An invented claim [S1] is true.', source_ids=[1])
        if cls is agents.CriticReview:
            return agents.CriticReview(supported=False, follows_instruction=True, issues=['Not in source'])
        raise ValueError(cls)
    monkeypatch.setattr(agents.te, '_generate_json', mock_gen)
    out = agents.run(s, 'Explain photosynthesis')
    assert out['status'] == 'unverified'
    assert out['review']['supported'] is False
    assert out['answer'] == ''


def test_learning_lab_endpoints_bm25_and_agent_toggle():
    from fastapi.testclient import TestClient
    from server import app, SESSIONS
    client = TestClient(app)
    s = TutorSession(api_key='demo')
    SESSIONS[s.id] = s
    resp = client.post(f'/api/session/{s.id}/settings', json={'agent_mode': True})
    assert resp.status_code == 200
    assert resp.json()['settings']['agent_mode'] is True
    status = client.get(f'/api/session/{s.id}/lab/rag-status')
    assert status.status_code == 200
    assert status.json()['chunks'] == 0
    agent = client.post(f'/api/session/{s.id}/lab/agent-teach', json={'question': 'hello'})
    assert agent.status_code == 400  # demo key cannot call Gemini
    SESSIONS.pop(s.id, None)
