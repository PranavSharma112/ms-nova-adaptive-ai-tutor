"""Optional GenAI study tools, grounded in uploaded materials and session history."""
import re
from typing import Literal
from pydantic import BaseModel, Field
import teacher_engine as te
from rl_engine import decode_state

class Card(BaseModel):
    question: str = Field(min_length=5, max_length=350)
    answer: str = Field(min_length=2, max_length=850)
    source: str = Field(default='', max_length=160)

class CardSet(BaseModel):
    cards: list[Card] = Field(min_length=1, max_length=16)

class Step(BaseModel):
    topic: str
    activity: str
    minutes: int = Field(ge=5, le=90)
    reason: str

class StudyPlan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=12)
    note: str = ''

class Misconception(BaseModel):
    concept: str
    explanation: str
    practice_question: str
    checking_question: str

class MisconceptionSet(BaseModel):
    findings: list[Misconception] = Field(default_factory=list, max_length=8)
    guidance: str = ''

def _profile(session):
    topics = session.progress()['topics']
    return {'subject': session.subject, 'topics': topics, 'recent_mistakes': session.misconceptions[-8:]}

def _evidence(session, query, k=5):
    hits = session.rag.retrieve(query, k=k)
    return [{'id': n, 'source': source, 'text': text[:900]} for n, source, text in hits]

def flashcards(session, topic='', count=6):
    topic = (topic or session.topic_name() or 'the course')[:100]
    count = max(3, min(12, int(count)))
    evidence = _evidence(session, topic, 6)
    if not evidence:
        return {'cards': [], 'evidence': [], 'message': 'Upload study notes containing the topic first. Flashcards are only generated from uploaded evidence.'}
    formatted = '\n'.join(f"[S{h['id']}] {h['source']}: {h['text']}" for h in evidence)
    prompt = (f'Create {count} accurate educational question-answer flashcards about {topic}. '
              'Use ONLY the source excerpts below. No invented facts. Each card source MUST match the source filename exactly as given. '
              'Avoid answer leakage in the question. Cover distinct concepts.\n\n' + formatted)
    result = te._generate_json(prompt, CardSet, session.api_key, temperature=0.2)
    allowed = {h['source'] for h in evidence}
    cards = [x.model_dump() for x in result.cards if x.source in allowed][:count]
    return {'cards': cards, 'evidence': evidence, 'message': '' if cards else 'No source-matched cards passed validation. Try another topic.'}

def plan(session, minutes=45):
    minutes = max(15, min(180, int(minutes)))
    info = _profile(session)
    topics = info['topics']
    if not topics:
        return {'steps': [], 'note': 'Start a course and complete some questions first.'}
    ordered = sorted(topics, key=lambda t: (t['mastery'], -t['n']))
    weights = [max(0.2, 1 - t['mastery']) for t in ordered[:4]]
    remaining = minutes
    steps=[]
    for i, (topic, w) in enumerate(zip(ordered[:4], weights)):
        mins = remaining if i == len(weights)-1 else max(5, round(minutes * w/sum(weights)/5)*5)
        mins = min(remaining, mins)
        if mins < 5: break
        remaining-=mins
        steps.append({'topic':topic['name'], 'activity':'Review the concept, then answer 3 practice questions',
                      'minutes':mins, 'reason':f"Current estimated mastery: {topic['mastery']:.0%}; {topic['n']} questions completed"})
    if steps and remaining: steps[-1]['minutes']+=remaining
    return {'steps':steps, 'note':'This plan uses measured topic mastery, not an LLM guess. It is a suggested schedule and can be adjusted.'}

def coach(session):
    mistakes = [re.sub(r'\s+', ' ', m)[:250] for m in session.misconceptions[-8:]]
    if not mistakes:
        return {'findings': [], 'guidance':'No misconceptions logged yet. Complete a few questions to receive targeted coaching.'}
    prompt = ('Act as an educational diagnostic coach. Analyze the following recorded mistake notes. '
              'Distinguish confirmed misunderstandings from possible ones; never invent student mistakes. '
              'For each finding provide a gentle explanation, one practice question and one checking question. '
              'Avoid diagnoses or claims about ability.\nSubject: '+str(session.subject)+'\nMistake notes:\n'+'\n'.join(mistakes))
    result = te._generate_json(prompt, MisconceptionSet, session.api_key, temperature=0.2)
    return result.model_dump()

def explain_rl(session):
    d = session.last_decision
    if not d:
        return {'ready':False, 'message':'Complete at least one adaptive question to see the actual RL decision.'}
    state = d.get('state')
    if not isinstance(state, int):
        return {'ready':False,'message':'No decoded policy state was logged for the most recent decision.'}
    m, difficulty, trend = decode_state(state)
    action = d.get('action')
    return {'ready': True, 'state_id':state, 'mastery_bin':m, 'previous_difficulty':difficulty+1,
            'trend_bin':trend, 'chosen_action':action,
            'action_label': 'Show worked example' if action==5 else f'Ask level {action+1} question' if isinstance(action,int) and 0<=action<=4 else 'Unknown',
            'propensity':d.get('propensity'), 'raw_decision':d,
            'note':'This shows logged state and selected action; it is not a causal explanation of why the policy learned its Q-values.'}

class ConceptEdge(BaseModel):
    prerequisite: str
    concept: str
    relationship: str

class ConceptMap(BaseModel):
    concepts: list[str] = Field(min_length=1, max_length=20)
    links: list[ConceptEdge] = Field(default_factory=list, max_length=25)

class PracticeItem(BaseModel):
    question: str
    answer: str
    explanation: str
    source: str

class PracticeSet(BaseModel):
    questions: list[PracticeItem] = Field(min_length=1, max_length=10)

def concept_map(session, topic=''):
    topic=(topic or session.topic_name() or 'course')[:100]
    hits=_evidence(session,topic,7)
    if not hits: return {'concepts':[], 'links':[], 'message':'Upload notes relevant to this topic first.'}
    excerpt='\n'.join(f"{h['source']}: {h['text']}" for h in hits)
    prompt=('Extract key concepts and prerequisite links from these notes about '+topic+'. '
            'Only use concepts and relationships actually supported by the excerpts. '
            'An empty links list is acceptable if prerequisites cannot be established.\n'+excerpt)
    result=te._generate_json(prompt, ConceptMap,session.api_key,temperature=0.15)
    concepts=list(dict.fromkeys(result.concepts))[:20]
    edges=[x.model_dump() for x in result.links if x.prerequisite in concepts and x.concept in concepts]
    return {'concepts':concepts,'links':edges,'evidence':hits}

def grounded_practice(session, topic='',count=5):
    topic=(topic or session.topic_name() or 'course')[:100]
    count=max(3,min(8,int(count)))
    hits=_evidence(session,topic,7)
    if not hits: return {'questions':[], 'message':'Upload relevant notes first.'}
    excerpt='\n'.join(f"{h['source']}: {h['text']}" for h in hits)
    prompt=(f'Produce {count} short-answer revision questions with correct answers and explanations '
            f'about {topic}, using ONLY the source excerpts. Each question must be answerable from '
            'the excerpts and attribute the exact matching source filename. Do not include outside facts.\n'+excerpt)
    result=te._generate_json(prompt,PracticeSet,session.api_key,temperature=0.2)
    sources={x['source'] for x in hits}
    questions=[x.model_dump() for x in result.questions if x.source in sources][:count]
    return {'questions':questions,'evidence':hits,'message':'' if questions else 'No questions passed the source validation.'}
