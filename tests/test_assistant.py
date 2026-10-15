import json

import pytest

from app import create_app
from assistant import EngineeringAssistant, chunk_text, extract_document


def test_chunking_overlap():
    assert chunk_text('one two three four five six', size=4, overlap=1) == ['one two three four', 'four five six']
    assert chunk_text('') == []
    with pytest.raises(ValueError):
        chunk_text('test', size=2, overlap=2)


def test_text_and_docx_extraction(tmp_path):
    from docx import Document
    text = tmp_path / 'example.txt'
    text.write_text('A fictional drawing review.', encoding='utf-8')
    assert extract_document(text) == [('A fictional drawing review.', 1, 'section')]
    document = Document()
    document.add_paragraph('Review the drawing.')
    path = tmp_path / 'example.docx'
    document.save(path)
    assert extract_document(path) == [('Review the drawing.', 1, 'section')]


class LocalTestModel:
    def embed(self, model, input):
        return {'embeddings': [[1.0 + text.lower().count('review'), 1.0 + text.lower().count('drawing'), 1.0] for text in input]}

    def chat(self, **kwargs):
        assert 'Source: sample.txt' in kwargs['messages'][1]['content']
        return iter([{'message': {'content': 'According to sample.txt, section 1: review the drawing.'}}])


def test_real_index_is_idempotent_and_retrievable(tmp_path):
    documents = tmp_path / 'documents'
    documents.mkdir()
    path = documents / 'sample.txt'
    path.write_text('Review the drawing before release.', encoding='utf-8')
    engine = EngineeringAssistant(tmp_path / 'database', client=LocalTestModel())
    assert engine.search('drawing') == []
    assert engine.ingest(documents) == 1
    assert engine.ingest(documents) == 1
    assert engine.collection.count() == 1
    result = engine.search('drawing review')
    assert result[0]['source'] == 'sample.txt'
    assert result[0]['location_type'] == 'section'
    response = create_app(engine).test_client().post('/api/stream', json={'question': 'Who reviews the drawing?'})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[0]['type'] == 'sources'
    assert 'review the drawing' in events[1]['text']
    assert events[-1]['type'] == 'done'


def test_stream_without_documents():
    class EmptyEngine:
        def search(self, question):
            return []
    client = create_app(EmptyEngine()).test_client()
    response = client.post('/api/stream', json={'question': 'A question'})
    assert 'No documents indexed' in response.text
    assert '"type": "done"' in response.text


def test_input_validation_and_static_boundaries():
    client = create_app().test_client()
    for payload in [None, [], {}, {'question': 10}, {'question': ''}, {'question': 'x' * 2001}]:
        assert client.post('/api/stream', json=payload).status_code == 400
    assert client.get('/').status_code == 200
    assert client.get('/logo.png').status_code == 200
    assert client.get('/chroma_db/chroma.sqlite3').status_code == 404
    assert client.get('/app.py').status_code == 404
