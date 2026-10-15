"""Local web server for the original Engineering Assistant interface."""

import json
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

BASE_DIR = Path(__file__).resolve().parent


def create_app(engine=None):
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 16384

    @app.get('/')
    def index():
        return send_from_directory(BASE_DIR, 'tobroco_assistant.html')

    @app.get('/logo.png')
    def logo():
        return send_from_directory(BASE_DIR, 'logo.png')

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error='Request too large'), 413

    @app.post('/api/stream')
    def answer():
        nonlocal engine
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('question'), str):
            return jsonify(error='Provide a question as JSON text'), 400
        question = payload['question'].strip()
        if not question or len(question) > 2000:
            return jsonify(error='Question must contain 1 to 2000 characters'), 400

        def event(data):
            return f'data: {json.dumps(data)}\n\n'

        def generate():
            nonlocal engine
            try:
                if engine is None:
                    from assistant import EngineeringAssistant
                    engine = EngineeringAssistant()
                sources = engine.search(question)
                yield event({'type': 'sources', 'data': [{key: source[key] for key in ('source', 'page', 'location_type')} for source in sources]})
                if not sources:
                    yield event({'type': 'token', 'text': 'No documents indexed yet. Run python assistant.py examples to index the fictional demo manual, or index your own documents.'})
                else:
                    for chunk in engine.stream_answer(question, sources):
                        token = chunk['message']['content']
                        if token:
                            yield event({'type': 'token', 'text': token})
            except Exception:
                app.logger.exception('Document query failed')
                yield event({'type': 'token', 'text': 'Unable to query the local model. Check that Ollama is running, both models are downloaded and documents have been indexed. See the server log for details.'})
            yield event({'type': 'done'})

        return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '5001')), debug=False)
