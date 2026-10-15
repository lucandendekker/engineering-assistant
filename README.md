# Engineering Assistant

**Ask questions about engineering documents and get answers with source references.**

A Retrieval-Augmented Generation (RAG) prototype developed by **Lucan den Dekker, Roef Peters and Rutger Baas** for **TOBROCO-GIANT** during Applied Data Science & AI at Breda University of Applied Sciences. In two weeks, we moved from a client brief to a working prototype.

[Project story on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7426908403483987968/) | [Portfolio project page](https://lucandendekker.nl/portfolio/engineering-assistant/)

## The challenge

Engineers work with extensive CAD design standards distributed across many documents. Finding the right guidance takes time, especially for new colleagues. Our prototype made those documents searchable through natural-language questions, with references back to the original filename and page.

## How it works

```text
PDF / DOCX / TXT -> extract text -> overlapping chunks -> Ollama embeddings
                                                               |
Question -> embedding -> retrieve relevant chunks from ChromaDB |
                       -> local LLM -> streamed answer + sources
```

- PDF text is extracted per page; DOCX text is divided into numbered sections.
- Text is split into 500-word chunks with 50-word overlap.
- Ollama generates embeddings; ChromaDB stores vectors and source metadata locally.
- The three closest chunks are sent to a local language model with citation instructions.
- The original web interface streams the response and lists retrieved references.

PDF references use physical page numbers. DOCX/TXT references use **sections**, not page numbers. Model-generated quotations and answers still need checking against the documents.

## Public repository scope

This is a standalone version of the school prototype. **Company documents, the populated vector database, old Git history and the school API/server address are not included.** The default model endpoint is your own local Ollama server. No school service or API key is required.

The repository includes an explicitly **fictional** demo manual. It is not TOBROCO-GIANT guidance and contains no real engineering specifications. Add your own authorised documents to use the assistant on real material. The TOBROCO-GIANT name and original logo identify the student project; they do not imply an official product or endorsement.

## Run locally

Use **Python 3.12** and install [Ollama](https://ollama.com/). Start Ollama and download the models:

```sh
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

The 3B model is a smaller default for local experimentation, not a claim to reproduce the original project's answer quality. Model downloads require internet access and disk space; inference runs against the configured local service.

```sh
python -m venv .venv
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

```sh
python -m pip install -r requirements.txt
python assistant.py examples
python app.py
```

Open **http://localhost:5001** and ask: **What should a reviewer record before releasing a drawing?**

To use your own material, place PDF, DOCX or TXT files in the ignored `documents/` folder, then run:

```sh
python assistant.py documents
```

Re-indexing the same filename replaces that document's chunks. Files removed from the folder are not automatically removed from the index. Use a new local database directory to start with an empty index. Scanned PDFs need OCR first; DOCX table extraction and OCR are not implemented.

## Configuration

Set environment variables in your shell before indexing and starting the server. `.env.example` lists the defaults; it is a reference, not an automatically loaded configuration file.

| Variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `LLM_MODEL` | `llama3.2:3b` |
| `EMBED_MODEL` | `mxbai-embed-large` |
| `PORT` | `5001` |

Changing the embedding model requires rebuilding the index in a fresh local `chroma_db/` directory. Do not mix embedding models in the same collection. This local prototype has no login system and should not be exposed directly to the internet.

## Files

```text
app.py                         Flask server and streaming API
assistant.py                   Extraction, indexing, retrieval and generation
tobroco_assistant.html         Original project interface, with streaming fixes
logo.png                       Original project branding
examples/demo-manual.txt        Fictional demonstration text
documents/                     Your own documents (ignored by Git)
notebooks/original_prototype.ipynb   Reference notebook, not the app entry point
tests/                         Pipeline and API tests
```

The reference notebook has its outputs removed, its school endpoint replaced with localhost and an unused broken image-extraction block removed. Use the Python app above for the supported setup.

## Validation

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests exercise chunking, document extraction, ChromaDB indexing/retrieval and streamed API responses with a deterministic test model client. They do not establish the quality of a live LLM's answers. A full live generation check requires Ollama and both downloaded models.
