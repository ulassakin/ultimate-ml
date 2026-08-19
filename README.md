# Ultimate ML

A local-first machine-learning knowledge base and spaced-repetition review app. The first milestone is a complete learning loop: browse a ResNet concept page, answer a question, reveal a concept refresher, self-rate, persist the review in SQLite, and see progress update.

## Run locally

Python 3.10+ is required.

```bash
make install
make validate
make test
make dev
```

Open <http://127.0.0.1:8000>. Review history is stored locally in `data/ultimate_ml.db` and ignored by Git.

## Structure

- `backend/` — FastAPI API, SQLite persistence, content validation, scheduler
- `frontend/` — lightweight web client served by FastAPI
- `content/` — editable JSON topics and questions; no educational content is hardcoded in application logic
- `assets/` — locally managed diagrams and their source metadata
- `scripts/validate_content.py` — consistency checker
- `tests/` — content, scheduling, persistence, and API vertical-slice tests

The API is isolated under `/api`, so a React/Next.js client can replace the deliberately small first-milestone frontend without changing content or persistence logic.

## Add content

Copy an existing JSON file in `content/topics/` or `content/questions/`, give it a unique ID, and run `make validate`. Validation rejects malformed JSON, duplicate IDs, invalid difficulties, missing topic references, and missing image paths.

## Current scope

Implemented: topic browsing/search, ResNet page and original architecture diagram, related topics and sources, due-question review, answer reveal and refresher, Again/Hard/Good/Easy scheduling, review history, dashboard totals, and simple weak-topic detection.

Not implemented by design: AI grading/generation, YouTube or paper ingestion, multimodal extraction, knowledge graphs, generated diagrams, and cloud services.

Externally sourced images should only be added after checking redistribution rights. The included residual-block diagram is original and marked CC0.

