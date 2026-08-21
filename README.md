# Ultimate ML

Ultimate ML is a local-first machine-learning knowledge base and spaced-repetition app. It combines quick recall with rich concept pages, mathematical foundations, manually managed architecture/pipeline assets, and optional AI-assisted authoring.

The core workflow always works offline after content is saved: browse a topic, answer a question, reveal the direct answer and a collapsible deep concept review, rate Again/Hard/Good/Easy, and let SQLite schedule the next review.

## Run locally

Python 3.10+ is required.

```bash
make install
make validate
make test
make dev
```

Open <http://127.0.0.1:8000>. Learning history, local AI settings, drafts, usage events, and the SQLite database live under `data/` and are ignored by Git.

## Architecture

- `backend/` — FastAPI, SQLite migrations, scheduler, content validation, and backend-only AI integration
- `backend/ai/` — provider abstraction, Responses API provider, structured schemas, versioned prompts, pricing, budget, and usage logic
- `frontend/` — framework-free local web client, plus a locally vendored MathJax distribution for offline LaTeX rendering
- `content/` — portable JSON topics/questions; approved AI content is saved here like manually authored content
- `assets/` — local architecture/pipeline figures with source and licence metadata
- `data/ultimate_ml.db` — local progress, settings, AI draft, and usage data; never commit it

The frontend remains deliberately small. Phase 2 adds authoring and review screens incrementally; the existing API boundaries make a future React migration possible without rewriting the backend or content library.

## Mathematical Foundations

`mathematical_foundations` is a first-class category. Topic V2 supports `intuition`, `big_picture`, `mathematical_foundation`, explicit prerequisite topic IDs, and equation objects containing both LaTeX and a plain-language explanation. The included Covariance and PCA examples demonstrate the format.

Math questions should test meaning and ML relevance rather than arithmetic drills. Full mathematics belongs on a topic and questions point back to it through `concept_refresher_topic_id`.

## Content and assets

Add topics and questions as JSON under `content/topics/` and `content/questions/`. The validator checks duplicate IDs, difficulties, topic/prerequisite references, required question refresher references, image paths, equation structure, source structure, and AI review metadata.

```bash
make validate
```

Keep personal or redistribution-uncertain figures outside version control (for example in a gitignored `assets/local/` directory). Do not assume paper figures can be redistributed. The included ResNet residual-block SVG is original and marked CC0.

## Optional OpenAI authoring

The app does not need OpenAI to run. AI controls remain unavailable if there is no key, while all saved learning features stay available.

To enable authoring locally:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` only in your local `.env`; never paste a key into the browser, source code, JSON content, or a chat. `.env` is ignored by Git. API billing is separate from any ChatGPT subscription. Also configure project-level OpenAI spend controls and alerts—the application’s own guardrail is useful but OpenAI Platform billing is authoritative.

`OPENAI_MODEL` documents the default model. The active model, explanation depth, enable/disable switch, monthly budget, and optional pricing override are stored locally through **Settings**. Default pricing for `gpt-5.4-mini` is centralized in `backend/ai/pricing.py`; update it or use a local override as pricing changes. The app labels every calculation as an estimate.

All model calls happen only in `backend/ai/openai_provider.py`, via the official OpenAI SDK and Responses API with JSON Schema structured output. No endpoint returns the key—only whether a key is configured. Automated tests use fakes and make zero network or paid API calls.

## $5 monthly local budget

The default local budget is **$5 USD per calendar month**. Before a topic draft, question draft, or section regeneration, Ultimate ML estimates a conservative maximum using centralized model pricing and configured token caps. It reserves that amount in SQLite and blocks the request if the current month’s estimated usage plus the reservation would exceed the budget. When a request succeeds, it reconciles the reservation with reported input/output/cached-token usage; failed requests are retained as failure events with no estimated cost.

View local estimates by operation and model in **Settings**. Changing the budget requires an explicit settings action. When the limit is exhausted, authoring stops but browsing, review, progress, and all saved content continue to work.

## Manual AI topic-generation flow

1. Start the app and open **Settings**; confirm a key is detected and keep or intentionally change the $5 budget.
2. Open **Create topic**, enter a title (for example, Gaussian Mixture Models), category, focus, and depth.
3. Review the displayed maximum-cost estimate, then generate a structured draft.
4. Edit the draft, save it locally, optionally regenerate just one section, and explicitly approve it.
5. Approval validates and writes a portable topic JSON file. It never overwrites an existing topic automatically.
6. Generate conceptual question candidates, edit/select them, then explicitly approve selected questions. Only then do they enter Daily Review.

Generated drafts are kept in SQLite with `draft`, `approved`, or `discarded` state. A browser refresh does not lose them. Full explanations remain only on the topic; approved questions keep concise direct/expanded answers and link to the topic for deep review.

## Scope

Implemented: local topic browsing/search, ResNet and math-rich topic pages, related/prerequisite relationships, source and image support, local LaTeX rendering, spaced repetition and progress, AI settings, backend-only structured topic/question authoring, review/approval, section regeneration, local draft persistence, usage estimation, and hard monthly budget blocking.

Not implemented by design: YouTube, paper/PDF, transcript, multimodal/video-frame, automatic source verification, web image search, AI answer grading, embeddings, cloud deployment, multi-user accounts, or an interactive knowledge graph.
