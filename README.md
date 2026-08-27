# Ultimate ML

Ultimate ML is a local-first machine-learning knowledge base and spaced-repetition app. It combines rich ML topic pages, intuition-first mathematical explanations, optional AI-assisted authoring, and conceptual questions scheduled locally for long-term retention.

Saved learning works offline: browse a topic, answer a question, reveal its answer and deep concept review, rate Again/Hard/Good/Easy, and let SQLite schedule the next review.

## Highlights

- **Learn deeply, then retain it.** Topic pages pair quick recall, intuition, practical ML context, sources, architecture/pipeline assets, and LaTeX mathematics with conceptual SRS questions.
- **Local-first by design.** Portable JSON stores approved topics/questions; SQLite stores drafts, learning history, settings, AI usage, and revisions locally.
- **Human-controlled AI assistance.** Structured authoring, explicit quality review, and question generation always produce editable drafts. Nothing is auto-approved.
- **A focused quality gate.** The reviewer checks technical and conceptual correctness, taxonomy, mathematical correctness, clarity, examples, misconceptions, limitations, ML relevance, source/provenance, and named-method completeness.
- **Transcript-first video learning.** An accessible YouTube transcript or pasted transcript becomes ranked ML concepts, human decisions, reviewed topics, questions, and SRS—without publishing full third-party transcripts.
- **Privacy- and budget-aware AI.** OpenAI calls are backend-only; tests use fakes; usage stays local; and a configurable **$5/month** application-side hard guardrail estimates and blocks overspend.
- **Safe local editing and recovery.** Topic/question editors preserve stable IDs and review history, create backups before durable edits, and make no OpenAI call for local edits or approval.

### The authoring loop

```text
Idea, transcript, or existing draft
→ structured AI authoring draft
→ optional AI quality review/repair
→ human edit and provenance inspection
→ explicit approval
→ conceptual questions
→ spaced repetition
```

Ultimate ML intentionally does **not** maintain a prerequisite/related-topic knowledge graph. The product focuses on high-quality learning content and reliable review rather than graph inference.

## Run locally

Python 3.10+ is required.

```bash
make install
make validate
make test
make dev
```

Open <http://127.0.0.1:8000>. Local drafts, settings, usage events, review history, and the SQLite database live under `data/` and are ignored by Git.

## Architecture

- `backend/` — FastAPI, SQLite, content validation, SRS scheduler, and backend-only AI integration
- `backend/ai/` — provider abstraction, structured schemas, versioned prompts, pricing, budget, and local usage tracking
- `backend/youtube/` — isolated transcript-provider adapter and local transcript-cache service
- `frontend/` — framework-free local web client with vendored MathJax for offline LaTeX rendering
- `content/` — portable approved topic/question JSON
- `assets/` — local architecture/pipeline figures with source and licence metadata
- `data/ultimate_ml.db` — local progress, settings, draft/import metadata, and usage data; never commit it
- `data/youtube_cache/` — gitignored full transcript cache; never public knowledge-base content by default

## Mathematical Foundations

`mathematical_foundations` is a first-class category. Topic V2 supports intuition, big-picture explanations, equation objects with LaTeX plus plain-language explanations, and deep mathematical sections.

Optional `mathematical_foundation.prerequisites` values are display-only background hints. They never create topic edges, never invoke a provider, and never block approval; empty lists are valid.

Math questions should test meaning, intuition, and ML relevance rather than arithmetic drills. Full explanations stay on a topic; questions link back through `concept_refresher_topic_id`.

## Content and assets

Add topics and questions as JSON under `content/topics/` and `content/questions/`.

```bash
make validate
```

The validator checks duplicate IDs, difficulty values, required question refresher references, image paths, equation structure, source structure, and AI review metadata. Legacy relationship fields in older content are tolerated but ignored by the product.

Keep personal or redistribution-uncertain figures outside version control (for example in `assets/local/`).

## Optional OpenAI authoring

The app does not need OpenAI to run. Without a key, all saved learning, editing, SRS, progress, and local content features continue to work.

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` only in your local `.env`; never put it in the browser, source code, JSON content, or Git. API billing is separate from a ChatGPT subscription. Configure OpenAI Platform spend controls and alerts as well—the local guardrail is helpful but Platform billing is authoritative.

All model calls happen only in `backend/ai/openai_provider.py` using the official OpenAI SDK and Responses API structured output. No endpoint returns the key; Settings only reports whether one is configured. Automated tests use fakes and make zero network or paid API calls.

## $5 monthly local budget

The default local budget is **$5 USD per calendar month**. Before an authoring, quality-review, question-generation, or section-regeneration call, Ultimate ML estimates a conservative maximum from centralized pricing/token caps and blocks the call if the local month would exceed the budget. Successful calls reconcile reported usage; failures remain locally recorded as failures.

The two topic-authoring costs are deliberately separate:

- **Generate Draft** reserves and makes exactly one `topic_draft` authoring call.
- **Run Quality Review** reserves and makes exactly one quality-review call (`topic_quality_review_existing` or its YouTube equivalent).

There is no relationship retrieval, resolver call, or hidden metadata cost. Local edits, review approval, question approval, and SRS never spend AI budget.

## Manual AI topic flow

1. Start the app and open **Settings**. Confirm a key is detected and keep or intentionally change the $5 budget.
2. Open **Create topic**, provide title, category, difficulty, depth, and focus.
3. Review the authoring-only estimate and choose **Generate draft**. The new draft shows **Quality review: Not run**.
4. Open the draft or Draft Queue and inspect/edit the content. You may choose **Run Quality Review** first; it sends the existing full draft to the reviewer and never regenerates the topic or touches questions. A local risk hint recommends review for math-heavy, source-derived, or technically complex topics.
5. Choose **Approve** after schema, taxonomy, and deterministic validation pass. Review is optional by default: unreviewed or failed-review drafts can be approved after a one-time confirmation, while a completed review with real unresolved blockers still prevents approval. The backend owns the durable topic ID and writes portable content JSON.
6. Generate conceptual question candidates only when desired, then review and locally approve selected questions into Daily Review.

## Existing draft quality review

Earlier paid AI drafts are never regenerated or automatically changed when the quality gate improves. Active legacy drafts show **Quality review: Not run** until the user explicitly runs review.

An existing-draft review stores a pre-review snapshot and a separate repaired revision in local SQLite. It is idempotent by payload hash and reviewer prompt version: reopening/reusing the same reviewed revision does not spend again; **Run explicit re-review** is a separately estimated, deliberate call. Reviewing never creates, deletes, approves, or changes questions, imports, review history, or approved topics.

Quality Review is a focused correctness pass, not a second authoring pass. It preserves correct prose and makes the minimum material edits for technical, mathematical, taxonomy/schema, provenance, or misleading-content issues. The backend derives a structured field-level diff from the original and final payload, verifies model-reported changes and fixed-issue claims against that diff, removes stale report entries after normalization, and permits **Ready** only when the final payload and report agree. Each review response and local usage record reports its input tokens, output tokens, and estimated cost; no hidden resolver or relationship call is made.

## Local corrections and question recovery

Approved topics have **Edit Topic** and an **Advanced JSON Editor**. They parse/validate before atomic save, preserve the durable topic ID and question references, and create a local backup in `data/topic_revisions/`. These operations make zero OpenAI calls.

Question Management shows approved questions and recoverable candidate batches. Approving existing candidates is a local, idempotent operation: it validates candidates, writes only missing questions, recognizes already-approved matches, and returns per-item status. Local question edits/revisions preserve review history.

## Transcript-first YouTube learning

**Import YouTube** supports:

```text
transcript → concise ML concepts → Create new / Enrich existing / Ignore
→ normal topic draft → explicit quality review → questions → spaced repetition
```

1. Paste a standard YouTube URL to retrieve accessible captions, or paste a transcript manually. The manual fallback always works when captions are unavailable.
2. Full transcripts are stored only under gitignored `data/youtube_cache/`. The database keeps metadata, a short preview, analysis state, and local usage—not a public-content transcript copy.
3. Concept extraction is concise and source-grounded. It preserves meaningful named methods/models (for example SimCLR, CLIP, DINO, DINOv3, and InfoNCE) while filtering passing mentions.
4. Batch expansion uses conservative sequential, individual structured authoring calls. Existing active work for the same normalized video concept is reused rather than charged again.
5. Video-derived drafts use the same explicit quality-review/approval flow and keep source-derived evidence separate from AI-expanded educational explanation. Approval remains required.

Transcript retrieval uses `youtube-transcript-api` behind `backend/youtube/transcript_provider.py`. It requests only publicly accessible captions; it does not authenticate, bypass restrictions, scrape frames, or use multimodal analysis.

## Scope

Implemented: local browsing/search, math-rich topics, source and architecture/pipeline image support, offline LaTeX rendering, spaced repetition/progress, backend-only structured topic/question authoring, explicit quality review, local editing with revisions, idempotent local question approval/recovery, local usage estimates/budget blocking, and transcript-first YouTube import with draft queues and duplicate-draft reuse.

Not implemented by design: a prerequisite/related-topic graph, embeddings/vector search, video-frame extraction, OCR, multimodal video analysis, paper/PDF lookup, automatic source verification, web image search, AI answer grading, cloud deployment, multi-user accounts, or automatic approval.
