# Ultimate ML

Ultimate ML is a local-first machine-learning knowledge base and spaced-repetition app. It combines quick recall with rich concept pages, mathematical foundations, manually managed architecture/pipeline assets, and optional AI-assisted authoring.

The core workflow always works offline after content is saved: browse a topic, answer a question, reveal the direct answer and a collapsible deep concept review, rate Again/Hard/Good/Easy, and let SQLite schedule the next review.

## Highlights

- **Learn deeply, then retain it.** Rich ML concept pages pair intuition, practical context, source links, architecture/pipeline assets, and LaTeX mathematics with conceptual spaced-repetition questions.
- **Local-first by design.** Content is portable JSON; learning history, drafts, settings, usage, and revisions stay in local SQLite. Saved learning works without an API key or network connection.
- **AI assistance with human control.** Structured topic generation, quality review, relationship resolution, and question drafting all end in editable drafts. Nothing becomes durable content or enters review automatically.
- **A real quality gate, not just generated JSON.** The pipeline checks taxonomy, concept type, technical claims, equations, source grounding, provenance, sparse relationships, and mathematical prerequisites before a human approves a topic.
- **Transcript-first video learning.** Turn an accessible YouTube transcript or pasted transcript into ranked ML concepts, explicit create/enrich/ignore decisions, reviewed topics, and eventually spaced repetition—without storing full third-party transcripts in the public knowledge base.
- **Privacy- and budget-aware AI.** OpenAI calls are backend-only, tests use fakes, API usage is recorded locally, and a configurable **$5/month** local hard guardrail estimates and blocks overspend before paid calls.
- **Safe local editing and recovery.** Topic/question editors preserve IDs and references, create revisions before durable changes, support restore/retry paths, and never spend AI budget for local edits or approvals.

### The authoring loop

```text
Idea, transcript, or existing draft
→ structured AI draft
→ deterministic checks + optional AI quality review
→ edit and inspect provenance/relationships/math
→ explicit approval
→ conceptual questions
→ spaced repetition
```

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
- `backend/youtube/` — isolated transcript-provider adapter and local transcript-cache service
- `frontend/` — framework-free local web client, plus a locally vendored MathJax distribution for offline LaTeX rendering
- `content/` — portable JSON topics/questions; approved AI content is saved here like manually authored content
- `assets/` — local architecture/pipeline figures with source and licence metadata
- `data/ultimate_ml.db` — local progress, settings, AI drafts/import metadata, and usage data; never commit it
- `data/youtube_cache/` — gitignored full transcript cache; never public knowledge-base content by default

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
2. Open **Create topic**, enter a title (for example, Gaussian Mixture Models), category, learning difficulty, explanation depth, and focus. Difficulty and explanation depth are intentionally separate.
3. Review the displayed maximum-cost estimate, then generate a structured draft.
4. Edit the draft, save it locally, optionally regenerate just one section, and explicitly approve it.
5. Approval validates and writes a portable topic JSON file. It never overwrites an existing topic automatically.
6. Generate conceptual question candidates, edit/select them, then explicitly approve selected questions. Only then do they enter Daily Review.

Generated drafts are kept in SQLite with `draft`, `approved`, or `discarded` state. A browser refresh does not lose them. Full explanations remain only on the topic; approved questions keep concise direct/expanded answers and link to the topic for deep review.

## Local corrections and question recovery

Approved topics have an **Edit Topic** action. The normal editor uses the same structured fields as draft review; **Advanced JSON Editor** loads the current durable JSON for precise repairs. Both validate before saving, preserve the topic ID and question references, create a local prior-version backup in `data/topic_revisions/`, and atomically replace the content file. A saved revision can be restored from the editor. These operations make **zero OpenAI calls**.

Each topic also has **Question Management**: approved-question count, recoverable draft batches, local approved-question editing, and an archive action that removes a question from active review without deleting its history. Generated candidates persist in SQLite until discarded or approved.

Approving existing question candidates is a local, idempotent operation. It validates all selected candidates first, removes unavailable generated relationship IDs with a stored warning, writes only missing questions, recognizes already-approved matching questions, and returns per-item approved/already-approved/failed results. The saved state supports retry/reload recovery; approval never generates questions or spends AI budget. Local backups of question edits and reconciliation are in `data/question_revisions/`.

## Transcript-first YouTube learning

**Import YouTube** supports one local workflow: transcript → concise ML concepts → explicit Create new / Enrich existing / Ignore decision → normal Ultimate Topic draft review → conceptual question draft review → spaced repetition. It deliberately does not save a generic video summary as a topic.

1. Paste a standard YouTube URL to retrieve an accessible transcript, or paste a transcript manually. The manual path always works when captions are unavailable or access is restricted.
2. The complete transcript is stored only in `data/youtube_cache/`, which is ignored by Git. The database stores metadata, a short preview, analysis state, and local usage data—not a public-content transcript copy.
3. Review concise concept cards. Source-grounded evidence and approximate timestamps are visibly separate from AI-generated learning rationale. Exact title/ID matching is deterministic; the server validates every selected existing-topic ID. Extraction v2 can keep meaningful named methods/models (for example SimCLR, CLIP, DINO, or InfoNCE) beside their broader concepts, while filtering passing mentions.
4. Select several unmatched concepts as **Create new** and exact matches as **Enrich existing**. The selection stays on the video page and shows a maximum batch estimate before any model call. The batch is deliberately executed as conservative, sequential, individual structured requests—not one giant request.
5. Existing active work for the same video and normalized concept is shown as **Draft exists** / **Review draft** and is reused without another model call or charge. Nothing is created automatically and duplicate-topic creation is blocked.
6. The persistent **Draft Queue** survives restarts and shows pending, generating, ready, failed, approved, and discarded work. Successful drafts remain if a later task fails; retry addresses only the failed item. You can stop after the current request and reopen the import at any time.
7. Expansion calls the same V2 Ultimate Topic Generator and opens the normal structured review editor. A video-derived draft includes source/provenance context and a link back to its video analysis. An enrichment preserves existing architecture/pipeline asset fields. Approval is still required before durable JSON changes.
8. After approval, generate conceptual video-derived question candidates only when you choose. Batch topic generation never auto-generates questions.

Transcript retrieval uses `youtube-transcript-api` behind `backend/youtube/transcript_provider.py`. It only requests publicly accessible captions through the provider; it does not authenticate, bypass restrictions, scrape frames, or work around unavailable transcripts. If retrieval fails, paste the transcript instead.

Concept extraction, video topic expansion, and video question generation are all tracked as their own local AI usage operations. Before extraction or a selected draft batch, the UI shows a conservative maximum cost and remaining local budget; afterward it shows estimated actual local cost for the import. Re-analysis updates only the analysis result and records a new usage event—it never deletes drafts, approved content, or prior usage. These calls use the same backend-only key and $5 monthly hard guardrail as ordinary authoring.

## Scope

Implemented: local topic browsing/search, ResNet and math-rich topic pages, related/prerequisite relationships, source and image support, local LaTeX rendering, spaced repetition and progress, AI settings, backend-only structured topic/question authoring, local approved-topic/question editing with revisions, idempotent local question approval/recovery, section regeneration, local draft persistence, usage estimation, and hard monthly budget blocking; plus transcript-first YouTube import, v2 named-method-aware concept extraction, deterministic existing-topic matching, reviewed create/enrich topic generation, persistent draft queues, batch-safe sequential generation, duplicate-draft reuse, import reopening, and video-specific usage tracking.

Not implemented by design: video-frame extraction, OCR, multimodal video analysis, paper/PDF lookup, automatic source verification, web image search, AI answer grading, embeddings, cloud deployment, multi-user accounts, or an interactive knowledge graph.

## AI generation quality contract

New AI topic drafts use a quality gate, not just a JSON-schema check:

```text
request/source → taxonomy + concept type + catalog → structured generation
→ deterministic normalization → structured AI quality-review/repair
→ deterministic lint → human review → explicit approval
```

The backend owns durable IDs. Models never receive or author them; a new durable ID is derived server-side at approval, an enrichment uses its explicit existing target, and approved-topic editing preserves the current ID. Draft-only backend identity is visible as a preview and is not an editable durable ID.

The generator and reviewer receive the local taxonomy, including definitions, positive examples, and important negative examples. They classify one of `broad_concept`, `named_method`, `architecture`, `loss_or_objective`, `mathematical_concept`, `training_mechanism`, or `evaluation_concept`, so named methods, mathematical ideas, objectives, and architectures are not authored with one generic template. For example, Knowledge Distillation is a deep-learning training mechanism, while CLIP and Contrastive Learning are representation-learning methods/objectives rather than `ml_fundamentals`.

Durable prerequisite and related-topic edges are deliberately strict: a prerequisite must materially block understanding, and a related edge must be educationally strong. A supplied catalog ID also needs an explicit rationale. Empty lists are preferred to weak CNN/ResNet/gradient-flow/optimization links; missing but useful neighbours remain `suggested_new_topic_relationships` rather than substituted IDs.

The reviewer repairs a complete structured topic and reports its fixed issues, remaining blocking issues, warnings, and confidence. It checks technical claims, taxonomy, difficulty, tags, relation semantics, equations and notation, assumptions/scaling/gradient direction, named-method completeness, source grounding, provenance, and overclaims. Local lint then rejects bad category/IDs/self or duplicate relations, malformed or unexplained equations, duplicate suggestions, noncanonical YouTube URLs, and model-supplied IDs. If a blocking issue remains, the persisted draft is labelled **Needs attention**, never **Ready**; no draft is auto-approved.

For video-derived topics the source URL is canonicalized locally to `https://www.youtube.com/watch?v=VIDEO_ID`. Evidence timestamps remain only in `source_provenance.source_derived.timestamp_seconds`; source-derived evidence stays separate from the AI-expanded educational explanation, and full transcripts remain in gitignored local cache.

Before a topic or video-topic expansion, the UI shows separate conservative maximum estimates for generation and quality review plus their total and remaining budget. The two calls are recorded as separate local usage events; the existing $5 monthly application hard guard checks the combined maximum before either paid call. Tests use fake providers only and include regression fixtures for Knowledge Distillation, CLIP, Contrastive Learning, malformed math, strict relationships, canonical YouTube provenance, and unresolved review warnings.

## Existing draft quality review

Earlier paid AI topic drafts are never regenerated or automatically modified when the quality gate improves. Legacy drafts show **Generation: Complete** and **Quality review: Not run** until you explicitly choose **Run quality review** from the draft screen or Draft Queue.

That action sends the existing full draft directly to the current structured reviewer; it never calls the topic generator and never generates, deletes, approves, or otherwise changes questions, review history, imports, approved content, or progress. The preflight shows the original already-paid generation estimate, the separate review maximum, the current reviewer version, and remaining local budget. Existing-draft reviews are recorded separately as `topic_quality_review_existing`.

Before a repair is applied, Ultimate ML stores an exact `pre_quality_review` snapshot in local SQLite. It then stores the repaired topic and report as a separate `quality_review` revision. Both can be viewed and restored from the draft screen. A legacy draft is only changed after its optional reviewer call succeeds; failures retain the original content and are marked **Review failed**. Review states are `not_reviewed`, `reviewing`, `reviewed`, `needs_attention`, and `review_failed`; unresolved blockers produce **Needs attention**, never automatic approval.

Duplicate-spend protection hashes the reviewed draft content together with the reviewer prompt version. Reopening or normally running review again for the same repaired revision reuses the saved result with no API call. **Run explicit re-review** is a distinct user action with a new estimate and one new reviewer call. No batch review is performed automatically; each opt-in review is one draft, one structured reviewer call.

## Topic relationships contract

Ultimate ML preserves both `prerequisite_topic_ids` and `related_topic_ids` as a sparse curriculum graph—not a similarity graph. Empty arrays are valid and often preferred.

- A **prerequisite** is something a learner would be materially blocked or substantially confused without first. Training with an optimizer, seeing a term in an equation, or useful general background is not enough.
- A **related topic** is a strong direct neighbor, extension, close alternative, important comparison, or central architecture/objective pairing. Generic implementation adjacency is not enough.

The soft targets are 0–3 prerequisites and 0–4 related topics, never quotas. Each proposed durable edge needs a rationale and `high`, `medium`, or `low` confidence. Only high-confidence edges remain durable; medium-confidence edges become suggestions/warnings and low-confidence edges are removed. Missing important concepts remain `suggested_new_topic_relationships`, never substitute weak catalog IDs.

The generator/reviewer and deterministic lint reject generic graph heuristics such as Gradient Descent from ordinary training, CNN from neural-network use, ResNet/Gradient Flow from residual paths, PCA/Covariance from embeddings, and Knowledge Distillation from teacher/student language. Rules are concept-type-aware: architectures prioritize core architectural prerequisites and canonical comparisons; named methods their parent/direct objective; losses real mathematical dependencies; mathematical concepts true dependency chains.

Lint checks invalid IDs, duplicates, self-links, cross-list duplicates, sparsity, and high-confidence rationales. Clearly weak durable relationships are quality-review blocking issues and must be repaired before a draft is Ready. The structured editor explains the strict meanings and shows stored AI rationale/confidence for saved edges. Approved topics are never rewritten automatically; existing drafts gain these rules only when explicitly reviewed or re-reviewed.

## Quality-review calibration

Source-derived evidence and AI-expanded educational explanation remain deliberately separate. A short or limited YouTube segment is a **warning** when the source-derived summary is accurate and the broader material is explicitly labelled AI-expanded; it does not by itself produce **Needs attention**. Blocking source-grounding issues are unsupported claims presented as source-derived, contradictory timestamps/source identity, or missing/misleading provenance.

`Needs attention` is reserved for unresolved technical errors, broken mathematics, invalid taxonomy, misleading attribution, weak durable relationships, missing method-defining content, or contradictory/malformed content. Useful secondary canonical details are warnings. For example, an otherwise correct SimCLR NT-Xent explanation that omits explicit symmetric i→j/j→i wording receives a completeness warning rather than a blocker.

`mathematical_foundation.prerequisites` is independent from the durable topic graph. It contains only tools needed to follow equations. During future generation or an explicit review, generic items such as `cnn` and `gradient-descent` are removed from SimCLR’s mathematical prerequisites and replaced with precise concepts: vectors/dot products, cosine similarity, softmax, cross-entropy, and contrastive-learning intuition. Equivalent policies cover CLIP and Knowledge Distillation. This calibration never automatically changes approved topics, including SimCLR.

## Scalable metadata resolution

Educational authoring and metadata resolution are now separate. Generation and quality review receive only the small taxonomy registry and write the explanatory topic content; they do not receive the full approved-topic catalog or choose durable catalog IDs. The final normal flow is:

```text
Generate educational draft → quality review/repair → local candidate retrieval
→ one structured relationship resolver → deterministic guardrails → human review → approval
```

The taxonomy registry is intentionally small and category-based. Each category records a definition, positive and negative examples, and compatible concept types. It is not a table of topic-specific answers. The approved library stores a compact retrieval document for each topic (`topic_id`, title, one-sentence summary, concept type, category, and selected tags). A dependency-free local TF-IDF/cosine retrieval fallback ranks candidates, so no new embedding model, vector database, or network service is required. `ULTIMATE_ML_METADATA_TOP_K` controls the candidate cap (default 15, bounded to 1–20).

Only those top-K compact candidates are sent in **one** structured relationship-resolution call. The resolver returns prerequisite/related candidates with rationale and confidence plus rejected candidates. High-confidence decisions are durable edges; medium-confidence decisions become suggestions; low confidence is discarded. Retrieval similarity is never itself a relationship. No high-confidence decision cleanly yields `[]` / `[]` and remains approval-ready.

Local guardrails are intentionally universal rather than title-specific: no self, duplicate, overlapping, or nonexistent IDs; sparse maximums; high-confidence rationales; and rejection of stock implementation adjacency such as optimizer→prerequisite, optional CNN/ResNet backbones, residual→gradient-flow, embedding→PCA/covariance, or teacher/student→distillation. The prior title-specific relationship and math-policy table has been retired.

Mathematical prerequisites are separate from the topic graph and use a controlled vocabulary inferred from the equations and their explanations. Architecture/training labels cannot enter this list. Broad topics can record section-level prerequisites, so specialized mathematics (for example covariance) does not become global merely because one subsection uses it. Before a draft becomes Ready, the backend checks that the resolver audit, category/concept type, saved edges/rationales, and controlled math prerequisites agree with the persisted payload; a mismatch is **Needs attention**.

The explicit **Rebuild relationships for topic** action (also available for an active draft) runs only local retrieval, the compact resolver if needed, and deterministic filters. It never regenerates educational prose, questions, source material, or IDs. Approved topics get a local revision backup before the metadata-only save. Results are cached by compact current-topic hash, retrieved candidate hashes, and resolver prompt version; an unchanged rebuild reuses the cached result at $0. Existing topics and paid drafts are never changed automatically.

Generation/quality-review estimates now include the resolver maximum as a separately tracked local usage operation (`metadata_relationship_resolution`). Local retrieval, linting, math normalization, and cached/no-candidate resolutions cost $0. Automated tests use fake providers only.

## Draft action reliability

Draft Queue actions are single-flight local mutations: **Save edits**, **Rebuild relationships**, and **Validate and approve** cannot overlap for one draft. Each action has explicit idle/loading/success/error state and always releases its controls in a `finally` path. The browser applies a 45-second `AbortController` timeout; timeout, network, malformed-response, and backend errors remain visible beside the draft instead of leaving a spinner running.

Errors include the HTTP status, backend message, and available validation field/area. A successful save replaces the browser’s draft state with the returned persisted revision. Rebuild returns the persisted draft as well, including its cache/cost result, and the screen reloads from that response before another action is allowed. Approval saves first, then validates and approves that same persisted revision.

Resolver audit checks remain strict but relationship IDs are canonicalized as sets for comparison, so selector ordering alone cannot cause a mismatch. A rebuild writes `prerequisite_topic_ids`, `related_topic_ids`, their high-confidence justifications, and `metadata_resolution` together in the same SQLite draft payload; it never merges old pre-review relationship fields back afterward. A genuinely changed manual relationship selection remains visible as needing a rebuild rather than silently changing the resolver audit.

## Advanced JSON editor state synchronization

Draft review has one canonical browser payload. The structured editor and **Advanced JSON editor** are views of that same object, not independent draft copies. **Apply advanced JSON** parses and performs a practical local schema check before replacing that canonical payload, immediately re-renders the structured fields (including mathematical prerequisites), marks the draft dirty, and visibly confirms that the JSON is applied locally. It does not make an API call until **Save edits**.

Save serializes that canonical payload directly. The returned persisted draft revision then replaces the canonical payload, refreshes both editing surfaces, records its revision metadata, and clears the dirty state. **Validate & approve** auto-saves a dirty payload first, then validates and approves that exact persisted revision; it cannot validate an earlier view of the form.

Resolver-owned category, relationship, and mathematical-prerequisite metadata remains strict. An edit that differs from the resolver audit is visibly marked stale and must be rebuilt before approval; restoring the exact audited values, such as DINO’s global `Gradients` prerequisite, clears the stale state. JSON parse/schema errors and backend validation errors remain visible locally, and no editor, save, validation, or approval action invokes OpenAI.
