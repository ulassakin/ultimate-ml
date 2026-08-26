/* Explicit-only metadata maintenance. It never regenerates educational content. */
const metadataResolutionDraftReview = draftReview;
draftReview = async function(id) {
  await metadataResolutionDraftReview(id);
  const heading = app.querySelector('h1');
  if (!heading) return;
  const card = document.createElement('section');
  card.className = 'card';
  card.innerHTML = '<h2>Relationship metadata</h2><p class="muted">Rebuild uses local retrieval and one compact resolver call at most. It changes only sparse relationship metadata; it never regenerates educational content or questions.</p><p class="notice" id="metadata-estimate">Checking local cache…</p><div class="actions"><button class="secondary" id="rebuild-metadata">Rebuild relationships</button></div>';
  heading.after(card);
  const estimateNode = card.querySelector('#metadata-estimate'), button = card.querySelector('#rebuild-metadata');
  try {
    const estimate = await api('/ai/drafts/' + id + '/metadata-rebuild-estimate');
    estimateNode.textContent = estimate.cached ? 'A matching local resolution is cached; rebuild costs $0.00.' : `Resolver maximum: ${money(estimate.maximum_estimated_cost_usd)} · ${estimate.candidate_count} retrieved candidates · remaining: ${money(estimate.remaining_budget_usd)}.`;
    button.onclick = async () => { if (!window.confirm(estimate.cached ? 'Reuse the cached relationship resolution? This does not call OpenAI.' : `Resolve relationships from ${estimate.candidate_count} compact candidates? Maximum cost: ${money(estimate.maximum_estimated_cost_usd)}.`)) return; busy(button, 'Rebuilding…'); try { await request('/ai/drafts/' + id + '/rebuild-relationships', 'POST'); location.hash = '#/drafts/' + id; } catch (error) { button.disabled = false; button.classList.remove('loading'); button.textContent = error.message; } };
  } catch (error) { estimateNode.textContent = error.message; button.disabled = true; }
};

const metadataResolutionTopicDetails = topicDetails;
topicDetails = async function(id) {
  await metadataResolutionTopicDetails(id);
  const article = app.querySelector('article');
  if (!article) return;
  const card = document.createElement('section');
  card.className = 'section card';
  card.innerHTML = '<h2>Relationship metadata</h2><p class="muted">This explicit maintenance action preserves the topic ID and educational content, creates a local revision, and never regenerates or approves anything.</p><p class="notice" id="topic-metadata-estimate">Checking local cache…</p><button class="secondary" id="rebuild-topic-metadata">Rebuild relationships for topic</button>';
  article.append(card);
  const estimateNode = card.querySelector('#topic-metadata-estimate'), button = card.querySelector('#rebuild-topic-metadata');
  try {
    const estimate = await api('/topics/' + id + '/metadata-rebuild-estimate');
    estimateNode.textContent = estimate.cached ? 'A matching local resolution is cached; rebuild costs $0.00.' : `Resolver maximum: ${money(estimate.maximum_estimated_cost_usd)} · ${estimate.candidate_count} retrieved candidates.`;
    button.onclick = async () => { if (!window.confirm(estimate.cached ? 'Reuse the cached relationship resolution?' : `Rebuild sparse relationships? Maximum cost: ${money(estimate.maximum_estimated_cost_usd)}.`)) return; busy(button, 'Rebuilding…'); try { await request('/topics/' + id + '/rebuild-relationships', 'POST'); location.hash = '#/topics/' + id; } catch (error) { button.disabled = false; button.classList.remove('loading'); button.textContent = error.message; } };
  } catch (error) { estimateNode.textContent = error.message; button.disabled = true; }
};
