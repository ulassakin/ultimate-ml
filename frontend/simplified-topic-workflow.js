/* Final UI layer for the graph-free, explicit-review authoring workflow. */
const simpleTopicFields=['one_sentence_summary','quick_recall','big_picture','why_it_exists','intuition','core_explanation','mechanism','ml_relevance','practical_example','deep_dive'];
const simpleCategories=['mathematical_foundations','ml_fundamentals','classical_ml','deep_learning','computer_vision','transformers','representation_learning','object_detection','anomaly_detection','uncertainty_calibration','research_evaluation'];
const stripLegacyGraph=payload=>{const copy=structuredClone(payload);for(const field of ['prerequisite_topic_ids','related_topic_ids','relationship_justifications','suggested_new_topic_relationships','relationship_warnings','metadata_resolution'])delete copy[field];return copy;};
const categoryOptions=value=>simpleCategories.map(id=>`<option value="${id}" ${value===id?'selected':''}>${title(id)}</option>`).join('');

createTopic=async function(){
  const [settings,usage,estimate]=await Promise.all([api('/ai/settings'),api('/ai/usage'),api('/ai/topic-draft-estimate')]);
  app.innerHTML=`<section class="form"><div class="eyebrow">AI authoring</div><h1>Create topic</h1><p class="lead">Generate a structured learning draft. Quality review is a separate, explicit step.</p><form id="topic-form"><label>Topic title<input name="title" required placeholder="Gaussian Mixture Models"></label><label>Primary category<select name="category">${categoryOptions('mathematical_foundations')}</select></label><label>Difficulty<select name="difficulty">${['beginner','intermediate','advanced'].map(value=>`<option value="${value}">${title(value)}</option>`).join('')}</select></label><label>Tags (comma-separated)<input name="tags"></label><label>Focus<textarea name="focus" rows="5" placeholder="What should this explanation make clear?"></textarea></label><label>Explanation depth<select name="depth">${['standard','deep','ultimate'].map(value=>`<option value="${value}" ${value==='ultimate'?'selected':''}>${title(value)}</option>`).join('')}</select></label>${['mathematics','examples','misconceptions'].map(value=>`<label class="check"><input type="checkbox" name="${value}" checked> Include ${title(value)}</label>`).join('')}<p class="notice">Generate Draft maximum: ${money(estimate.maximum_estimated_cost_usd)} · remaining monthly budget: ${money(usage.remaining_budget_usd)}. This makes one authoring call only.</p><button id="generate-topic" ${settings.api_key_configured&&settings.enabled?'':'disabled'}>Generate draft</button><div id="topic-status" class="status" aria-live="polite"></div></form></section>`;
  const form=document.querySelector('#topic-form'),button=document.querySelector('#generate-topic'),status=document.querySelector('#topic-status');
  form.onsubmit=async event=>{
    event.preventDefault();
    const values=new FormData(form);
    const payload={title:values.get('title'),category:values.get('category'),difficulty:values.get('difficulty'),tags:(values.get('tags')||'').split(',').map(value=>value.trim()).filter(Boolean),focus:values.get('focus'),depth:values.get('depth'),include_mathematics:values.get('mathematics')==='on',include_examples:values.get('examples')==='on',include_misconceptions:values.get('misconceptions')==='on'};
    try{
      const draft=await runUiAction({key:'topic-generate',button,loadingLabel:'Generating…',status,startedMessage:'Generating topic draft…',scope:'TOPIC UI',action:()=>request('/ai/topic-draft','POST',payload)});
      if(!draft?.ignored)location.hash='#/drafts/'+draft.id;
    }catch(_){/* runUiAction keeps the form and exposes a retryable error. */}
  };
};

topic=async function(id){
  const item=await api('/topics/'+id),math=item.mathematical_foundation;
  app.innerHTML=`<article><div class="topic-head"><a class="text-link" href="#/topics">← All topics</a><div class="eyebrow" style="margin-top:24px">${title(item.category)} · ${title(item.difficulty)}</div><h1>${esc(item.title)}</h1><p class="lead">${esc(item.one_sentence_summary)}</p>${(item.tags||[]).map(tag=>`<span class="pill">${esc(tag)}</span>`).join('')}</div><section class="card section"><div class="eyebrow">20–60 seconds</div><h2>Quick recall</h2><p>${text(item.quick_recall)}</p></section>${item.big_picture?`<section class="section"><h2>Big picture</h2><p>${text(item.big_picture)}</p></section>`:''}${item.intuition?`<section class="section"><h2>Intuition</h2><p>${text(item.intuition)}</p></section>`:''}<section class="section"><h2>The central idea</h2><p>${text(item.core_explanation)}</p></section>${math?`<section class="section math-section"><div class="eyebrow">Mathematical foundations</div><h2>${esc(math.overview||'Mathematical foundation')}</h2>${(math.prerequisites||[]).map(value=>`<span class="pill">${esc(value)}</span>`).join('')}${(math.sections||[]).map(section=>`<div class="math-block"><h3>${esc(section.title)}</h3><p>${text(section.explanation)}</p>${(section.equations||[]).map(equation=>`<div class="equation">\\[${esc(equation.latex)}\\]</div><p class="muted">${text(equation.explanation)}</p>`).join('')}</div>`).join('')}</section>`:''}${item.mechanism?`<section class="section"><h2>Mechanism</h2><p>${text(item.mechanism)}</p></section>`:''}${item.ml_relevance?`<section class="section"><h2>ML relevance</h2><p>${text(item.ml_relevance)}</p></section>`:''}${item.practical_example?`<section class="section"><h2>Practical example</h2><p>${text(item.practical_example)}</p></section>`:''}<section class="section"><h2>Sources</h2>${(item.sources||[]).map(source=>`<p class="source">${source.url?`<a class="text-link" href="${esc(source.url)}" target="_blank" rel="noreferrer">${esc(source.title)}</a>`:esc(source.title)}<br><span class="muted">${title(source.type)}</span></p>`).join('')}</section></article>`;
  typeset();
};

draftReview=async function(id){
  const [draft,estimate,revisions]=await Promise.all([api('/ai/drafts/'+id),api('/ai/drafts/'+id+'/quality-review-estimate'),api('/ai/drafts/'+id+'/quality-revisions')]);
  if(draft.state!=='draft'){app.innerHTML=`<section class="empty"><h1>Draft ${esc(draft.state)}</h1></section>`;return;}
  let payload=stripLegacyGraph(draft.payload),dirty=false;
  const review=draft.quality_review||{status:'not_run'};
  const reviewLabel={not_run:'Not run',running:'Reviewing',reviewed:'Reviewed',failed:'Review failed'}[review.status]||title(review.status);
  app.innerHTML=`<section class="form"><div class="eyebrow">AI draft · human review required</div><h1>${esc(draft.title)}</h1><p class="notice">Quality review: <strong>${esc(reviewLabel)}</strong>. Generate Draft has already been paid; review sends this existing draft to one quality-review call only.</p><div id="draft-status" class="status" aria-live="polite"></div><form id="draft-form"><div class="field-grid"><label>Title<input data-simple-field="title" value="${esc(payload.title)}"></label><label>Category<select data-simple-field="category">${categoryOptions(payload.category)}</select></label><label>Difficulty<select data-simple-field="difficulty">${['beginner','intermediate','advanced'].map(value=>`<option value="${value}" ${payload.difficulty===value?'selected':''}>${title(value)}</option>`).join('')}</select></label><label>Tags<input id="simple-tags" value="${esc((payload.tags||[]).join(', '))}"></label>${simpleTopicFields.map(field=>`<label class="wide">${title(field)}<textarea data-simple-field="${field}">${esc(payload[field]||'')}</textarea></label>`).join('')}<label class="wide">Mathematical foundation overview<textarea id="simple-math-overview">${esc(payload.mathematical_foundation?.overview||'')}</textarea></label></div><div class="actions"><button type="button" id="simple-save" class="secondary">Save edits</button><button type="button" id="simple-approve">Validate & approve</button><button type="button" id="simple-discard" class="danger">Discard</button></div></form><details><summary>Advanced JSON editor</summary><p class="muted">Apply JSON locally, then save. Topic IDs remain backend-owned on approval.</p><textarea id="simple-json" class="draft-editor">${esc(JSON.stringify(payload,null,2))}</textarea><button type="button" id="simple-apply" class="secondary">Apply advanced JSON</button></details><section class="card"><h2>Quality review</h2><p class="notice">Review maximum: ${money(estimate.maximum_estimated_cost_usd)} · remaining: ${money(estimate.remaining_budget_usd)}</p><button type="button" id="simple-quality" ${review.status==='running'?'disabled':''}>${review.status==='reviewed'?'Run explicit re-review':'Run quality review'}</button>${revisions.revisions.length?`<p class="muted">${revisions.revisions.length} saved quality-review revision(s) are available for recovery.</p>`:''}</section></section>`;
  const status=document.querySelector('#draft-status'),form=document.querySelector('#draft-form'),saveButton=document.querySelector('#simple-save'),approveButton=document.querySelector('#simple-approve'),discardButton=document.querySelector('#simple-discard'),qualityButton=document.querySelector('#simple-quality');
  const sync=()=>{
    form.querySelectorAll('[data-simple-field]').forEach(field=>payload[field.dataset.simpleField]=field.value);
    payload.tags=document.querySelector('#simple-tags').value.split(',').map(value=>value.trim()).filter(Boolean);
    payload.mathematical_foundation={...(payload.mathematical_foundation||{sections:[],prerequisites:[]}),overview:document.querySelector('#simple-math-overview').value};
    payload=stripLegacyGraph(payload);dirty=true;
    document.querySelector('#simple-json').value=JSON.stringify(payload,null,2);
  };
  const persist=async()=>{
    sync();
    const saved=await request('/ai/drafts/'+id,'PUT',{payload});
    payload=stripLegacyGraph(saved.payload);dirty=false;
    document.querySelector('#simple-json').value=JSON.stringify(payload,null,2);
    return saved;
  };
  form.oninput=sync;form.onchange=sync;
  document.querySelector('#simple-apply').onclick=()=>{
    try{
      const parsed=JSON.parse(document.querySelector('#simple-json').value);
      if(!parsed||Array.isArray(parsed)||!parsed.title||!parsed.category||!parsed.difficulty||!parsed.quick_recall||!parsed.core_explanation)throw new Error('Required topic fields are missing.');
      payload=stripLegacyGraph(parsed);dirty=true;
      form.querySelectorAll('[data-simple-field]').forEach(field=>{if(field.dataset.simpleField in payload)field.value=payload[field.dataset.simpleField]||'';});
      document.querySelector('#simple-tags').value=(payload.tags||[]).join(', ');
      document.querySelector('#simple-math-overview').value=payload.mathematical_foundation?.overview||'';
      showOperationStatus(status,'Advanced JSON applied locally. Save edits to persist it.');
    }catch(error){showOperationStatus(status,`Invalid JSON: ${error.message}`,'error');}
  };
  saveButton.onclick=async()=>{try{await runUiAction({key:`draft:${id}:mutation`,button:saveButton,loadingLabel:'Saving…',status,startedMessage:'Saving changes…',successMessage:'Draft saved.',scope:'DRAFT UI',conflicts:[approveButton,qualityButton,discardButton],action:persist});}catch(_){}};
  approveButton.onclick=async()=>{try{const approved=await runUiAction({key:`draft:${id}:mutation`,button:approveButton,loadingLabel:'Approving…',status,startedMessage:dirty?'Saving changes before approval…':'Validating draft…',scope:'DRAFT UI',conflicts:[saveButton,qualityButton,discardButton],action:async()=>{if(dirty)await persist();const validation=await api('/ai/drafts/'+id+'/validate');if(!validation.valid)throw new Error(validation.errors.map(item=>`${item.field}: ${item.message}`).join(' '));return request('/ai/drafts/'+id+'/approve');}});if(!approved?.ignored)location.hash='#/questions/create/'+slug(payload.title);}catch(_){}};
  discardButton.onclick=async()=>{try{const discarded=await runUiAction({key:`draft:${id}:mutation`,button:discardButton,loadingLabel:'Discarding…',status,startedMessage:'Discarding draft…',scope:'DRAFT UI',conflicts:[saveButton,approveButton,qualityButton],action:()=>request('/ai/drafts/'+id+'/discard')});if(!discarded?.ignored)location.hash='#/draft-queue';}catch(_){}};
  qualityButton.onclick=async()=>{
    const force=review.status==='reviewed';
    if(!window.confirm(`Run ${force?'an explicit re-review':'quality review'}? Maximum cost: ${money(estimate.maximum_estimated_cost_usd)}.`))return;
    try{const result=await runUiAction({key:`draft:${id}:mutation`,button:qualityButton,loadingLabel:force?'Re-reviewing…':'Reviewing…',status,startedMessage:'Quality review is running…',scope:'QUALITY UI',conflicts:[saveButton,approveButton,discardButton],action:()=>request('/ai/drafts/'+id+'/quality-review','POST',{force})});if(!result?.ignored)location.hash='#/drafts/'+id;}catch(_){}}
};

createQuestions=async function(topicId){
  const [topic,settings]=await Promise.all([api('/topics/'+topicId),api('/ai/settings')]);
  app.innerHTML=`<section class="form"><div class="eyebrow">AI authoring</div><h1>Generate questions for ${esc(topic.title)}</h1><p class="lead">Candidates are conceptual and remain drafts until you choose and approve them.</p><form id="question-form"><label>Focus<input name="focus"></label><label>Number of candidates<input name="count" type="number" min="1" max="12" value="5"></label><button id="generate-questions" ${settings.api_key_configured&&settings.enabled?'':'disabled'}>Generate questions</button><div id="question-status" class="status" aria-live="polite"></div></form></section>`;
  const form=document.querySelector('#question-form'),button=document.querySelector('#generate-questions'),status=document.querySelector('#question-status');
  form.onsubmit=async event=>{event.preventDefault();const values=new FormData(form);try{const draft=await runUiAction({key:`question-generate:${topicId}`,button,loadingLabel:'Generating…',status,startedMessage:'Generating conceptual question candidates…',scope:'QUESTION UI',action:()=>request('/ai/question-draft','POST',{topic_id:topicId,focus:values.get('focus'),count:Number(values.get('count'))})});if(!draft?.ignored)location.hash='#/question-drafts/'+draft.id;}catch(_){}};
};

const graphFreeEditTopic=editTopic;
editTopic=async function(id){
  await graphFreeEditTopic(id);
  for(const selector of ['#topic-prerequisite_topic_ids','#topic-related_topic_ids'])document.querySelector(selector)?.closest('label')?.style.setProperty('display','none');
};
