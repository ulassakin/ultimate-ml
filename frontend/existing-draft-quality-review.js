/* Opt-in review for paid, pre-gate drafts. It never invokes topic generation. */
const existingQualityLabel=status=>({not_reviewed:'Not run',reviewing:'Reviewing',reviewed:'Reviewed',needs_attention:'Needs attention',review_failed:'Review failed'}[status]||title(status));
const existingQualityDraftQueue=draftQueue;
draftQueue=async function(){
  await existingQualityDraftQueue();
  const host=app.querySelector('section');
  if(!host)return;
  const data=await api('/ai/topic-drafts');
  const card=document.createElement('section');
  card.className='section card';
  card.innerHTML=`<h2>All active topic drafts</h2><p class="muted">Original generation is already paid. Quality review is optional and never regenerates a topic or questions.</p><ul class="queue-list">${data.drafts.map(d=>`<li class="queue-row"><span><strong>${esc(d.title)}</strong><br><small>Quality review: ${esc(existingQualityLabel(d.quality_review.status))}</small></span><span><a class="button secondary" href="#/drafts/${d.id}">Review draft</a>${d.quality_review.status!=='reviewing'?` <button class="existing-quality-run secondary" data-id="${d.id}">Run quality review</button>`:''}</span></li>`).join('')||'<li class="muted">No active topic drafts.</li>'}</ul>`;
  host.append(card);
  card.querySelectorAll('.existing-quality-run').forEach(button=>button.onclick=async()=>{
    try{
      const estimate=await api('/ai/drafts/'+button.dataset.id+'/quality-review-estimate');
      const draft=data.drafts.find(item=>item.id===button.dataset.id), force=['reviewed','needs_attention'].includes(draft.quality_review.status);
      const message=`Original generation is already paid and will not repeat. Quality-review maximum: ${money(estimate.maximum_estimated_cost_usd)}. Remaining budget: ${money(estimate.remaining_budget_usd)}.${force?' This is an explicit re-review.':''}`;
      if(!window.confirm(message))return;
      busy(button,force?'Re-reviewing…':'Reviewing…');
      const result=await request('/ai/drafts/'+button.dataset.id+'/quality-review','POST',{force});
      if(result.reused)location.hash='#/drafts/'+button.dataset.id;else location.hash='#/drafts/'+button.dataset.id;
    }catch(error){button.disabled=false;button.classList.remove('loading');button.textContent=error.message}
  });
};

const existingQualityDraftReview=draftReview;
draftReview=async function(id){
  await existingQualityDraftReview(id);
  const heading=app.querySelector('h1');
  if(!heading)return;
  const [draft,estimate,revisions]=await Promise.all([api('/ai/drafts/'+id),api('/ai/drafts/'+id+'/quality-review-estimate'),api('/ai/drafts/'+id+'/quality-revisions')]);
  const state=draft.quality_review||{status:'not_reviewed'};
  const relationshipHints={prerequisite_topic_ids:'Prerequisites are strict learning dependencies: a learner would be materially blocked or confused without them first.',related_topic_ids:'Related topics are strong conceptual neighbors, extensions, alternatives, comparisons, or central pairings—not generic implementation adjacency.'};
  Object.entries(relationshipHints).forEach(([field,hint])=>{const select=app.querySelector('#'+field);if(select&&!select.dataset.relationshipHint){select.dataset.relationshipHint='true';select.insertAdjacentHTML('beforebegin',`<p class="muted">${esc(hint)} Empty is often correct.</p>`);}});
  const currentEdges=(draft.payload.relationship_justifications||[]).filter(edge=>(edge.relationship==='prerequisite'?draft.payload.prerequisite_topic_ids: draft.payload.related_topic_ids||[]).includes(edge.topic_id));
  if(currentEdges.length){const relationships=document.createElement('section');relationships.className='card';relationships.innerHTML=`<h2>Saved relationship rationale</h2><ul>${currentEdges.map(edge=>`<li><strong>${esc(title(edge.relationship))}</strong> · ${esc(edge.topic_id)} · ${esc(edge.confidence||'not rated')}<br><span class="muted">${esc(edge.reason)}</span></li>`).join('')}</ul>`;heading.after(relationships);}
  const card=document.createElement('section');
  card.className='card';
  const original=draft.metadata?.request_cost_usd;
  card.innerHTML=`<h2>Existing draft quality review</h2><p><strong>Generation:</strong> Complete${original!==undefined?` · already paid (${money(original)})`:''}<br><strong>Quality review:</strong> ${esc(existingQualityLabel(state.status))}</p><p class="muted">This sends the existing full draft to the reviewer only. It never reruns topic generation or changes questions, review history, or approved content.</p><p class="notice">Review maximum: ${money(estimate.maximum_estimated_cost_usd)} · remaining: ${money(estimate.remaining_budget_usd)} · reviewer: ${esc(estimate.reviewer_prompt_version)}</p><div class="actions"><button id="run-existing-quality" ${state.status==='reviewing'?'disabled':''}>${state.status==='reviewed'||state.status==='needs_attention'?'Run explicit re-review':'Run quality review'}</button></div>${revisions.revisions.length?`<details><summary>Quality-review revisions (${revisions.revisions.length})</summary><ul>${revisions.revisions.map(r=>`<li>${esc(title(r.revision_type))} · ${esc(r.created_at)} · ${esc(r.payload_hash.slice(0,12))} <button class="view-quality-revision secondary" data-revision="${r.id}">View</button> <button class="restore-quality-revision secondary" data-revision="${r.id}">Restore</button></li>`).join('')}</ul><pre id="quality-revision-json" class="draft-editor"></pre></details>`:''}<p id="existing-quality-status" class="muted"></p>`;
  heading.after(card);
  const run=card.querySelector('#run-existing-quality');
  run.onclick=async()=>{const force=state.status==='reviewed'||state.status==='needs_attention';busy(run,force?'Re-reviewing…':'Reviewing…');card.querySelector('#existing-quality-status').textContent=force?'This explicit re-review has a new estimate and makes one reviewer call.':'Original generation will not be repeated.';try{const result=await request('/ai/drafts/'+id+'/quality-review','POST',{force});if(result.reused)card.querySelector('#existing-quality-status').textContent=result.reason;else location.hash='#/drafts/'+id}catch(error){run.disabled=false;run.classList.remove('loading');run.textContent='Run quality review';card.querySelector('#existing-quality-status').textContent=error.message}};
  card.querySelectorAll('.restore-quality-revision').forEach(button=>button.onclick=async()=>{busy(button,'Restoring…');try{await request(`/ai/drafts/${id}/quality-revisions/${button.dataset.revision}/restore`,'POST');location.hash='#/drafts/'+id}catch(error){button.disabled=false;button.textContent=error.message}});
  card.querySelectorAll('.view-quality-revision').forEach(button=>button.onclick=async()=>{const revision=await api(`/ai/drafts/${id}/quality-revisions/${button.dataset.revision}`);card.querySelector('#quality-revision-json').textContent=JSON.stringify(revision.payload,null,2)});
};
