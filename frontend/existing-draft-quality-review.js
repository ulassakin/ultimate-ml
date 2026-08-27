/* Opt-in review for paid, pre-gate drafts. It never invokes topic generation. */
const existingQualityLabel=status=>({not_run:'Not run',running:'Reviewing',reviewed:'Reviewed',failed:'Review failed'}[status]||title(status));
const draftLifecycleBusy=new Set();
let draftLifecycleNotice='';
const lifecycleLabel=status=>({awaiting_quality_review:'Awaiting quality review',running:'Quality review running',incomplete:'Incomplete / needs attention',failed:'Failed',ready_for_review:'Ready for review',quality_reviewed:'Quality reviewed'}[status]||title(status));
async function runDraftLifecycleAction(button, work){
  const id=button.dataset.id;
  if(draftLifecycleBusy.has(id))return;
  draftLifecycleBusy.add(id);
  const label=button.textContent;
  busy(button,button.dataset.action==='delete'?'Deleting…':'Restarting…');
  try{return await work();}
  catch(error){draftLifecycleNotice=`Draft action failed: ${error.message}`;return null;}
  finally{draftLifecycleBusy.delete(id);if(button.isConnected){button.disabled=false;button.classList.remove('loading');button.textContent=label;}}
}
const existingQualityDraftQueue=draftQueue;
draftQueue=async function(){
  await existingQualityDraftQueue();
  const host=app.querySelector('section');
  if(!host)return;
  const data=await api('/ai/topic-drafts');
  const card=document.createElement('section');
  card.className='section card';
  card.innerHTML=`<h2>All active topic drafts</h2><p class="muted">Generate Draft pays only for authoring. Quality review never regenerates a topic or questions, and runs only when you choose it. Restart creates a fresh, separately budgeted generation and archives the old draft only after success.</p><p id="draft-lifecycle-status" class="${draftLifecycleNotice?'status':'muted'}">${esc(draftLifecycleNotice)}</p><ul class="queue-list">${data.drafts.map(d=>{const restartable=['incomplete','failed'].includes(d.lifecycle_status);return `<li class="queue-row"><span><strong>${esc(d.title)}</strong><br><small>${esc(lifecycleLabel(d.lifecycle_status))} · Quality review: ${esc(existingQualityLabel(d.quality_review.status))}</small></span><span><a class="button secondary" href="#/drafts/${d.id}">Review draft</a>${!restartable&&d.quality_review.status!=='running'?` <button class="existing-quality-run secondary" data-id="${d.id}">${d.quality_review.status==='reviewed'?'Run explicit re-review':'Run quality review'}</button>`:''}${restartable?` <button class="restart-draft secondary" data-id="${d.id}" data-action="restart">Restart generation</button>`:''} <button class="delete-draft danger" data-id="${d.id}" data-action="delete">Delete draft</button></span></li>`}).join('')||'<li class="muted">No active topic drafts.</li>'}</ul>`;
  host.append(card);
  card.querySelectorAll('.existing-quality-run').forEach(button=>button.onclick=async()=>{
    try{
      const estimate=await api('/ai/drafts/'+button.dataset.id+'/quality-review-estimate');
      const draft=data.drafts.find(item=>item.id===button.dataset.id), force=draft.quality_review.status==='reviewed';
      const message=`Original generation is already paid and will not repeat. Quality-review maximum: ${money(estimate.maximum_estimated_cost_usd)}. Remaining budget: ${money(estimate.remaining_budget_usd)}.${force?' This is an explicit re-review.':''}`;
      if(!window.confirm(message))return;
      busy(button,force?'Re-reviewing…':'Reviewing…');
      const result=await request('/ai/drafts/'+button.dataset.id+'/quality-review','POST',{force});
      if(result.reused)location.hash='#/drafts/'+button.dataset.id;else location.hash='#/drafts/'+button.dataset.id;
    }catch(error){button.disabled=false;button.classList.remove('loading');button.textContent=error.message}
  });
  card.querySelectorAll('.delete-draft').forEach(button=>button.onclick=async()=>{
    if(!window.confirm('Delete this draft? This removes the draft only. Approved topics and questions are not affected.'))return;
    const result=await runDraftLifecycleAction(button,()=>api('/ai/drafts/'+button.dataset.id,{method:'DELETE'}));
    if(result){draftLifecycleNotice='Draft deleted. Approved topics and questions were not affected.';await draftQueue();}
    else{await draftQueue();}
  });
  card.querySelectorAll('.restart-draft').forEach(button=>button.onclick=async()=>{
    const result=await runDraftLifecycleAction(button,async()=>{
      const estimate=await api('/ai/drafts/'+button.dataset.id+'/restart-estimate');
      if(!window.confirm(`Restart generation from the original inputs? This makes a new provider call. Maximum cost: ${money(estimate.maximum_estimated_cost_usd)}. Remaining budget: ${money(estimate.remaining_budget_usd)}.`))return {cancelled:true};
      return request('/ai/drafts/'+button.dataset.id+'/restart','POST');
    });
    if(result?.cancelled)return;
    if(result){draftLifecycleNotice=`Fresh draft created for ${result.draft.payload.title}. The old incomplete draft was archived.`;await draftQueue();}
    else{await draftQueue();}
  });
};

const existingQualityDraftReview=draftReview;
draftReview=async function(id){
  await existingQualityDraftReview(id);
  const heading=app.querySelector('h1');
  if(!heading)return;
  const [draft,estimate,revisions]=await Promise.all([api('/ai/drafts/'+id),api('/ai/drafts/'+id+'/quality-review-estimate'),api('/ai/drafts/'+id+'/quality-revisions')]);
  const state=draft.quality_review||{status:'not_run'};
  const card=document.createElement('section');
  card.className='card';
  const original=draft.metadata?.request_cost_usd;
  card.innerHTML=`<h2>Draft quality review</h2><p><strong>Generation:</strong> Complete${original!==undefined?` · already paid (${money(original)})`:''}<br><strong>Quality review:</strong> ${esc(existingQualityLabel(state.status))}</p><p class="muted">This sends the existing full draft to the reviewer only. It never reruns topic generation or changes questions, review history, or approved content.</p><p class="notice">Review maximum: ${money(estimate.maximum_estimated_cost_usd)} · remaining: ${money(estimate.remaining_budget_usd)} · reviewer: ${esc(estimate.reviewer_prompt_version)}</p><div class="actions"><button id="run-existing-quality" ${state.status==='running'?'disabled':''}>${state.status==='reviewed'?'Run explicit re-review':'Run quality review'}</button></div>${revisions.revisions.length?`<details><summary>Quality-review revisions (${revisions.revisions.length})</summary><ul>${revisions.revisions.map(r=>`<li>${esc(title(r.revision_type))} · ${esc(r.created_at)} · ${esc(r.payload_hash.slice(0,12))} <button class="view-quality-revision secondary" data-revision="${r.id}">View</button> <button class="restore-quality-revision secondary" data-revision="${r.id}">Restore</button></li>`).join('')}</ul><pre id="quality-revision-json" class="draft-editor"></pre></details>`:''}<p id="existing-quality-status" class="muted"></p>`;
  heading.after(card);
  const run=card.querySelector('#run-existing-quality');
  run.onclick=async()=>{const force=state.status==='reviewed';busy(run,force?'Re-reviewing…':'Reviewing…');card.querySelector('#existing-quality-status').textContent=force?'This explicit re-review has a new estimate and makes one reviewer call.':'Original generation will not be repeated.';try{const result=await request('/ai/drafts/'+id+'/quality-review','POST',{force});if(result.reused)card.querySelector('#existing-quality-status').textContent=result.reason;else location.hash='#/drafts/'+id}catch(error){run.disabled=false;run.classList.remove('loading');run.textContent='Run quality review';card.querySelector('#existing-quality-status').textContent=error.message}};
  card.querySelectorAll('.restore-quality-revision').forEach(button=>button.onclick=async()=>{busy(button,'Restoring…');try{await request(`/ai/drafts/${id}/quality-revisions/${button.dataset.revision}/restore`,'POST');location.hash='#/drafts/'+id}catch(error){button.disabled=false;button.textContent=error.message}});
  card.querySelectorAll('.view-quality-revision').forEach(button=>button.onclick=async()=>{const revision=await api(`/ai/drafts/${id}/quality-revisions/${button.dataset.revision}`);card.querySelector('#quality-revision-json').textContent=JSON.stringify(revision.payload,null,2)});
};
