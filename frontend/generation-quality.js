/* Small additive UI layer: keep Phase 3.x's structured editor intact. */
const generationQualityCreateTopic = createTopic;
createTopic = async function(){
  await generationQualityCreateTopic();
  const usage=document.querySelector('.usage');
  if(!usage)return;
  try{
    const estimate=await api('/ai/topic-draft-estimate');
    const detail=document.createElement('p');
    detail.className='muted';
    detail.textContent=estimate.operations.map(item=>`${title(item.operation_type)}: ${money(item.maximum_estimated_cost_usd)}`).join(' · ')+` · total maximum: ${money(estimate.maximum_estimated_cost_usd)}`;
    usage.after(detail);
  }catch(_){/* Existing estimate remains useful if the local server is mid-upgrade. */}
};

const generationQualityDraftReview = draftReview;
draftReview = async function(id){
  await generationQualityDraftReview(id);
  try{
    const draft=await api('/ai/drafts/'+id);
    const report=draft.payload?.quality_report;
    if(!report)return;
    const heading=app.querySelector('h1');
    if(!heading)return;
    const card=document.createElement('section');
    card.className='card';
    const remaining=report.blocking_issues_remaining||[];
    const warnings=report.warnings||[];
    card.innerHTML=`<strong>Generation quality: ${draft.payload.quality_status==='needs_attention'?'Needs attention':'Reviewed'}</strong><p class="muted">Taxonomy, relationships, mathematics, named-method completeness, and provenance were checked. Confidence: ${esc(report.confidence||'medium')}.</p>${remaining.length?`<div class="warnings">${remaining.map(item=>esc(item.message)).join('<br>')}</div>`:''}${warnings.length?`<p class="muted">Warnings: ${warnings.length}</p>`:''}`;
    heading.after(card);
  }catch(_){/* Review remains usable even if a status refresh fails. */}
};
