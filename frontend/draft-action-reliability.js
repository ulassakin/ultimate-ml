/* Final draft-action layer: one canonical payload, one mutation at a time. */
const draftActionReliabilityBaseReview = draftReview;
const ADVJSON_DEBUG_BUILD = 'advjson-debug-v2';
const advjsonDebug = (event, detail={}) => console.info('[ADVJSON DEBUG]', event, JSON.stringify(detail));

// `payload` is the sole mutable draft object. Structured fields and raw JSON
// are derived views; `persisted` holds revision metadata only.
const draftEditorState = {draftId:null, active:false, payload:null, persisted:null, catalog:[], dirty:false, auditStale:false};

const draftActionError = error => error instanceof Error ? error.message : 'Unexpected local request failure. Please retry.';
const deepCopy = value => structuredClone(value);
const relationIds = values => [...new Set((Array.isArray(values) ? values : []).filter(value => typeof value === 'string' && value))].sort();
const sameRelations = (left,right) => JSON.stringify(relationIds(left)) === JSON.stringify(relationIds(right));

function actionStatus(message, isError=false) {
  const area=document.querySelector('#draft-action-status'), validation=document.querySelector('#validation');
  if(area){area.className=isError?'status error':'status';area.textContent=message;}
  if(isError&&validation){validation.className='status error';validation.textContent=message;}
}
function mutationButtons(disabled) {
  ['#save-draft','#approve-draft','#rebuild-metadata','#apply-json'].forEach(selector=>{
    const button=document.querySelector(selector);if(button)button.disabled=disabled;
  });
}
function persistedDraft(value, operation) {
  if(!value||typeof value!=='object'||!value.payload||typeof value.payload!=='object')throw new Error(`${operation} returned a malformed response. Nothing else was submitted; reload the draft and retry.`);
  return value;
}
function schemaError(payload) {
  if(!payload || typeof payload !== 'object' || Array.isArray(payload)) return 'The advanced editor must contain one topic JSON object.';
  const requiredStrings=['title','category','difficulty','one_sentence_summary','quick_recall','core_explanation'];
  const missing=requiredStrings.filter(field=>typeof payload[field] !== 'string' || !payload[field].trim());
  if(missing.length)return `Schema error: required text field${missing.length===1?'':'s'} ${missing.join(', ')} ${missing.length===1?'is':'are'} missing.`;
  if(!['beginner','intermediate','advanced'].includes(payload.difficulty))return 'Schema error: difficulty must be beginner, intermediate, or advanced.';
  for(const field of ['tags','prerequisite_topic_ids','related_topic_ids'])if(payload[field] !== undefined&&!Array.isArray(payload[field]))return `Schema error: ${field} must be an array.`;
  if(payload.mathematical_foundation !== undefined && (payload.mathematical_foundation === null || typeof payload.mathematical_foundation !== 'object' || Array.isArray(payload.mathematical_foundation)))return 'Schema error: mathematical_foundation must be an object when present.';
  return '';
}
function currentMathAudit(payload) {
  const foundation=payload.mathematical_foundation || {};
  return {global:Array.isArray(foundation.prerequisites)?foundation.prerequisites:[],sections:Object.fromEntries((Array.isArray(foundation.sections)?foundation.sections:[]).filter(section=>section&&typeof section==='object').map((section,index)=>[String(section.title ?? index),Array.isArray(section.prerequisites)?section.prerequisites:[]]))};
}
function refreshAuditStaleness() {
  const payload=draftEditorState.payload, audit=payload?.metadata_resolution;
  if(!payload||!audit||typeof audit!=='object'){draftEditorState.auditStale=false;return false;}
  const edges=audit.durable_edges || {};
  const relationshipsMatch=sameRelations(edges.prerequisites,payload.prerequisite_topic_ids)&&sameRelations(edges.related,payload.related_topic_ids);
  // Math prerequisite order is audit data: exact equality is what the backend
  // validates, so restoring ["Gradients"] clears the stale marker.
  const mathMatches=JSON.stringify(audit.math_prerequisites || {})===JSON.stringify(currentMathAudit(payload));
  draftEditorState.auditStale=!(relationshipsMatch&&mathMatches&&audit.resolved_category===payload.category&&audit.resolved_concept_type===payload.concept_type);
  advjsonDebug('resolver audit comparison',{audit_math_prerequisites:audit.math_prerequisites,current_math_prerequisites:currentMathAudit(payload),audit_stale:draftEditorState.auditStale});
  return draftEditorState.auditStale;
}
function updateJsonView() {
  const editor=document.querySelector('#advanced-json');
  if(editor&&draftEditorState.payload)editor.value=JSON.stringify(draftEditorState.payload,null,2);
}
function ensureMathPrerequisiteEditor() {
  const overview=document.querySelector('#math-overview');
  if(!overview||document.querySelector('#math-prerequisites'))return;
  // This is a normal structured view of resolver-audited math metadata, so a
  // restored Advanced JSON value is immediately inspectable without reopening JSON.
  overview.closest('label')?.insertAdjacentHTML('afterend','<label class="wide">Mathematical prerequisites (one per line)<textarea id="math-prerequisites"></textarea></label>');
}
function renderCanonicalPayload() {
  const payload=draftEditorState.payload;
  if(!payload)return;
  ensureMathPrerequisiteEditor();
  document.querySelectorAll('[data-field]').forEach(element=>{element.value=payload[element.dataset.field] ?? '';});
  document.querySelectorAll('[data-list]').forEach(element=>{element.value=(payload[element.dataset.list]||[]).join('\n');});
  const tags=document.querySelector('#tags');if(tags)tags.value=(payload.tags||[]).join(', ');
  const overview=document.querySelector('#math-overview');if(overview)overview.value=payload.mathematical_foundation?.overview ?? '';
  const mathPrerequisites=document.querySelector('#math-prerequisites');if(mathPrerequisites)mathPrerequisites.value=(payload.mathematical_foundation?.prerequisites||[]).join('\n');
  for(const field of ['prerequisite_topic_ids','related_topic_ids']){
    const select=document.querySelector('#'+field);if(!select)continue;
    const selected=new Set(payload[field]||[]);[...select.options].forEach(option=>option.selected=selected.has(option.value));
  }
  updateJsonView();
  advjsonDebug('structured form after rerender',{math_prerequisites:mathPrerequisites?.value ?? null});
  if(draftEditorState.auditStale)actionStatus('Resolver-owned metadata is stale for these edits. Save is allowed, but rebuild relationships before approval unless the edited values exactly match the audit.',true);
}
function syncStructuredFields() {
  const payload=draftEditorState.payload;
  if(!payload)return;
  document.querySelectorAll('[data-field]').forEach(element=>{payload[element.dataset.field]=element.value;});
  document.querySelectorAll('[data-list]').forEach(element=>{payload[element.dataset.list]=element.value.split('\n').map(value=>value.trim()).filter(Boolean);});
  const tags=document.querySelector('#tags');if(tags)payload.tags=tags.value.split(',').map(value=>value.trim()).filter(Boolean);
  const overview=document.querySelector('#math-overview');if(overview){payload.mathematical_foundation={...(payload.mathematical_foundation||{sections:[],prerequisites:[]}),overview:overview.value};}
  const mathPrerequisites=document.querySelector('#math-prerequisites');if(mathPrerequisites){payload.mathematical_foundation={...(payload.mathematical_foundation||{sections:[],prerequisites:[]}),prerequisites:mathPrerequisites.value.split('\n').map(value=>value.trim()).filter(Boolean)};}
  for(const field of ['prerequisite_topic_ids','related_topic_ids']){
    const select=document.querySelector('#'+field);if(select)payload[field]=[...select.selectedOptions].map(option=>option.value);
  }
  draftEditorState.dirty=true;
  refreshAuditStaleness();
  updateJsonView();
}
function bindStructuredEditors() {
  const form=document.querySelector('#draft-form');
  if(!form)return;
  form.oninput=()=>syncStructuredFields();
  form.onchange=()=>syncStructuredFields();
}
function adoptPersistedDraft(value, message='Saved the persisted draft revision.') {
  const draft=persistedDraft(value,'Save edits');
  draftEditorState.payload=deepCopy(draft.payload);
  draftEditorState.persisted={id:draft.id,updated_at:draft.updated_at,state:draft.state};
  draftEditorState.dirty=false;
  refreshAuditStaleness();
  renderCanonicalPayload();
  actionStatus(message+(draftEditorState.auditStale?' Resolver metadata remains stale; rebuild before approval.':''),draftEditorState.auditStale);
  return draft;
}
async function persistCanonical() {
  syncStructuredFields();
  const body={payload:deepCopy(draftEditorState.payload)};
  advjsonDebug('Save request body',{draft_id:draftEditorState.draftId,payload:body.payload,math_prerequisites:body.payload.mathematical_foundation?.prerequisites ?? null});
  const saved=await request('/ai/drafts/'+draftEditorState.draftId,'PUT',body);
  advjsonDebug('Save backend response',{draft_id:draftEditorState.draftId,persisted_payload:saved.payload,persisted_math_prerequisites:saved.payload?.mathematical_foundation?.prerequisites ?? null,updated_at:saved.updated_at});
  return adoptPersistedDraft(saved);
}
async function runDraftMutation(operation, button, work) {
  if(draftEditorState.active){actionStatus('Another draft action is already in progress. Wait for it to finish.',true);return null;}
  draftEditorState.active=true;
  const originalLabel=button.textContent;
  mutationButtons(true);busy(button,operation+'…');actionStatus(operation+' in progress…');
  try{return await work();}
  catch(error){actionStatus(`${operation} failed: ${draftActionError(error)}`,true);return null;}
  finally{draftEditorState.active=false;if(button.isConnected){button.disabled=false;button.classList.remove('loading');button.textContent=originalLabel;}mutationButtons(false);}
}
function validationMessage(validation) {
  return (validation.errors||[]).map(item=>`${item.field||'validation'}: ${item.message||'Invalid value'}`).join(' · ')||'Validation failed.';
}
async function refreshReliableDraft(id) {
  await draftActionReliabilityBaseReview(id);
  const [draft,catalog]=await Promise.all([api('/ai/drafts/'+id),api('/topics')]);
  if(draft.state!=='draft')return;
  draftEditorState.draftId=id;draftEditorState.catalog=catalog;draftEditorState.payload=deepCopy(draft.payload);
  draftEditorState.persisted={id:draft.id,updated_at:draft.updated_at,state:draft.state};draftEditorState.dirty=false;
  refreshAuditStaleness();
  const heading=app.querySelector('h1');
  if(heading&&!document.querySelector('#draft-action-status'))heading.insertAdjacentHTML('afterend','<p id="draft-action-status" class="status" aria-live="polite"></p>');
  if(heading&&!document.querySelector('#draft-editor-build'))heading.insertAdjacentHTML('afterend',`<p id="draft-editor-build" class="muted">Draft editor build: ${ADVJSON_DEBUG_BUILD}</p>`);
  renderCanonicalPayload();bindStructuredEditors();

  const saveButton=document.querySelector('#save-draft');
  if(saveButton)saveButton.onclick=async()=>runDraftMutation('Save edits',saveButton,persistCanonical);

  const rebuildButton=document.querySelector('#rebuild-metadata');
  if(rebuildButton)rebuildButton.onclick=async()=>runDraftMutation('Rebuild relationships',rebuildButton,async()=>{
    if(draftEditorState.dirty)await persistCanonical();
    const result=await request('/ai/drafts/'+id+'/rebuild-relationships','POST');
    return adoptPersistedDraft(result.draft,result.usage?.cached?'Reused the cached persisted relationship resolution ($0.00).':'Rebuilt and persisted relationship metadata.');
  });

  const approveButton=document.querySelector('#approve-draft');
  if(approveButton)approveButton.onclick=async()=>{const approved=await runDraftMutation('Validate and approve',approveButton,async()=>{
    advjsonDebug('Validate and approve begins',{draft_id:id,dirty:draftEditorState.dirty,persisted_revision:draftEditorState.persisted,payload_math_prerequisites:draftEditorState.payload?.mathematical_foundation?.prerequisites ?? null,resolver_audit_math_prerequisites:draftEditorState.payload?.metadata_resolution?.math_prerequisites ?? null});
    if(draftEditorState.dirty)await persistCanonical();
    advjsonDebug('Validate uses persisted revision',{draft_id:id,persisted_revision:draftEditorState.persisted,payload_math_prerequisites:draftEditorState.payload?.mathematical_foundation?.prerequisites ?? null,resolver_audit_math_prerequisites:draftEditorState.payload?.metadata_resolution?.math_prerequisites ?? null});
    const validation=await api('/ai/drafts/'+id+'/validate');
    if(!validation||typeof validation.valid!=='boolean')throw new Error('Validation returned a malformed response. The draft was not approved.');
    if(!validation.valid)throw new Error(validationMessage(validation));
    const result=await request('/ai/drafts/'+id+'/approve');
    if(!result||!result.topic)throw new Error('Approval returned a malformed response. Reload the Draft Queue before retrying.');
    return result;
  });if(approved){const videoId=draftEditorState.payload.generation_metadata?.youtube_import_id;location.hash=videoId?`#/youtube/${videoId}/questions/${approved.topic.id}`:'#/questions/create/'+slug(approved.topic.title);}};

  const rawButton=document.querySelector('#apply-json');
  if(rawButton){
    advjsonDebug('Apply handler attached',{build:ADVJSON_DEBUG_BUILD,button_id:rawButton.id,button_text:rawButton.textContent.trim(),previous_handler_present:Boolean(rawButton.onclick)});
    rawButton.onclick=async()=>{advjsonDebug('visible Apply button clicked',{build:ADVJSON_DEBUG_BUILD,button_id:rawButton.id});return runDraftMutation('Apply advanced JSON',rawButton,async()=>{
    const rawJson=document.querySelector('#advanced-json').value;
    advjsonDebug('Advanced JSON textarea read',{raw_json:rawJson});
    let payload;try{payload=JSON.parse(rawJson);advjsonDebug('Advanced JSON parse succeeded',{parsed_math_prerequisites:payload.mathematical_foundation?.prerequisites ?? null});}catch(error){advjsonDebug('Advanced JSON parse failed',{message:error.message});throw new Error('Invalid JSON: '+error.message);}
    const problem=schemaError(payload);if(problem)throw new Error(problem);
    draftEditorState.payload=deepCopy(payload);draftEditorState.dirty=true;advjsonDebug('canonical state after Apply',{canonical_math_prerequisites:draftEditorState.payload.mathematical_foundation?.prerequisites ?? null});refreshAuditStaleness();renderCanonicalPayload();
    actionStatus(draftEditorState.auditStale?'Advanced JSON applied to current draft state, but resolver metadata is stale. Rebuild before approval unless the audit values were restored exactly.':'Advanced JSON applied to current draft state. Save edits to persist this revision.',draftEditorState.auditStale);
    return payload;
  });};
  }
}
draftReview=refreshReliableDraft;
