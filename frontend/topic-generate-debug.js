// Temporary, cache-busted instrumentation for the Create Topic click-to-fetch path.
const TOPIC_GENERATE_DEBUG_BUILD='topic-generate-debug-v1';
const topicGenerateDebugLog=(event,details={})=>console.info('[TOPIC GENERATE DEBUG]',event,JSON.stringify(details));

function topicGeneratePayloadFromForm(form){
  const values=new FormData(form);
  return {
    title:values.get('title'),
    category:values.get('category'),
    difficulty:values.get('difficulty'),
    depth:values.get('depth'),
    tags:(values.get('tags')||'').split(',').map(value=>value.trim()).filter(Boolean),
    focus:values.get('focus'),
    include_mathematics:values.get('mathematics')==='on',
    include_examples:values.get('examples')==='on',
    include_misconceptions:values.get('misconceptions')==='on',
    suggest_related_topics:true
  };
}

const topicGenerateDebugCreateTopic=createTopic;
createTopic=async function(){
  try{
    await topicGenerateDebugCreateTopic();
    const form=document.querySelector('#topic-form');
    const button=document.querySelector('#generate-topic');
    if(!form||!button){
      topicGenerateDebugLog('Create Topic render missing expected form or button',{form:!!form,button:!!button});
      return;
    }
    const marker=document.createElement('p');
    marker.id='topic-generate-debug-build';
    marker.className='muted';
    marker.textContent='Create topic build: topic-generate-debug-v1';
    form.insertBefore(marker,form.firstChild);
    const originalSubmit=form.onsubmit;
    button.addEventListener('click',()=>topicGenerateDebugLog('button click',{button_id:button.id,button_type:button.type||'submit',disabled:button.disabled}));
    form.onsubmit=async function(event){
      topicGenerateDebugLog('submit handler entry',{default_prevented:event.defaultPrevented});
      try{
        const payload=topicGeneratePayloadFromForm(form);
        topicGenerateDebugLog('current form values',payload);
        topicGenerateDebugLog('generated request payload',payload);
        topicGenerateDebugLog('request URL and method',{url:'/api/ai/topic-draft',method:'POST'});
        return await originalSubmit.call(form,event);
      }catch(error){
        topicGenerateDebugLog('synchronous handler exception',{name:error?.name||typeof error,message:error?.message||String(error)});
        throw error;
      }
    };
  }catch(error){
    topicGenerateDebugLog('Create Topic instrumentation exception',{name:error?.name||typeof error,message:error?.message||String(error)});
    throw error;
  }
};
