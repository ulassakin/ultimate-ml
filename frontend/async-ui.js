/* Small shared async-action helper for the framework-free authoring UI. */
const uiActionsInFlight=new Set();

const uiActionLog=(scope,event,details={})=>console.info(`[${scope}] ${event}`,details);

const nextPaint=()=>new Promise(resolve=>{
  if(typeof requestAnimationFrame==='function')requestAnimationFrame(()=>resolve());
  else setTimeout(resolve,0);
});

const showOperationStatus=(element,message,kind='')=>{
  if(!element)return;
  element.className=`status${kind?` ${kind}`:''}`;
  element.textContent=message;
  element.setAttribute('role','status');
  element.setAttribute('aria-live','polite');
};

async function runUiAction({key,button,loadingLabel,status,startedMessage,successMessage,scope='ULTIMATE UI',action,conflicts=[]}){
  if(uiActionsInFlight.has(key)){
    uiActionLog(scope,'duplicate click ignored',{key});
    return {ignored:true};
  }
  uiActionsInFlight.add(key);
  const idleLabel=button?.dataset.idleLabel||button?.textContent||'';
  const disabledConflicts=conflicts.filter(item=>item&&item!==button&&!item.disabled);
  if(button){
    button.dataset.idleLabel=idleLabel;
    button.disabled=true;
    button.setAttribute('aria-busy','true');
    button.classList.add('loading');
    button.textContent=loadingLabel;
  }
  disabledConflicts.forEach(item=>{item.disabled=true;item.dataset.uiConflictDisabled='true';});
  showOperationStatus(status,startedMessage||loadingLabel);
  uiActionLog(scope,'clicked',{key});
  const startedAt=performance.now();
  try{
    // Yield one frame after mutating the DOM. This gives Safari and other
    // browsers a chance to paint the loading label before provider work starts.
    await nextPaint();
    const value=await action();
    uiActionLog(scope,'completed',{key,elapsed_ms:Math.round(performance.now()-startedAt)});
    if(successMessage)showOperationStatus(status,successMessage);
    return value;
  }catch(error){
    const message=error?.message||'Request failed. Please retry.';
    uiActionLog(scope,'failed',{key,elapsed_ms:Math.round(performance.now()-startedAt),message});
    showOperationStatus(status,`Request failed: ${message}`,'error');
    throw error;
  }finally{
    uiActionsInFlight.delete(key);
    if(button?.isConnected){
      button.disabled=false;
      button.removeAttribute('aria-busy');
      button.classList.remove('loading');
      button.textContent=idleLabel;
    }
    disabledConflicts.forEach(item=>{
      if(item.isConnected&&item.dataset.uiConflictDisabled==='true'){
        item.disabled=false;
        delete item.dataset.uiConflictDisabled;
      }
    });
  }
}
