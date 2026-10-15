/* Pure browser-clock prompt latency bookkeeping; also tested under Node.
 * Never subtract remote monotonic timestamps from performance.now().
 */
(function(root){
  "use strict";
  class H3Latency {
    constructor(){
      this.submissions=new Map();
      this.observations=new Map();
      this.clientDecodeMs=null;
      this.clientBufferMs=null;
      this.rtcDecodeMs=null;
      this.rtcBufferMs=null;
      this.rtcPrevious=null;
    }
    submit(version, at, ackAt){
      // Re-sending an unchanged prompt is an acknowledgement, not a new version.
      if(!this.submissions.has(version))this.submissions.set(version,{version,at,ackMs:ackAt-at});
    }
    observe(version, proposal, at){
      if(version>0 && !this.observations.has(version))this.observations.set(version,{version,proposal,at});
    }
    samples(){
      return [...this.submissions.values()].map(s=>{
        const o=this.observations.get(s.version);
        return {...s,proposal:o?.proposal,shownAt:o?.at,e2eMs:o ? Math.max(0,o.at-s.at):null};
      });
    }
    latest(){
      return this.samples().filter(s=>s.e2eMs!==null).sort((a,b)=>b.shownAt-a.shownAt)[0]||null;
    }
    rtc(stats){
      const p=this.rtcPrevious;
      if(p && stats.id===p.id){
        const n=stats.framesDecoded-p.framesDecoded;
        if(n>0 && Number.isFinite(stats.totalDecodeTime) && Number.isFinite(p.totalDecodeTime))
          this.rtcDecodeMs=Math.max(0,(stats.totalDecodeTime-p.totalDecodeTime)*1000/n);
        const emitted=stats.jitterBufferEmittedCount-p.jitterBufferEmittedCount;
        if(emitted>0 && Number.isFinite(stats.jitterBufferDelay) && Number.isFinite(p.jitterBufferDelay))
          this.rtcBufferMs=Math.max(0,(stats.jitterBufferDelay-p.jitterBufferDelay)*1000/emitted);
      }
      this.rtcPrevious={...stats};
    }
    rows(state){
      const metrics=state?.proposal_metrics||[];
      return this.samples().slice(-6).map(s=>{
        const adopted=metrics.some(m=>m.prompt_version===s.version);
        const newerShown=[...this.observations.keys()].some(v=>v>s.version);
        return {...s,status:s.e2eMs!==null?"已显示":
          newerShown ? "未显示（已跳过）":
          !adopted && state?.latest_prompt_version>s.version ? "已被更新输入覆盖":
          state?.stopped ? "已停止，未显示":
          adopted ? "生成 / 等待显示":"等待服务端处理"};
      });
    }
  }
  if(typeof module!=="undefined" && module.exports)module.exports=H3Latency;
  else root.H3Latency=H3Latency;
})(typeof window!=="undefined"?window:globalThis);
