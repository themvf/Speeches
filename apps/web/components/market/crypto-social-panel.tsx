"use client";
import { useEffect, useState } from "react";
import { CryptoInfluencerPanel } from "./crypto-influencer-panel";
import type { Tracking } from "@/lib/crypto-social";

type Edge = {source_id:string;target_id:string;source:string;target:string;kind:string;weight:number;evidence:string};
type Research = {
 status:string;
 tracking?:Tracking|null;
 pilot?:{reserved_credits:number;credit_limit:number;estimated_credits:number|null;start_at:string;end_at:string;outstanding:number};
 daily?:{day:string;posts:number;authors:number;originals:number;replies:number;quotes:number;reposts:number;exhausted:number;searched:number;windows:number}[];
 accounts?:{id:string;handle:string;posts:number;active_days:number;posts_per_day:number;median_likes:number|null;amplifiers:number}[];
 edges?:Edge[];
 posts?:{id:string;text:string;url:string;handle:string}[];
};

export function CryptoSocialPanel() {
 const [coin,setCoin]=useState("ZCAT");
 const [data,setData]=useState<Research|null>(null);
 const [error,setError]=useState("");
 const [kind,setKind]=useState("all");
 const [selected,setSelected]=useState<Edge|null>(null);
 useEffect(()=>{
  const controller=new AbortController();
  setData(null);setError("");setSelected(null);
  fetch(`/api/market/crypto/social?coin=${coin}`,{signal:controller.signal})
   .then(async r=>{const body=await r.json();if(!r.ok||!body.ok)throw new Error();setData(body.data);})
   .catch(()=>{if(!controller.signal.aborted)setError("Could not load saved research. Try reloading.");});
  return ()=>controller.abort();
 },[coin]);
 const edges=(data?.edges??[]).filter(e=>kind==="all"||e.kind===kind);
 const ids=Array.from(new Set(edges.flatMap(e=>[e.source_id,e.target_id]))).slice(0,20);
 const visible=edges.filter(e=>ids.includes(e.source_id)&&ids.includes(e.target_id));
 const positions=new Map(ids.map((id,i)=>[id,{x:300+210*Math.cos(2*Math.PI*i/ids.length),y:230+165*Math.sin(2*Math.PI*i/ids.length)}]));
 const names=new Map(edges.flatMap(e=>[[e.source_id,e.source],[e.target_id,e.target]]));
 const panel="rounded-xl border border-[color:var(--line)] p-4 space-y-3";
 return <section className={panel} aria-label="Crypto social research">
  <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="font-semibold">X Research · ZCAT & Zcash</h2>
   <select aria-label="Research coin" value={coin} onChange={e=>setCoin(e.target.value)} className="rounded bg-slate-900 p-2 text-sm"><option value="ZCAT">Anonymous Cat (ZCAT)</option><option value="ZEC">Zcash (ZEC)</option></select>
  </div>
  <p className="text-xs text-[color:var(--ink-faint)]">Saved research only. Opening this panel consumes no X API credits. Counts are observed matches, not total X volume.</p>
  {coin==="ZCAT"&&<p className="break-all text-xs text-[color:var(--ink-faint)]">User-supplied token address: HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR. Independent identity verification pending.</p>}
  {error?<p role="alert">{error}</p>:!data?<p>Loading saved research…</p>:data.status!=="ready"?<p>The pilot has not collected data yet. Run the manual Crypto Social Research Pilot workflow after configuring its credentials.</p>:<>
   <p className="text-sm">Reserved: {Number(data.pilot?.reserved_credits??0).toLocaleString()} / 50,000 credits · Estimated used: {Number(data.pilot?.estimated_credits??0).toLocaleString()} · {data.pilot?.start_at?.slice(0,10)} to {data.pilot?.end_at?.slice(0,10)} (end exclusive, UTC)</p>
   {Number(data.pilot?.outstanding)>0&&<p role="status" className="text-amber-400">Collection is paused for an outstanding or uncertain request. Its credit reservation is retained.</p>}
   <div className="overflow-x-auto"><table className="w-full text-left text-xs"><caption className="text-left py-2 font-semibold">Daily observed activity</caption>
    <thead><tr>{["UTC day","Posts","Authors","Originals","Replies","Quotes","Reposts","Search coverage"].map(h=><th className="p-2" key={h}>{h}</th>)}</tr></thead>
    <tbody>{data.daily?.map(d=><tr key={d.day} className="border-t border-[color:var(--line)]"><td className="p-2">{d.day}</td><td>{d.searched?d.posts:"—"}</td><td>{d.searched?d.authors:"—"}</td><td>{d.originals}</td><td>{d.replies}</td><td>{d.quotes}</td><td>{d.reposts}</td><td>{!d.searched?"Not sampled":d.exhausted===d.windows?"Search exhausted; completeness unverified":`Partial · ${d.searched}/${d.windows} windows searched`}</td></tr>)}</tbody>
   </table></div>
   {data.tracking?<CryptoInfluencerPanel key={coin} data={data.tracking}/>:<div className="overflow-x-auto"><table className="w-full text-left text-xs"><caption className="text-left py-2 font-semibold">Candidate influencers · sampled coin activity</caption>
    <thead><tr>{["Account","Posts","Observed/day¹","Active days","Median likes²","Distinct amplifiers³"].map(h=><th className="p-2" key={h}>{h}</th>)}</tr></thead>
    <tbody>{data.accounts?.map(a=><tr key={a.id} className="border-t border-[color:var(--line)]"><td className="p-2"><a href={`https://x.com/${encodeURIComponent(a.handle)}`} target="_blank" rel="noreferrer">@{a.handle}</a></td><td>{a.posts}</td><td>{a.posts_per_day}</td><td>{a.active_days}</td><td>{a.median_likes??"—"}</td><td>{a.amplifiers}</td></tr>)}</tbody>
   </table><p className="text-xs text-[color:var(--ink-faint)]">¹ Sampled coin posts ÷ 7 calendar days, not all account posts. ² Counts at collection time; post ages differ. ³ Distinct observed quoting/reposting accounts. A quote may be critical. This is a candidate ranking, not verified influence.</p></div>}
   <div><label className="text-sm">Interaction graph <select className="ml-2 rounded bg-slate-900 p-2" value={kind} onChange={e=>{setKind(e.target.value);setSelected(null);}}>{["all","reply","quote","repost","mention"].map(k=><option key={k}>{k}</option>)}</select></label>
    <p className="text-xs text-[color:var(--ink-faint)]">Up to 20 accounts from the 60 strongest saved connections. Arrows point from actor to target. No liker identities or inferred coordination.</p>
    {!visible.length?<p className="py-4 text-sm">No observed connections for this filter yet.</p>:<svg viewBox="0 0 600 460" className="w-full max-w-3xl" role="img" aria-label="Observed account interaction network">
     <defs><marker id="social-arrow" markerWidth="8" markerHeight="8" refX="17" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#67e8f9"/></marker></defs>
     {visible.map(e=>{const a=positions.get(e.source_id)!,b=positions.get(e.target_id)!;return <line key={`${e.source_id}-${e.target_id}-${e.kind}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#67e8f9" strokeOpacity={0.45} strokeWidth={Math.min(6,1+Math.log2(e.weight+1))} markerEnd="url(#social-arrow)" onClick={()=>setSelected(e)}><title>{e.source} → {e.target}: {e.kind} ({e.weight})</title></line>;})}
     {ids.map(id=>{const p=positions.get(id)!;const n=new Set(visible.filter(e=>e.target_id===id&&["quote","repost"].includes(e.kind)).map(e=>e.source_id)).size;return <g key={id}><circle cx={p.x} cy={p.y} r={6+Math.min(10,n)} fill="#67e8f9"/><text x={p.x} y={p.y-18} textAnchor="middle" fill="currentColor" fontSize="10">{names.get(id)}</text></g>;})}
    </svg>}
    {selected&&<p className="text-sm"><a className="underline" href={selected.evidence} target="_blank" rel="noreferrer">{selected.source} → {selected.target}: {selected.kind} · open example post</a></p>}
    <details><summary className="cursor-pointer text-sm">Connections and source posts</summary><ul className="text-xs space-y-2 mt-2">{edges.map(e=><li key={`${e.source_id}-${e.target_id}-${e.kind}`}><a className="underline" href={e.evidence} target="_blank" rel="noreferrer">{e.source} → {e.target} · {e.kind} × {e.weight}</a></li>)}</ul></details>
   </div>
   <details><summary className="cursor-pointer text-sm">Recent saved posts</summary><ul className="space-y-3 mt-3 text-sm">{data.posts?.map(p=><li key={p.id}><a className="underline" href={p.url} target="_blank" rel="noreferrer">@{p.handle}</a><p className="whitespace-pre-wrap break-words">{p.text}</p></li>)}</ul></details>
  </>}
 </section>;
}
