"use client";
import {useState} from 'react';
import {priceLabel,type MarketPoint,type RunDay,type RunPost} from '@/lib/crypto-run';
export function CryptoRunChart({days,market,posts,coverage,from,to,onRange}:{days:string[];market:MarketPoint[];posts:RunPost[];coverage:RunDay[];from:string;to:string;onRange:(a:string,b:string)=>void}){
 const [log,setLog]=useState(false);const [anchor,setAnchor]=useState<number|null>(null);const [hover,setHover]=useState<number|null>(null);
 const prices=new Map(market.map(p=>[p.day,p]));const counts=new Map<string,number>();for(const p of posts){const day=p.posted_at.slice(0,10);counts.set(day,(counts.get(day)??0)+1);}
 const searched=new Set(coverage.filter(d=>d.searched>0).map(d=>d.day));
 const values=days.map(d=>prices.get(d)?.close).filter((v):v is number=>v!=null&&v>0);const max=Math.max(...values,1e-9),min=Math.min(...values,max);
 const maxVolume=Math.max(...market.map(p=>p.volume),1),maxPosts=Math.max(...counts.values(),1);
 const x=(i:number)=>66+(i+.5)*864/Math.max(days.length,1);const bw=864/Math.max(days.length,1);
 const y=(p:number)=>log?175-(Math.log(p)-Math.log(min))/(Math.log(max/min)||1)*130:175-p/max*130;
 let path='',previous=false;days.forEach((d,i)=>{const p=prices.get(d);if(!p){previous=false;return;}path+=`${previous?'L':'M'}${x(i)},${y(p.close)} `;previous=true;});
 const index=(event:React.PointerEvent<SVGRectElement>)=>{const rect=event.currentTarget.ownerSVGElement!.getBoundingClientRect();return Math.max(0,Math.min(days.length-1,Math.floor(((event.clientX-rect.left)/rect.width*960-66)/bw)));};
 const lo=anchor==null?days.indexOf(from):Math.min(anchor,hover??anchor),hi=anchor==null?days.indexOf(to):Math.max(anchor,hover??anchor);
 const selected=hover==null?null:days[hover],point=selected?prices.get(selected):null;
 return <div>
  <div className="flex flex-wrap items-center justify-between gap-2 mb-2"><p className="text-xs text-[color:var(--ink-faint)]">Drag across the chart to explore a period. Tap to select a day.</p><button className="rounded-lg border border-[color:var(--line)] px-3 py-2 text-xs" aria-pressed={log} onClick={()=>setLog(!log)}>{log?'Log price scale':'Linear price scale'}</button></div>
  <div className="overflow-x-auto"><svg style={{minWidth:640}} viewBox="0 0 960 400" className="w-full touch-pan-y" role="img" aria-label="Aligned daily price, trading volume and observed social posts. Amber marks missing market history; dashes mark unsearched social days.">
   <rect x="66" y="26" width="864" height="150" fill="var(--bg-elev-strong)" rx="5"/>
   {days.map((d,i)=>!prices.has(d)?<rect key={d} x={66+i*bw} y="26" width={bw} height="150" fill="#fbbf24" opacity=".075"/>:null)}
   {lo>=0&&hi>=0&&<rect x={66+lo*bw} y="26" width={Math.max(bw,(hi-lo+1)*bw)} height="329" fill="#7dd3fc" fillOpacity=".04" stroke="#7dd3fc" strokeOpacity=".4"/>}
   <text x="66" y="17" fill="#7dd3fc" fontSize="12">Price · USD</text><text x="58" y="46" textAnchor="end" fill="var(--ink-faint)" fontSize="10">{values.length?priceLabel(max):'—'}</text><text x="58" y="175" textAnchor="end" fill="var(--ink-faint)" fontSize="10">{values.length?log?priceLabel(min):'$0':'—'}</text>
   <path d={path} fill="none" stroke="#7dd3fc" strokeWidth="2.5"/>{days.map((d,i)=>{const p=prices.get(d);return p?.complete===false?<circle key={d} cx={x(i)} cy={y(p.close)} r="4" fill="#fbbf24"><title>Incomplete day: {d}</title></circle>:null;})}
   {values.length===0&&<text x="498" y="104" textAnchor="middle" fill="#fcd34d" fontSize="13">No market history available for this source</text>}
   <text x="66" y="199" fill="#7dd3fc" fontSize="12">Trading volume · USD</text>
   {days.map((d,i)=>{const p=prices.get(d);return p?<rect key={d} x={66+i*bw+1} y={264-p.volume/maxVolume*52} width={Math.max(1,bw-2)} height={p.volume/maxVolume*52} fill="#38bdf8" opacity=".55"/>:null;})}
   <text x="58" y="223" textAnchor="end" fill="var(--ink-faint)" fontSize="10">{maxVolume>=1e6?(maxVolume/1e6).toFixed(1)+'M':maxVolume>=1000?(maxVolume/1000).toFixed(1)+'K':maxVolume.toFixed(0)}</text>
   <text x="66" y="290" fill="#c4b5fd" fontSize="12">Social posts · observed sample</text>
   {days.map((d,i)=>{const n=counts.get(d)??0;return n>0||searched.has(d)?<rect key={d} x={66+i*bw+1} y={355-n/maxPosts*52} width={Math.max(1,bw-2)} height={n?Math.max(1,n/maxPosts*52):1} fill="#a78bfa" opacity=".85"/>:<path key={d} d={`M${x(i)-2},350h4`} stroke="#fbbf24" opacity=".6"/>;})}
   <text x="58" y="314" textAnchor="end" fill="var(--ink-faint)" fontSize="10">{maxPosts}</text>
   {[0,.25,.5,.75,1].map(f=>{const i=Math.round(f*(days.length-1));return <text key={f} x={x(i)} y="382" textAnchor={f===0?'start':f===1?'end':'middle'} fill="var(--ink-faint)" fontSize="11">{days[i]?.slice(5)}</text>;})}
   {hover!=null&&<path d={`M${x(hover)},26V355`} stroke="#e2e8f0" strokeDasharray="3 4" opacity=".6"/>}
   <rect x="66" y="26" width="864" height="329" fill="transparent" style={{cursor:'crosshair'}}
    onPointerDown={e=>{const i=index(e);setAnchor(i);setHover(i);e.currentTarget.setPointerCapture(e.pointerId);}}
    onPointerMove={e=>setHover(index(e))} onPointerLeave={()=>{if(anchor==null)setHover(null);}}
    onPointerCancel={()=>{setAnchor(null);setHover(null);}}
    onPointerUp={e=>{if(anchor!=null){const i=index(e);onRange(days[Math.min(i,anchor)],days[Math.max(i,anchor)]);setAnchor(null);}}}/>
  </svg></div>
  <p className="min-h-5 text-xs text-[color:var(--ink-faint)]">{selected?`${selected} UTC · Price ${point?priceLabel(point.close)+(point.complete===false?' (incomplete day)':''):'unavailable'} · ${counts.has(selected)?counts.get(selected)+' observed posts':searched.has(selected)?'0 returned posts':'Social activity not searched'}`:'Daily UTC observations. Missing history remains blank; unequal search coverage limits comparisons.'}</p>
 </div>;
}
