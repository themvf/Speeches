export type RunEdge={target_id:string;target:string;kind:string};
export type RunPost={id:string;author_id:string;handle:string;text:string;url:string;posted_at:string;kind:string;edges:RunEdge[];followers?:number|null;followers_observed_at?:string|null;likes?:number|null;quotes?:number|null;reposts?:number|null;metrics_observed_at?:string|null};
export type RunDay={day:string;searched:number;windows:number;exhausted:number};
export type RunData={status:string;posts:RunPost[];days:RunDay[];total:number;limit:number;start:string;end:string};
export type MarketPoint={day:string;close:number;volume:number;complete?:boolean;observedAt?:string;fetchId?:string};
export type Pool={id:string;name:string;created:string;liquidity:number;side:'base'|'quote'};
export type RunMarket={status:string;points:MarketPoint[];pools:Pool[];selected:Pool|null;source:string;sourceUrl:string;note:string;observedAt:string;storage?:string;archiveRevisions?:number;investigation?:{focus_start:string|null;focus_end:string|null;focus_reason:string;started_at:string;status:string;max_requests:number;requests_saved:number}};
export type Person={id:string;handle:string;first:string;posts:RunPost[];incoming:RunPost[];participants:number};
export type NetworkEdge={source:string;target:string;kind:string;weight:number;evidence:string};
export function peopleIn(posts:RunPost[]):Person[]{
 const map=new Map<string,Person>();
 const ensure=(id:string,handle:string,time:string)=>{let p=map.get(id);if(!p){p={id,handle,first:time,posts:[],incoming:[],participants:0};map.set(id,p);}if(time<p.first)p.first=time;return p;};
 for(const post of posts){ensure(post.author_id,post.handle,post.posted_at).posts.push(post);for(const e of post.edges){if(e.target_id!==post.author_id)ensure(e.target_id,e.target,post.posted_at).incoming.push(post);}}
 for(const p of map.values()){p.incoming=Array.from(new Map(p.incoming.map(x=>[x.id,x])).values());p.participants=new Set(p.incoming.map(x=>x.author_id)).size;}
 return [...map.values()].sort((a,b)=>a.first.localeCompare(b.first)||b.participants-a.participants||a.id.localeCompare(b.id));
}
export function networkEdges(posts:RunPost[],kind='all'):NetworkEdge[]{
 const map=new Map<string,NetworkEdge>();
 for(const p of posts)for(const e of p.edges){if(e.target_id===p.author_id||(kind!=='all'&&e.kind!==kind))continue;const key=`${p.author_id}:${e.target_id}:${e.kind}`;const found=map.get(key);if(found)found.weight++;else map.set(key,{source:p.author_id,target:e.target_id,kind:e.kind,weight:1,evidence:p.url});}
 return [...map.values()].sort((a,b)=>b.weight-a.weight||a.source.localeCompare(b.source)||a.target.localeCompare(b.target)||a.kind.localeCompare(b.kind));
}
// Deterministic force layout; proximity represents displayed links, not common control.
export function layoutNetwork(edges:NetworkEdge[],limit=32){
 const degree=new Map<string,number>();for(const e of edges){degree.set(e.source,(degree.get(e.source)??0)+e.weight);degree.set(e.target,(degree.get(e.target)??0)+e.weight);}
 const ids=[...degree.keys()].sort((a,b)=>degree.get(b)!-degree.get(a)!||a.localeCompare(b)).slice(0,limit);
 const nodes=ids.map((id,i)=>({id,x:100+(i%6)*130,y:70+Math.floor(i/6)*75,weight:degree.get(id)!}));
 const map=new Map(nodes.map(n=>[n.id,n]));const links=edges.filter(e=>map.has(e.source)&&map.has(e.target)).slice(0,120);
 for(let k=0;k<160;k++){
  const delta=new Map(nodes.map(n=>[n.id,{x:(430-n.x)*.005,y:(230-n.y)*.005}]));
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j];const dx=a.x-b.x,dy=a.y-b.y,d2=Math.max(64,dx*dx+dy*dy);const f=1000/d2;delta.get(a.id)!.x+=dx*f;delta.get(a.id)!.y+=dy*f;delta.get(b.id)!.x-=dx*f;delta.get(b.id)!.y-=dy*f;}
  for(const e of links){const a=map.get(e.source)!,b=map.get(e.target)!;const dx=b.x-a.x,dy=b.y-a.y;const d=Math.max(1,Math.hypot(dx,dy)),f=(d-110)*.016;delta.get(a.id)!.x+=dx/d*f;delta.get(a.id)!.y+=dy/d*f;delta.get(b.id)!.x-=dx/d*f;delta.get(b.id)!.y-=dy/d*f;}
  for(const n of nodes){const d=delta.get(n.id)!;n.x=Math.max(90,Math.min(770,n.x+Math.max(-10,Math.min(10,d.x))));n.y=Math.max(40,Math.min(425,n.y+Math.max(-10,Math.min(10,d.y))));}
 }
 return {nodes,links,total:degree.size};
}
export function daysBetween(start:string,end:string){const days:string[]=[];for(let t=Date.parse(start+'T00:00:00Z');t<=Date.parse(end+'T00:00:00Z');t+=86400000)days.push(new Date(t).toISOString().slice(0,10));return days;}
export function scopePosts(posts:RunPost[],start:string,end:string){return posts.filter(p=>p.posted_at.slice(0,10)>=start&&p.posted_at.slice(0,10)<=end);}
export function largestDailyGain(points:MarketPoint[]){let best:{day:string;percent:number}|null=null;const sorted=points.filter(p=>p.complete!==false).sort((a,b)=>a.day.localeCompare(b.day));for(let i=1;i<sorted.length;i++){const a=sorted[i-1],b=sorted[i];if(a.close<=0||Date.parse(b.day)-Date.parse(a.day)!==86400000)continue;const percent=(b.close/a.close-1)*100;if(percent>0&&(!best||percent>best.percent))best={day:b.day,percent};}return best;}
export function priceLabel(n:number){return n>=100?'$'+n.toLocaleString(undefined,{maximumFractionDigits:0}):n>=.01?'$'+n.toFixed(3):'$'+n.toPrecision(3);}

export function filterEvidence(posts:RunPost[],coin:string,mode:string){
 if(mode==='all')return posts;
 if(coin==='DPONS')return posts.filter(p=>p.text.toLowerCase().includes('0x0e6d1ebb33f3b8f2d09bacf3b1a1d5c581110c33')||(mode!=='contract'&&/(^|[^a-z0-9_])dpons(?![a-z0-9_])|\bdiamond\s+pons\b/i.test(p.text)));
 if(coin==='PONS')return posts.filter(p=>p.text.toLowerCase().includes('0x39dbed3a2bd333467115de45665cc57f813c4571')||(mode!=='contract'&&(/[$#]pons(?![a-z0-9_])/i.test(p.text)||(/\bpons\b/i.test(p.text)&&/robinhood|ponsdotfamily/i.test(p.text)))));
 const contract='HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR';
 return posts.filter(p=>coin==='ZCAT'?(p.text.includes(contract)||(mode!=='contract'&&/(^|[^a-z0-9_])zcat(?![a-z0-9_])|anonymous\s+cat/i.test(p.text))):/(^|[^a-z0-9_])zcash(?![a-z0-9_])|[$#]zec(?![a-z0-9_])/i.test(p.text));
}
