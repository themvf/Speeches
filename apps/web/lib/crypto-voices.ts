import {filterEvidence,type RunPost} from './crypto-run.ts';
import {COINS,coinConfig} from './crypto-coins.ts';
export const VOICE_VERSION='voices-v1';
export const CONTRACTS:Record<string,string>=Object.fromEntries(COINS.filter(c=>c.address).map(c=>[c.symbol,c.address!]));
const official=(coin:string)=>coinConfig(coin).official;
export const ROLES=['Early discoverers','Original analysis','Amplifiers','Project / platform','Reporting feeds'] as const;
export type VoicePost=RunPost&{bio?:string|null};
const iso=(x:unknown)=>x?new Date(x as string).toISOString():null;
const format=(s:string)=>s.toLowerCase().replace(/https?:\/\/\S+|\$\w+|@[\w]+|0x[a-f0-9]+/g,' ').replace(/[\d,.]+/g,'#').replace(/\s+/g,' ').trim();
export function voiceEvidence(raw:VoicePost[],coin:string){
 const all=[...new Map(raw.map(p=>[p.id,{...p,posted_at:iso(p.posted_at)!,followers_observed_at:iso(p.followers_observed_at),metrics_observed_at:iso(p.metrics_observed_at)}])).values()].sort((a,b)=>a.posted_at.localeCompare(b.posted_at));
 const exact=filterEvidence(all,coin,'contract');const anchor=exact[0]?.posted_at??null;
 const native=!coinConfig(coin).address;const eligible=filterEvidence(all,coin,'words').filter(p=>native||filterEvidence([p],coin,'contract').length||!!anchor&&p.posted_at>=anchor); // name-collision exclusions live in the registry
 return {all,eligible,anchor,excluded:all.length-eligible.length};
}
export function rankVoices(raw:VoicePost[],coin:string,cutoff?:string){
 const native=!coinConfig(coin).address;
 const evidence=voiceEvidence(raw.filter(p=>!cutoff||new Date(p.posted_at).getTime()<Date.parse(cutoff)),coin);const posts=evidence.eligible.filter(p=>p.kind!=='repost');const groups=new Map<string,VoicePost[]>();
 for(const p of posts){const rows=groups.get(p.author_id)??[];rows.push(p);groups.set(p.author_id,rows);}
 const incoming=new Map<string,Set<string>>(),subsequent=new Map<string,Set<string>>();
 for(const p of posts)for(const e of p.edges??[]){if(e.target_id===p.author_id)continue;const a=incoming.get(e.target_id)??new Set();a.add(p.author_id);incoming.set(e.target_id,a);const first=groups.get(e.target_id)?.[0]?.posted_at;if(first&&p.posted_at>first&&['quote','repost','reply'].includes(e.kind)){const b=subsequent.get(e.target_id)??new Set();b.add(p.author_id);subsequent.set(e.target_id,b);}}
 const voices=[...groups].map(([id,ps])=>{const formats=new Set(ps.map(p=>format(p.text))),repeat=1-formats.size/ps.length,originals=ps.filter(p=>p.kind==='original');const latest=[...ps].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
 const project=official(coin).includes(latest.handle.toLowerCase());const feed=!project&&(/watch|alert|scan|tracker|bot/i.test(latest.handle)||(ps.length>=5&&repeat>=.6));
 const analysis=[...new Map(originals.filter(p=>p.text.length>=160&&/\b(because|therefore|revenue|liquidity|supply|fees|risk|mechanism|distribution|burn|volume)\b/i.test(p.text)).map(p=>[format(p.text),p])).values()];
 const first=ps[0],contractPosts=filterEvidence(ps,coin,'contract');const days=new Set(ps.map(p=>p.posted_at.slice(0,10))).size;
 const early=contractPosts[0]?.posted_at??first.posted_at;const age=evidence.anchor?(Date.parse(early)-Date.parse(evidence.anchor))/86400000:null;
 const earlyScore=!feed&&!project&&age!=null&&age>=0&&age<=7?(contractPosts.length?50:15)+Math.max(0,30-age*4)+Math.min(subsequent.get(id)?.size??0,20):0;
 const scores:Record<string,number>={'Early discoverers':earlyScore,'Original analysis':!feed&&!project&&analysis.length>0?Math.min(analysis.length,5)*12+Math.min(subsequent.get(id)?.size??0,30):0,'Amplifiers':!feed&&!project?Math.min(subsequent.get(id)?.size??0,100)+Math.min(Math.log10((latest.followers??0)+1)*3,20):0,'Project / platform':project?Math.min(incoming.get(id)?.size??0,100)+1:0,'Reporting feeds':feed?Math.min(days,7)*5+Math.min(formats.size,10):0};
 return {id,handle:latest.handle,followers:latest.followers??null,first:first.posted_at,posts:ps.length,days,originals:originals.length,analysis:analysis.length,repeatPercent:Math.round(repeat*100),interactors:incoming.get(id)?.size??0,subsequent:subsequent.get(id)?.size??0,contractPosts:contractPosts.length,identity:contractPosts.length?'Exact contract evidence':native?coinConfig(coin).name+' text evidence':'Context after contract anchor · provisional',scores,role:project?'Project / platform':feed?'Reporting feeds':analysis.length?'Original analysis':earlyScore>0?'Early discoverers':'Amplifiers',examples:[...new Map([...contractPosts.slice(0,1),...analysis.slice(0,1),first,...ps.slice(-2)].map(p=>[p.id,p])).values()].slice(0,5)};
 });
 return {version:VOICE_VERSION,anchor:evidence.anchor,loaded:raw.length,eligible:posts.length,excluded:evidence.excluded,voices};
}
export type Voice=ReturnType<typeof rankVoices>['voices'][number];
