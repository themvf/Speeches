// Rings: who an account is posting with, for one coin over one window. Pure heuristics over saved posts,
// computed at request time so the window can be anything. Association only; a ring is a reading aid, not a verdict.
import type {RunPost} from './crypto-run.ts';
export type Ring=1|2|3|4|5|6|7|8;
export const RING_LABEL:Record<Ring,string>={1:'hub',2:'defender',3:'witness',4:'campaign',5:'early caller',6:'piggyback',7:'feed',8:'critic'};
export const RING_DESCRIPTION:Record<Ring,string>={
 1:'Replies to and relays lines from the same few accounts; the cluster that generates most of the volume',
 2:'Argues with accounts attacking the coin (vamp, copy, beta); paying a cost to defend it',
 3:'Reports receiving rewards or gains from holding; testimony rather than argument',
 4:'Tags the same outside account repeatedly to draw its attention',
 5:'Posted the contract address in the first fifth of the window',
 6:'Uses the coin to promote a different ticker',
 7:'Alert or tracker feed, or repeats one template',
 8:'Calls the coin a rug, bundle, vamp or scam',
};
export type RingResult={ring:Ring|null;tags:Ring[];posts:number;evidence:string};
const FEED_HANDLE=/alert|bot\b|scan|track|whale|signal|sniper|radar/i;
const CRITIC=/\b(rug|rugged|scam|scammer|bundl(e|ed|es)|honeypot|cabal|insider|psyop|vamp coin|dump(ed|ing)? on|jeet(ed|s)?|farm(ed|ing) you|exit liquidity|kol shill)/i;
const VAMP_WORDS=/\b(vamp|vamped|vamping|copy|copycat|beta|same play|steal(ing)? momentum|gravedanc|being compared|both (can )?exist|better meme|does it better|best coin win|only one (token|memecoin)|same pair)/i;
const WITNESS=/\b(received|receiving|earn(ed|ing|s)?|airdrop(ped|s)?|sent me|sends me|sending me|in my wallet|payout|paid out|(just )?(for|from) holding|got .{0,12}\bzec\b)/i;
const CASHTAG=/\$([A-Za-z][A-Za-z0-9]{1,12})/g;
const norm=(t:string)=>t.toLowerCase().replace(/https?:\/\/\S+/g,'').replace(/[^a-z0-9$ ]+/g,' ').split(/\s+/).filter(Boolean);
export function computeRings(posts:RunPost[],coin:string,pairedWith:string[]=[]):Map<string,RingResult>{
 const out=new Map<string,RingResult>();
 const rows=posts.filter(p=>p.kind!=='repost');if(!rows.length)return out;
 const byAuthor=new Map<string,RunPost[]>();for(const p of rows){const a=byAuthor.get(p.author_id)??[];a.push(p);byAuthor.set(p.author_id,a);}
 const handleOf=new Map<string,string>();for(const p of rows)handleOf.set(p.author_id,p.handle);
 const times=rows.map(p=>Date.parse(p.posted_at));const t0=Math.min(...times),t1=Math.max(...times);const earlyCut=t0+(t1-t0)*0.2;
 const ignore=new Set([coin.toLowerCase(),...pairedWith.map(x=>x.toLowerCase())]);
 // Pair weights: one per reply, quote or mention in either direction; a relayed line counts two. A strong tie is weight three or more (a relayed line plus one reply, or three replies).
 const pair=new Map<string,number>();const key=(a:string,b:string)=>a<b?a+'\u0000'+b:b+'\u0000'+a;
 const outside=new Map<string,number>();const tagsOutside=new Map<string,Map<string,number>>();const inDeg=new Map<string,number>();
 for(const p of rows)for(const e of p.edges){if(e.target_id===p.author_id)continue;
  if(byAuthor.has(e.target_id)){pair.set(key(p.author_id,e.target_id),(pair.get(key(p.author_id,e.target_id))??0)+1);inDeg.set(e.target_id,(inDeg.get(e.target_id)??0)+1);}
  else{outside.set(e.target_id,(outside.get(e.target_id)??0)+1);const m=tagsOutside.get(p.author_id)??new Map();m.set(e.target_id,(m.get(e.target_id)??0)+1);tagsOutside.set(p.author_id,m);}}
 const topOutside=[...outside].sort((a,b)=>b[1]-a[1])[0]?.[0]??null;
 // Relayed lines: a 5-gram used by three or more authors.
 const grams=new Map<string,Set<string>>();
 for(const p of rows){const w=norm(p.text);for(let i=0;i+5<=w.length;i++){const g=w.slice(i,i+5).join(' ');const s=grams.get(g)??new Set();s.add(p.author_id);grams.set(g,s);}}
 const relayed=new Set<string>();
 for(const [g,authors] of grams)if(authors.size>=3&&!/[0-9a-z]{30,}/.test(g)){const list=[...authors];for(const a of list){relayed.add(a);for(const b of list)if(a<b)pair.set(key(a,b),(pair.get(key(a,b))??0)+2);}}
 // Hub = the 2-core of strong ties around the most-replied author: every member keeps two or more strong partners who are also members.
 const strong=new Map<string,Set<string>>();
 for(const [k,w] of pair)if(w>=3){const [a,b]=k.split('\u0000');(strong.get(a)??strong.set(a,new Set()).get(a)!).add(b);(strong.get(b)??strong.set(b,new Set()).get(b)!).add(a);}
 const seed=[...inDeg].sort((a,b)=>b[1]-a[1])[0]?.[0];
 let hub=new Set([...strong].filter(([a,s])=>s.size>=2&&((byAuthor.get(a)?.length??0)>=2||a===seed)).map(([a])=>a));
 for(;;){const next=new Set([...hub].filter(a=>[...(strong.get(a)??[])].filter(b=>hub.has(b)).length>=2));if(next.size===hub.size)break;hub=next;}
 if(seed&&hub.size){const comp=new Set<string>();const q=[seed];comp.add(seed);while(q.length){const a=q.pop()!;for(const b of strong.get(a)??[])if(hub.has(b)&&!comp.has(b)){comp.add(b);q.push(b);}}hub=new Set([...hub].filter(a=>comp.has(a)));}
 if(hub.size<3)hub.clear();
 const ties=(a:string)=>[...(strong.get(a)??[])].filter(b=>hub.has(b)).length;
 for(const [id,ps] of byAuthor){
  const handle=handleOf.get(id)??id;const texts=ps.map(p=>p.text);const n=ps.length;
  const tags:Ring[]=[];const why:string[]=[];
  const templates=new Set(texts.map(t=>norm(t).slice(0,8).join(' ')));
  if(FEED_HANDLE.test(handle)||(n>=5&&templates.size/n<=0.4)){tags.push(7);why.push(FEED_HANDLE.test(handle)?'feed-like handle':`${n} posts, ${templates.size} distinct openings`);}
  const other=new Map<string,number>();let otherPosts=0;
  for(const t of texts){const found=new Set<string>();for(const m of t.matchAll(CASHTAG)){const s=m[1].toLowerCase();if(!ignore.has(s))found.add(s);}if(found.size){otherPosts++;for(const s of found)other.set(s,(other.get(s)??0)+1);}}
  const [topOther,topOtherN]=[...other].sort((a,b)=>b[1]-a[1])[0]??[null,0];
  if(topOther&&topOtherN>=2&&otherPosts/n>=0.5){tags.push(6);why.push(`$${topOther.toUpperCase()} in ${topOtherN} of ${n} posts`);}
  const critic=texts.filter(t=>CRITIC.test(t)).length;if(critic>=1&&critic/n>=0.5){tags.push(8);why.push(`${critic} of ${n} posts call it a rug, bundle or scam`);}
  if(hub.has(id)){tags.push(1);why.push(`${ties(id)} repeated ties inside the hub${relayed.has(id)?', relayed a line':''}`);}
  const defend=ps.filter(p=>VAMP_WORDS.test(p.text)&&!CRITIC.test(p.text)).length;
  if(defend>=1&&!tags.includes(8)){tags.push(2);why.push(`${defend} repl${defend===1?'y':'ies'} arguing the vamp question`);}
  const camp=topOutside?tagsOutside.get(id)?.get(topOutside)??0:0;if(camp>=3){tags.push(4);why.push(`tagged the same outside account ${camp} times`);}
  const wit=texts.filter(t=>WITNESS.test(t)).length;if(wit>=1&&!tags.includes(8)){tags.push(3);why.push(`${wit} post${wit===1?'':'s'} reporting rewards received`);}
  const first=Math.min(...ps.map(p=>Date.parse(p.posted_at)));const ca=ps.some(p=>/[1-9A-HJ-NP-Za-km-z]{32,44}|0x[0-9a-fA-F]{40}/.test(p.text));
  if(ca&&first<=earlyCut){tags.push(5);why.push('contract address in the first fifth of the window');}
  const order:Ring[]=[7,6,8,1,2,4,3,5];const ring=order.find(r=>tags.includes(r))??null;
  out.set(id,{ring,tags,posts:n,evidence:why.join(' · ')||'saved posts only'});
 }
 return out;
}
export function ringToken(text:string):Ring|null{const m=text.match(/^ring:?(\d)$/i);const n=m?Number(m[1]):NaN;return n>=1&&n<=8?n as Ring:null;}

// Day-1 circle: who the coin's day-1 supporters reply to, quote or tag. A peer posts on the coin too; an outside target
// does not (an account being lobbied, not a supporter). Day-1 supporters themselves are never in the circle.
export type CircleEntry={handle:string;inside:boolean;count:number;from:Map<string,number>}; // from: day-1 handle → edges
export function dayOneCircle(posts:RunPost[],dayOne:Set<string>):Map<string,CircleEntry>{
 const out=new Map<string,CircleEntry>();const authors=new Set(posts.map(p=>p.author_id));
 for(const p of posts){if(!dayOne.has(p.author_id)||p.kind==='repost')continue;
  for(const e of p.edges){if(e.target_id===p.author_id||dayOne.has(e.target_id))continue;
   const c=out.get(e.target_id)??{handle:e.target,inside:authors.has(e.target_id),count:0,from:new Map()};
   c.count++;c.from.set(p.handle,(c.from.get(p.handle)??0)+1);out.set(e.target_id,c);}}
 return out;
}
export function circleEvidence(c:CircleEntry){return `${c.inside?'peer':'outside target'} · ${c.count} repl${c.count===1?'y':'ies'}/tags from ${[...c.from].sort((a,b)=>b[1]-a[1]).slice(0,4).map(([h,n])=>'@'+h+(n>1?' ×'+n:'')).join(', ')}`;}
