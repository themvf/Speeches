import {filterEvidence,type RunPost} from './crypto-run.ts';
export type WatcherPost=RunPost&{coins:string[];bio?:string|null;corpus_total?:number};
const whale=/\bwhale(?:s|watch|watcher)?\b|\bsmart (?:money|wallets?)\b|\blarge (?:wallet|holder|transfer|transaction|buy|sell)\b|巨鲸|鲸鱼|聪明钱包|聪明钱|大额(?:转账|买入|卖出)/i;
const volume=/\b(?:trading|trade|buy|sell|24h|daily|hourly|high|surging|record)\s+volume\b|\bvolume\s*(?:[:：]|surge|spike|up|hit|reach|exceed|\$|\d)|\bvol\s*[:：]|交易量|成交量|交易额/i;
const amount=/(?:\$|USD\s*)\s*\d[\d,.]*(?:\s*[kmb])?|\b\d[\d,.]*\s*(?:USD|ETH|SOL|BTC|ZEC|million|billion|K|M)\b|\d[\d,.]*\s*(?:万|亿)/i;
const link=/https?:\/\/(?:[^\s/]+\.)?(?:etherscan\.io|solscan\.io|gmgn\.ai|dexscreener\.com|arkhamintelligence\.com|intel\.arkm\.com|nansen\.ai)\b/i;
const transaction=new RegExp('(?:(?:bought|sold|purchased|transferred|deposited|withdrew|buy|sell|buying|selling|holds|holding)\\b|买入|卖出|转账)[^.!?\\n]{0,60}(?:'+amount.source+')','i');

function template(text:string){return text.toLowerCase().replace(/https?:\/\/\S+|0x[a-f0-9]+|\$[a-z]+|@[\w]+/g,' ').replace(/[\d,.]+/g,'#').replace(/\s+/g,' ').trim();}
export function watcherSignals(text:string){const w=text.split(/\n\s*\n/).some(part=>whale.test(part)&&transaction.test(part)),v=text.split(/\n/).some(line=>volume.test(line)&&amount.test(line));return {whale:w,volume:v,supported:w||v,linked:link.test(text)};}

export function rankWatchers(raw:WatcherPost[],coin='ALL'){
 const normalized=raw.map(p=>({...p,posted_at:new Date(p.posted_at).toISOString(),followers_observed_at:p.followers_observed_at?new Date(p.followers_observed_at).toISOString():null}));
 const dedup=new Map<string,WatcherPost>();for(const p of normalized)if(!dedup.has(p.id))dedup.set(p.id,p);
 const posts=[...dedup.values()].filter(p=>p.kind!=='repost'&&p.coins.some(c=>(coin==='ALL'||c===coin)&&filterEvidence([p],c,'words').length));
 const groups=new Map<string,WatcherPost[]>();for(const p of posts){const a=groups.get(p.author_id)??[];a.push(p);groups.set(p.author_id,a);}
 const rows=[...groups].flatMap(([id,all])=>{
  const matched=all.map(p=>({p,s:watcherSignals(p.text)})).filter(x=>x.s.supported);
  if(!matched.length)return [];
  const days=new Set(matched.map(x=>x.p.posted_at.slice(0,10))).size,formats=new Set(matched.map(x=>template(x.p.text))).size;
  const bio=all.find(p=>p.bio)?.bio??'',bioMatch=/whale|smart money|on.chain|volume|鲸|钱包/i.test(bio);
  const specialist=bioMatch||/forensic|tracking|alerts|volume|on.chain|smart money/i.test(bio)||/watch|alert|scan|radar|tracker|bot|oracle/i.test(all[0].handle);
  if(!specialist&&matched.length<3)return [];
  const whales=matched.filter(x=>x.s.whale).length,volumes=matched.filter(x=>x.s.volume).length,linked=matched.filter(x=>x.s.linked).length;
  const latest=[...all].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
  const score=8*Math.min(matched.length,10)+12*Math.min(days,5)+6*Math.min(formats,4)+4*Math.min(linked,5)+(specialist?40:0);
  const examples=[...matched].sort((a,b)=>Number(b.s.linked)-Number(a.s.linked)||b.p.posted_at.localeCompare(a.p.posted_at)).slice(0,3).map(({p,s})=>({id:p.id,text:p.text,url:p.url,day:p.posted_at.slice(0,10),coins:p.coins,kind:s.whale&&s.volume?'Whale & volume':s.whale?'Whale activity':'Volume',linked:s.linked}));
  return [{id,handle:latest.handle,followers:latest.followers??null,followersObserved:latest.followers_observed_at??null,bio,score,
   specialist,category:whales&&volumes?'Whale & volume watcher':whales?'Whale watcher':'Volume reporter',
   confidence:matched.length>=3&&days>=2?'Repeated evidence':'Provisional',
   matched:matched.length,total:all.length,whalePosts:whales,volumePosts:volumes,days,linked,
   templated:matched.length>=3&&formats<=Math.ceil(matched.length/3),coins:[...new Set(matched.flatMap(x=>x.p.coins))].sort(),examples}];
 });
 return rows.sort((a,b)=>b.score-a.score||b.days-a.days||b.matched-a.matched||(b.followers??-1)-(a.followers??-1)||a.id.localeCompare(b.id));
}
export type Watcher=ReturnType<typeof rankWatchers>[number];
