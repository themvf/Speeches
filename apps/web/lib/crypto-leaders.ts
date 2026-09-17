// Cross-coin account summary shared by the leaders board, signals and the account page. Association only.
export type CoinRole={coin:string;role:string;early:boolean;analysis:boolean;amplifier:boolean;posts:number;first:string|null;
 day:number|null; // calendar day of the account's first post counted from the coin's first saved contract post (day 1 = same day); null without an anchor
 contract:boolean}; // posted the contract address itself
export type CoinImpact={coin:string;episodes:number;median_24h:number|null;median_excess_24h:number|null;share_up_24h:number|null};
export type Leader={account_id:string;handle:string;followers:number|null;roles:CoinRole[];impact:CoinImpact[];watcher:string|null;
 coins:string[];early_coins:number;episodes:number;median_excess_24h:number|null;median_24h:number|null;up:number;hit_rate:number|null};
export const MIN_LINKED_POSTS=3; // below this the price columns say nothing and the row ranks beneath any account with a sample
export function dayAfterAnchor(first:string|null,anchor:string|null){
 if(!first||!anchor)return null;const d=Math.floor((Date.parse(first)-Date.parse(anchor))/86400000);return Number.isFinite(d)?d+1:null;
}
function weightedMedian(impact:CoinImpact[],pick:(i:CoinImpact)=>number|null){
 // Episode-weighted median of per-coin medians keeps a one-post coin from dominating the cross-coin figure.
 const w=impact.filter(i=>pick(i)!=null).flatMap(i=>Array(i.episodes).fill(Number(pick(i))) as number[]).sort((a,b)=>a-b);
 return w.length?w[Math.floor(w.length/2)]:null;
}
export function summarize(accountId:string,handle:string,followers:number|null,roles:CoinRole[],impact:CoinImpact[],watcher:string|null):Leader{
 const coins=[...new Set([...roles.map(r=>r.coin),...impact.map(i=>i.coin)])].sort();
 const episodes=impact.reduce((n,i)=>n+i.episodes,0);
 const up=impact.reduce((n,i)=>n+Math.round((i.share_up_24h??0)*i.episodes),0);
 return {account_id:accountId,handle,followers,roles,impact,watcher,coins,early_coins:roles.filter(r=>r.early).length,episodes,
  median_excess_24h:weightedMedian(impact,i=>i.median_excess_24h),median_24h:weightedMedian(impact,i=>i.median_24h),up,hit_rate:episodes?up/episodes:null};
}
const sampled=(l:Leader)=>l.episodes>=MIN_LINKED_POSTS?1:0;
export function rankLeaders(rows:Leader[]){
 // Early-coin breadth first; then accounts with a real sample beat those without; then how often the pool was up; then size of the move.
 return [...rows].sort((a,b)=>b.early_coins-a.early_coins||sampled(b)-sampled(a)||(sampled(b)&&sampled(a)?(b.hit_rate??0)-(a.hit_rate??0):0)
  ||(b.median_excess_24h??-Infinity)-(a.median_excess_24h??-Infinity)||b.coins.length-a.coins.length||b.episodes-a.episodes||a.account_id.localeCompare(b.account_id));
}
export type PostingStyle='first'|'early'|'analysis'|'amplifier'|'feed'|'project'|'watcher';
export const STYLE_LABEL:Record<PostingStyle,string>={first:'first to share the contract',early:'among the first to post',analysis:'explains rather than hypes',amplifier:'mostly quotes others',feed:'alert feed',project:'project account',watcher:'reports whale trades'};
export function postingStyles(l:Leader):PostingStyle[]{
 const s=new Set<PostingStyle>();
 for(const r of l.roles){if(r.role==='Project / platform')s.add('project');else if(r.role==='Reporting feeds')s.add('feed');else{if(r.early)s.add(r.contract?'first':'early');if(r.analysis)s.add('analysis');if(r.amplifier)s.add('amplifier');}}
 if(l.watcher)s.add('watcher');
 if(s.has('analysis')||s.has('first')||s.has('early'))s.delete('amplifier');
 if(s.has('first'))s.delete('early'); // an account that explains or breaks a contract is not "mostly quoting"
 return [...s];
}
const signed=(x:number)=>`${x>0?'+':''}${Math.round(x*100)}%`;
export function afterPosting(l:Leader){
 if(l.episodes<MIN_LINKED_POSTS)return null;
 const parts=[`up ${l.up} of ${l.episodes} times`];
 if(l.median_excess_24h!=null)parts.push(`typically ${signed(l.median_excess_24h)} vs the coin's own drift a day later`);
 return parts.join(' · ');
}
export function earlyCoins(l:Leader){return l.roles.filter(r=>r.early).map(r=>({coin:r.coin,day:r.day}));}
export function whyLeader(l:Leader){
 const parts:string[]=[];
 const early=earlyCoins(l);if(early.length)parts.push(`early on ${early.map(e=>e.day?`${e.coin} day ${e.day}`:e.coin).join(', ')}`);
 const analysis=l.roles.filter(r=>r.analysis).map(r=>r.coin);if(analysis.length)parts.push(`explains ${analysis.join(', ')}`);
 const after=afterPosting(l);if(after)parts.push(after);else if(l.episodes)parts.push(`${l.episodes} price-linked post${l.episodes>1?'s':''} · too few to say`);
 if(l.watcher)parts.push(l.watcher.toLowerCase());
 return parts.join(' · ')||'saved posts only';
}
