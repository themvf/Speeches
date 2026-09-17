// Named-rule signals over saved posts and archived pool prices. Each signal names the rule that fired it;
// none claims causation. Pure functions over already-loaded rows so they can be unit tested.
import {watcherSignals} from './crypto-watchers.ts';
export type BoardCoin={symbol:string;name:string;price_now:number|null;price_24h_ago:number|null;volume_24h:number|null;volume_prev_24h:number|null;posts_24h:number;posts_prev_24h:number;authors_24h:number;watched_posting_24h:number;linked_posts:number;hourly_hours:number};
export type SignalPost={id:string;author_id:string;handle:string;text:string;posted_at:string;kind:string;url:string;coins:string[];contract:boolean;followers:number|null};
export type WatchedAccount={id:string;handle:string;reason:string};
export type Signal={at:string;coin:string;rule:'watched_contract_post'|'volume_surge'|'first_contract_post'|'attention_without_price'|'watched_trade_report';title:string;detail:string;account_id?:string;handle?:string;post_url?:string;severity:'high'|'medium'|'info'};
export const RULES:Record<Signal['rule'],string>={
 watched_contract_post:'A watched account posted the contract address',
 volume_surge:'Pool volume at least 2× the prior 24 hours with at least 30 saved posts',
 first_contract_post:'The first saved post carrying the contract address',
 attention_without_price:'At least 20 posts, the pool moved under 2%, and three accounts wrote most of them',
 watched_trade_report:'A watched account reported a trade or volume figure',
};
export const pctChange=(now:number|null,before:number|null)=>now==null||before==null||before<=0?null:now/before-1;
export const volumeRatio=(now:number|null,before:number|null)=>now==null||before==null||before<=0?null:now/before;
export function boardRows(coins:BoardCoin[]){
 return coins.map(c=>({...c,price_change:pctChange(c.price_now,c.price_24h_ago),volume_ratio:volumeRatio(c.volume_24h,c.volume_prev_24h),posts_change:c.posts_prev_24h>0?c.posts_24h/c.posts_prev_24h-1:null}))
  .sort((a,b)=>(b.volume_ratio??-1)-(a.volume_ratio??-1)||b.posts_24h-a.posts_24h||a.symbol.localeCompare(b.symbol));
}
export function evaluateSignals(input:{coins:BoardCoin[];posts:SignalPost[];watched:WatchedAccount[];firstContract:{coin:string;post_id:string;handle:string;account_id:string;posted_at:string;url:string}[];windowStart:string;topAuthorShare:{coin:string;share:number}[]}):Signal[]{
 const watched=new Map(input.watched.map(w=>[w.id,w]));const out:Signal[]=[];
 for(const c of boardRows(input.coins)){
  if(c.volume_ratio!=null&&c.volume_ratio>=2&&c.posts_24h>=30)out.push({at:input.windowStart,coin:c.symbol,rule:'volume_surge',severity:'medium',title:`Volume ${c.volume_ratio.toFixed(1)}× the prior 24h`,detail:`${c.posts_24h} saved posts from ${c.authors_24h} accounts in the same window.`});
  const share=input.topAuthorShare.find(s=>s.coin===c.symbol)?.share??0;
  if(c.posts_24h>=20&&c.price_change!=null&&Math.abs(c.price_change)<.02&&share>=.6)out.push({at:input.windowStart,coin:c.symbol,rule:'attention_without_price',severity:'info',title:'Attention without price',detail:`${c.posts_24h} posts, pool ${(c.price_change*100).toFixed(1)}%, three accounts wrote ${Math.round(share*100)}% of them.`});
 }
 for(const f of input.firstContract)if(f.posted_at>=input.windowStart)out.push({at:f.posted_at,coin:f.coin,rule:'first_contract_post',severity:'high',title:'First contract post saved',detail:`@${f.handle} posted the address.`,account_id:f.account_id,handle:f.handle,post_url:f.url});
 // Alert feeds post every few minutes: keep one trade report per account, coin and six-hour bucket, and only posts tied to a tracked coin.
 const seen=new Set<string>();
 for(const p of input.posts){
  const w=watched.get(p.author_id);if(!w||p.kind==='repost'||!p.coins.length)continue;
  const coin=p.coins[0];
  if(p.contract){const k=`c:${p.author_id}:${coin}`;if(seen.has(k))continue;seen.add(k);out.push({at:p.posted_at,coin,rule:'watched_contract_post',severity:'high',title:`@${p.handle} posted the contract`,detail:w.reason,account_id:p.author_id,handle:p.handle,post_url:p.url});}
  else if(watcherSignals(p.text).supported){const k=`t:${p.author_id}:${coin}:${p.posted_at.slice(0,11)}${Math.floor(Number(p.posted_at.slice(11,13))/6)}`;if(seen.has(k))continue;seen.add(k);out.push({at:p.posted_at,coin,rule:'watched_trade_report',severity:'medium',title:`@${p.handle} reported a trade or volume figure`,detail:w.reason,account_id:p.author_id,handle:p.handle,post_url:p.url});}
 }
 const weight={high:0,medium:1,info:2};
 return out.sort((a,b)=>weight[a.severity]-weight[b.severity]||b.at.localeCompare(a.at)||a.coin.localeCompare(b.coin)||a.rule.localeCompare(b.rule));
}
