// Cross-coin account summary shared by the leaders board and the account drawer. Association only.
export type CoinRole={coin:string;role:string;early:boolean;analysis:boolean;amplifier:boolean;posts:number;first:string|null};
export type CoinImpact={coin:string;episodes:number;median_24h:number|null;median_excess_24h:number|null;share_up_24h:number|null};
export type Leader={account_id:string;handle:string;followers:number|null;roles:CoinRole[];impact:CoinImpact[];watcher:string|null;
 coins:string[];early_coins:number;episodes:number;median_excess_24h:number|null};
export function summarize(accountId:string,handle:string,followers:number|null,roles:CoinRole[],impact:CoinImpact[],watcher:string|null):Leader{
 const coins=[...new Set([...roles.map(r=>r.coin),...impact.map(i=>i.coin)])].sort();
 const episodes=impact.reduce((n,i)=>n+i.episodes,0);
 // Episode-weighted median of per-coin medians keeps a one-post coin from dominating the cross-coin figure.
 const weighted=impact.filter(i=>i.median_excess_24h!=null).flatMap(i=>Array(i.episodes).fill(Number(i.median_excess_24h)) as number[]).sort((a,b)=>a-b);
 const median=weighted.length?weighted[Math.floor(weighted.length/2)]:null;
 return {account_id:accountId,handle,followers,roles,impact,watcher,coins,early_coins:roles.filter(r=>r.early).length,episodes,median_excess_24h:median};
}
export function rankLeaders(rows:Leader[]){
 return [...rows].sort((a,b)=>b.early_coins-a.early_coins||b.coins.length-a.coins.length||(b.median_excess_24h??-Infinity)-(a.median_excess_24h??-Infinity)||b.episodes-a.episodes||a.account_id.localeCompare(b.account_id));
}
export function whyLeader(l:Leader){
 const parts:string[]=[];
 if(l.early_coins)parts.push(`early on ${l.early_coins} coin${l.early_coins>1?'s':''} (${l.roles.filter(r=>r.early).map(r=>r.coin).join(', ')})`);
 const analysis=l.roles.filter(r=>r.analysis).map(r=>r.coin);if(analysis.length)parts.push(`original analysis on ${analysis.join(', ')}`);
 if(l.episodes)parts.push(`${l.episodes} price-linked episode${l.episodes>1?'s':''}`);
 if(l.watcher)parts.push(l.watcher.toLowerCase());
 return parts.join(' · ')||'saved posts only';
}
