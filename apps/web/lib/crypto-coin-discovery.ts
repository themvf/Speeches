// Deterministic extraction from saved source text. No price, identity or trade verification.
export const EXTRACTION_VERSION='watcher-coins-v1';
export type DiscoveryPost={id:string;author_id:string;handle:string;text:string;posted_at:string;kind:string;url:string};
const tracked=[
 {symbol:'ZCAT',network:'Solana',address:'HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR',name:/\banonymous cat\b/i},
 {symbol:'PONS',network:'Robinhood',address:'0x39dbed3a2bd333467115de45665cc57f813c4571',name:/\bponsdotfamily\b/i},
 {symbol:'DPONS',network:'Robinhood',address:'0x0e6d1ebb33f3b8f2d09bacf3b1a1d5c581110c33',name:/\bdiamond\s*pons\b/i},
 {symbol:'STANDARD',network:'Robinhood',address:'0x88ad8ddf1e3898412146a534538d418c6f8a9062',name:/\bthe standard reserve\b/i},
 {symbol:'ZEC',network:'Zcash',address:'',name:/\bzcash\b/i},
];
const canonical=(a:string)=>a.startsWith('0x')?a.toLowerCase():a;
const addressPattern='(?:0x[a-fA-F0-9]{40}(?![a-fA-F0-9])|[1-9A-HJ-NP-Za-km-z]{32,44}(?![1-9A-HJ-NP-Za-km-z]))';
const excluded=new Set(['USD','US','K','M','B']);
export type Reference={key:string;symbol:string;address:string|null;network:string;identity:string;tracked:boolean;action:'Buy'|'Sell'|'Accumulation'|'Volume'|'Launch'|'Mention';role:'Target'|'Context';amount:string|null;tx:string|null};
export function extractCoins(text:string):Reference[]{
 const symbols=[...new Set([...text.matchAll(/\$([a-zA-Z][a-zA-Z0-9_]{0,19})\b/g)].map(m=>m[1].toUpperCase()).filter(s=>!excluded.has(s)))];
 const launchTicker=text.match(/\bDex paid:\s*[^\n()]{1,100}\(([a-zA-Z][a-zA-Z0-9_]{0,19})\)\s*\//i)?.[1]?.toUpperCase();
 if(launchTicker&&!symbols.includes(launchTicker))symbols.push(launchTicker);
 for(const t of tracked)if(t.name.test(text)&&!symbols.includes(t.symbol))symbols.push(t.symbol);
 const targetMatches=[...text.matchAll(/\b(bought|buys?|purchased|sold|sells?|accumulated|accumulating)\s+(?:(\$[\d,.]+(?:[KMB])?)\s+(?:USD\s+)?(?:worth\s+)?of\s+)?\$([a-zA-Z][a-zA-Z0-9_]{0,19})\b/gi)];
 const targets=new Map(targetMatches.map(m=>[m[3].toUpperCase(),{action:(/^s/i.test(m[1])?'Sell':/^a/i.test(m[1])?'Accumulation':'Buy') as Reference['action'],amount:m[2]??null}]));
 // Conflicting actions for the same ticker in one post require review, not a guessed net action.
 for(const symbol of targets.keys())if(targetMatches.filter(m=>m[3].toUpperCase()===symbol).length>1)targets.delete(symbol);
 const addresses=new Set<string>();
 // Unlabelled addresses may be wallets or transaction hashes: don't call them token contracts.
 for(const m of text.matchAll(new RegExp('(?:\\bCA|\\bcontract(?: address)?|\\btoken address)\\s*[:=]\\s*('+addressPattern+')','gi')))addresses.add(canonical(m[1]));
 for(const m of text.matchAll(new RegExp('https?://(?:solscan\\.io|etherscan\\.io|basescan\\.org|bscscan\\.com)/token/('+addressPattern+')','gi')))addresses.add(canonical(m[1]));
 for(const t of tracked)if(t.address&&(t.address.startsWith('0x')?text.toLowerCase().includes(t.address):text.includes(t.address)))addresses.add(canonical(t.address));
 const explicitNetwork=/\b(?:chain|network)\s*[:=]\s*robinhood\b|\brobinhood chain\b/i.test(text)?'Robinhood':/\b(?:chain|network)\s*[:=]\s*solana\b|solscan\.io\/token\//i.test(text)?'Solana':/\b(?:chain|network)\s*[:=]\s*ethereum\b|etherscan\.io\/token\//i.test(text)?'Ethereum':/\b(?:chain|network)\s*[:=]\s*base\b|basescan\.org\/token\//i.test(text)?'Base':/\b(?:chain|network)\s*[:=]\s*bsc\b|bscscan\.com\/token\//i.test(text)?'BSC':'Unknown';
 const tx=text.match(/https?:\/\/(?:etherscan\.io|basescan\.org|bscscan\.com|solscan\.io)\/tx\/([a-zA-Z0-9]+)/i)?.[0]??null;
 const result:Reference[]=[];const assigned=new Set<string>();
 for(const address of addresses){
  const known=tracked.find(t=>t.address&&canonical(t.address)===address);
  const inferredSymbol=known?.symbol??(targets.size===1&&addresses.size===1?[...targets.keys()][0]:symbols.length===1&&addresses.size===1?symbols[0]:null);
  if(inferredSymbol)assigned.add(inferredSymbol);
  const network=known?.network??explicitNetwork;
  result.push({key:`address:${network}:${address}`,symbol:inferredSymbol??'Unresolved token',address,network,identity:known?'Tracked contract match':'Reported contract · unverified',tracked:!!known,action:'Mention',role:'Context',amount:null,tx});
 }
 for(const symbol of symbols){if(assigned.has(symbol))continue;const known=tracked.find(t=>t.symbol===symbol);
  // Ticker-only candidate groups are explicitly unresolved; never merge them into contracts.
  result.push({key:`ticker:${explicitNetwork}:${symbol}`,symbol,address:null,network:explicitNetwork,identity:'Ticker/name only · unresolved',tracked:!!known,action:'Mention',role:'Context',amount:null,tx});
 }
 for(const r of result){const target=targets.get(r.symbol);if(target){r.action=target.action;r.amount=target.amount;r.role='Target';}
  else if(result.length===1){r.role='Target';r.action=/\bvolume\b/i.test(text)&&/\d/.test(text)?'Volume':/\b(?:launch|migration|new pair|dex paid)\b/i.test(text)?'Launch':'Mention';}
 }
 return result;
}
export type CoinFinding={key:string;symbol:string;address:string|null;network:string;identity:string;tracked:boolean;posts:number;accounts:number;buys:number;sells:number;accumulation:number;volume:number;launches:number;context:number;repeatedReports:number;first:string;last:string;daily:{day:string;posts:number}[];examples:(DiscoveryPost&{action:string;role:string;amount:string|null})[]};
export function discoverCoins(raw:DiscoveryPost[]):CoinFinding[]{
 const rows=new Map<string,CoinFinding>();const authors=new Map<string,Set<string>>(),events=new Map<string,Set<string>>(),days=new Map<string,Map<string,number>>();const ids=new Set<string>();
 for(const p of raw){if(ids.has(p.id)||p.kind==='repost')continue;ids.add(p.id);
  for(const r of extractCoins(p.text)){
   // Ticker-only mentions remain candidate groups, never joined to resolved contracts.
   let c=rows.get(r.key);if(!c){c={...r,posts:0,accounts:0,buys:0,sells:0,accumulation:0,volume:0,launches:0,context:0,repeatedReports:0,first:p.posted_at,last:p.posted_at,daily:[],examples:[]};rows.set(r.key,c);authors.set(r.key,new Set());events.set(r.key,new Set());days.set(r.key,new Map());}
   c.posts++;authors.get(r.key)!.add(p.author_id);c.first=c.first<p.posted_at?c.first:p.posted_at;c.last=c.last>p.posted_at?c.last:p.posted_at;
   const day=p.posted_at.slice(0,10),d=days.get(r.key)!;d.set(day,(d.get(day)??0)+1);
   // Without a hash, only identical text on the same UTC day is collapsed as a repeated report.
   // Distinct reports are not represented as verified unique transactions.
   const fingerprint=`${r.action}:${r.tx??day+':'+p.text.replace(/\s+/g,' ').trim().toLowerCase()}`;
   const seen=events.get(r.key)!;if(seen.has(fingerprint))c.repeatedReports++;else{seen.add(fingerprint);if(r.role==='Context')c.context++;else if(r.action==='Buy')c.buys++;else if(r.action==='Sell')c.sells++;else if(r.action==='Accumulation')c.accumulation++;else if(r.action==='Volume')c.volume++;else if(r.action==='Launch')c.launches++;}
   if(c.examples.length<8)c.examples.push({...p,action:r.action,role:r.role,amount:r.amount});
  }
 }
 return [...rows.values()].map(c=>({...c,accounts:authors.get(c.key)!.size,daily:[...days.get(c.key)!].sort(([a],[b])=>a.localeCompare(b)).map(([day,posts])=>({day,posts}))})).sort((a,b)=>b.accounts-a.accounts||(b.buys+b.sells+b.accumulation+b.volume+b.launches)-(a.buys+a.sells+a.accumulation+a.volume+a.launches)||b.posts-a.posts||a.key.localeCompare(b.key));
}
