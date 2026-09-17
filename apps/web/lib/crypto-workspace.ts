// Route helpers for the research workspace. Each destination is a real path; the coin page keeps its
// selection (dates, inspected day, marked account) in the query string.
import type {Route} from 'next';
import {isCoin} from './crypto-coins.ts';
export const BASE='/market/crypto';
export const paths={signals:BASE as Route,people:`${BASE}/people` as Route,coin:(coin:string)=>`${BASE}/coins/${encodeURIComponent(coin)}` as Route,account:(id:string)=>`${BASE}/accounts/${encodeURIComponent(id)}` as Route,data:`${BASE}/data` as Route};
export type CoinQuery={from:string|null;to:string|null;day:string|null;highlight:string|null};
const isDay=(v:string|null)=>!!v&&/^\d{4}-\d{2}-\d{2}$/.test(v);
export function readCoinQuery(params:URLSearchParams):CoinQuery{
 return {from:isDay(params.get('from'))?params.get('from'):null,to:isDay(params.get('to'))?params.get('to'):null,day:isDay(params.get('day'))?params.get('day'):null,highlight:params.get('highlight')};
}
export function writeCoinQuery(q:CoinQuery){
 const p=new URLSearchParams();if(q.from)p.set('from',q.from);if(q.to)p.set('to',q.to);if(q.day)p.set('day',q.day);if(q.highlight)p.set('highlight',q.highlight);
 const s=p.toString();return s?'?'+s:'';
}
// Links from before the route split carried ?view=&coin=&account=. Map them onto real paths; null means nothing to redirect.
export function legacyPath(params:URLSearchParams,defaultCoin:string):Route|null{
 const view=params.get('view'),coin=isCoin(params.get('coin'))?params.get('coin')!:defaultCoin,account=params.get('account');
 if(!view&&!account)return null;
 if(account&&(view==='account'||!view))return paths.account(account);
 if(['people','reach','voices','impact'].includes(view??''))return paths.people;
 if(['coins','run','network','sentiment','posts','overview'].includes(view??'')&&params.get('coin'))return (paths.coin(coin)+writeCoinQuery(readCoinQuery(params))) as Route;
 if(view==='data')return paths.data;
 if(view==='watchers'||view==='overview')return paths.signals;
 return null;
}
