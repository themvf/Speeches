// Command-line query for the crypto workbench, plus the URL state one screen keeps. Pure so it can be tested.
import type {Route} from 'next';
import {COINS,isCoin} from './crypto-coins.ts';
import {MIN_LINKED_POSTS,postingStyles,dayOneCoins,type Leader} from './crypto-leaders.ts';
import {ringToken,RING_LABEL,type Ring,type RingResult,type CircleEntry} from './crypto-rings.ts';
export const BASE='/market/crypto';
export type Tab='people'|'posts'|'timeline'|'connections'|'data';
export const TABS:{id:Tab;label:string}[]=[{id:'people',label:'People'},{id:'posts',label:'X posts'},{id:'timeline',label:'Timeline'},{id:'connections',label:'Connections'},{id:'data',label:'Data'}];
export type Query={coin:string|null;handle:string|null;early:boolean;day:number|null;hit:number|null;posts:number|null;watcher:boolean;contract:string|null;ring:Ring|null;coins:number|null;day1:boolean;circle:boolean;unknown:string[]};
const CONTRACT=/^(0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44})$/;
export function coinForContract(address:string){const a=address.toLowerCase();return COINS.find(c=>c.address&&c.address.toLowerCase()===a)?.symbol??null;}
// Tokens: SYMBOL · @handle · contract address · early · day1 · circle · day<N · hit>0.x · posts>N · coins>N · watcher · ring:N. Anything else narrows the handle.
export function parseQuery(text:string):Query{
 const q:Query={coin:null,handle:null,early:false,day:null,hit:null,posts:null,watcher:false,contract:null,ring:null,coins:null,day1:false,circle:false,unknown:[]};
 for(const t of text.trim().split(/\s+/).filter(Boolean)){let m;
  if(t[0]==='@'){q.handle=t.slice(1).toLowerCase();continue;}
  const low=t.toLowerCase();
  if(low==='early'){q.early=true;continue;}
  if(low==='day1'||low==='d1'){q.day1=true;continue;}
  if(low==='circle'||low==='d1circle'){q.circle=true;continue;}
  if(low==='watcher'||low==='watchers'){q.watcher=true;continue;}
  const ring=ringToken(low);if(ring){q.ring=ring;continue;}
  if((m=low.match(/^day<(\d+)$/))){q.day=Number(m[1]);continue;}
  if((m=low.match(/^hit>(\d*\.?\d+)$/))){q.hit=Number(m[1]);continue;}
  if((m=low.match(/^posts>(\d+)$/))){q.posts=Number(m[1]);continue;}
  if((m=low.match(/^coins>(\d+)$/))){q.coins=Number(m[1]);continue;}
  if(CONTRACT.test(t)){q.contract=t;const c=coinForContract(t);if(c)q.coin=c;continue;}
  if(isCoin(t.toUpperCase())){q.coin=t.toUpperCase();continue;}
  if(/^[A-Za-z0-9_]{2,}$/.test(t)){q.handle=low;continue;}
  q.unknown.push(t);
 }
 return q;
}
export function matchesQuery(l:Leader,q:Query,coin:string|null,rings?:Map<string,RingResult>,circle?:Map<string,CircleEntry>){
 if(coin&&!l.coins.includes(coin))return false;
 if(q.circle&&!circle?.has(l.account_id))return false;
 if(q.ring!=null&&rings?.get(l.account_id)?.ring!==q.ring)return false;
 if(q.handle&&!l.handle.toLowerCase().includes(q.handle))return false;
 if(q.early&&!l.early_coins)return false;
 if(q.day1&&!dayOneCoins(l).length)return false;
 if(q.coins!=null&&!(l.coins.length>q.coins))return false;
 if(q.day!=null&&!l.roles.some(r=>r.early&&r.day!=null&&r.day<q.day!))return false;
 if(q.hit!=null&&!(l.episodes>=MIN_LINKED_POSTS&&(l.hit_rate??0)>q.hit))return false;
 if(q.posts!=null&&!(l.episodes>q.posts))return false;
 if(q.watcher&&!(l.watcher||postingStyles(l).includes('watcher')))return false;
 return true;
}
export function describeScope(q:Query,coin:string|null){
 return [coin??'all coins',q.handle?'@'+q.handle:null,q.early?'early':null,q.day1?'day-1':null,q.circle?'day-1 circle':null,q.coins!=null?'coins>'+q.coins:null,q.day!=null?'day<'+q.day:null,q.hit!=null?'hit>'+q.hit:null,q.posts!=null?'posts>'+q.posts:null,q.watcher?'watchers':null,q.ring!=null?`ring ${q.ring} ${RING_LABEL[q.ring]}`:null,q.contract&&!q.coin?'contract '+q.contract.slice(0,6)+'… (not tracked)':null].filter(Boolean).join(' · ');
}
// URL state: everything the screen shows is in the query string so a view is a bookmark.
export type State={coin:string|null;account:string|null;tab:Tab;q:string;day:string|null;highlight:string|null;from:string|null;to:string|null};
const isDay=(v:string|null)=>!!v&&/^\d{4}-\d{2}-\d{2}$/.test(v);
export const EMPTY:State={coin:null,account:null,tab:'people',q:'',day:null,highlight:null,from:null,to:null};
export function readState(params:URLSearchParams):State{
 const tab=params.get('tab');const coin=params.get('coin');
 return {coin:isCoin(coin?.toUpperCase())?coin!.toUpperCase():null,account:params.get('account'),tab:TABS.some(t=>t.id===tab)?tab as Tab:'people',q:params.get('q')??'',
  day:isDay(params.get('day'))?params.get('day'):null,highlight:params.get('highlight'),from:isDay(params.get('from'))?params.get('from'):null,to:isDay(params.get('to'))?params.get('to'):null};
}
export function writeState(s:State):Route{
 const p=new URLSearchParams();
 if(s.coin)p.set('coin',s.coin);if(s.account)p.set('account',s.account);if(s.tab!=='people')p.set('tab',s.tab);if(s.q)p.set('q',s.q);
 if(s.day)p.set('day',s.day);if(s.highlight)p.set('highlight',s.highlight);if(s.from)p.set('from',s.from);if(s.to)p.set('to',s.to);
 const str=p.toString();return (str?BASE+'?'+str:BASE) as Route;
}
// Links from the two earlier layouts (?view=… and /people, /coins/X, /accounts/id, /data) land on the same screen.
export function legacyState(pathname:string,params:URLSearchParams):State|null{
 const s={...EMPTY};let hit=false;
 const seg=pathname.startsWith(BASE)?pathname.slice(BASE.length).split('/').filter(Boolean):[];
 if(seg[0]==='people'){s.tab='people';hit=true;}
 else if(seg[0]==='coins'&&seg[1]){const c=decodeURIComponent(seg[1]).toUpperCase();if(isCoin(c)){s.coin=c;s.tab='timeline';hit=true;}}
 else if(seg[0]==='accounts'&&seg[1]){s.account=decodeURIComponent(seg[1]);hit=true;}
 else if(seg[0]==='data'){s.tab='data';hit=true;}
 const view=params.get('view'),coin=params.get('coin'),account=params.get('account');
 if(view||account){hit=true;if(isCoin(coin?.toUpperCase()))s.coin=coin!.toUpperCase();if(account)s.account=account;
  if(['people','reach','voices','impact'].includes(view??''))s.tab='people';else if(['coins','run','overview'].includes(view??''))s.tab='timeline';else if(view==='posts'||view==='sentiment')s.tab='posts';else if(view==='network')s.tab='connections';else if(view==='data')s.tab='data';}
 for(const k of ['from','to','day'] as const){const v=params.get(k);if(isDay(v)){s[k]=v;hit=true;}}
 if(params.get('highlight')){s.highlight=params.get('highlight');hit=true;}
 return hit?s:null;
}
