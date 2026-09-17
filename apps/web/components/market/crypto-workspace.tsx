"use client";
import {useCallback,useEffect,useRef,useState} from 'react';
import {useSearchParams} from 'next/navigation';
import {COINS,coinConfig} from '@/lib/crypto-coins';
import {readState,writeState,type CoinTab,type View,type WorkspaceState} from '@/lib/crypto-workspace';
import {CryptoSignalsView} from './crypto-signals-view';
import {CryptoPeopleView} from './crypto-people-view';
import {CryptoCoinView} from './crypto-coin-view';
import {CryptoAccountView} from './crypto-account-view';
import {CryptoDataView} from './crypto-data-view';
import styles from './crypto-research.module.css';
type SearchResult={coins:{symbol:string;name:string}[];accounts:{id:string;handle:string;followers:number|null}[];untrackedContract:string|null};
const NAV:[View,string][]=[['signals','Signals'],['people','People'],['coins','Coins'],['data','Data']];
export function CryptoWorkspace(){
 const params=useSearchParams();
 const [state,setState]=useState<WorkspaceState>(()=>readState(params,COINS[0].symbol));
 const [query,setQuery]=useState(''),[results,setResults]=useState<SearchResult|null>(null),[updated,setUpdated]=useState<string|null>(null);
 const searchBox=useRef<HTMLDivElement>(null);
 const today=new Date().toISOString().slice(0,10);
 useEffect(()=>{const next=window.location.pathname+writeState(state,{from:coinConfig(state.coin).archiveStart,to:today});if(next!==window.location.pathname+window.location.search)window.history.replaceState(window.history.state,'',next);},[state,today]);
 useEffect(()=>{const c=new AbortController();fetch('/api/market/crypto/status',{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok&&b.data.lastCollection)setUpdated(String(b.data.lastCollection).slice(11,16)+' UTC');}).catch(()=>{});return()=>c.abort();},[]);
 useEffect(()=>{if(query.trim().length<2){setResults(null);return;}const c=new AbortController();const t=setTimeout(()=>{fetch(`/api/market/crypto/search?q=${encodeURIComponent(query.trim())}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok)setResults(b.data);}).catch(()=>{});},250);return()=>{clearTimeout(t);c.abort();};},[query]);
 useEffect(()=>{const close=(e:MouseEvent)=>{if(searchBox.current&&!searchBox.current.contains(e.target as Node))setResults(null);};document.addEventListener('mousedown',close);return()=>document.removeEventListener('mousedown',close);},[]);
 const go=useCallback((patch:Partial<WorkspaceState>)=>setState(s=>({...s,...patch})),[]);
 const openCoin=(coin:string,tab:CoinTab='timeline')=>go({view:'coins',coin,tab,day:null,highlight:null,from:null,to:null});
 const openAccount=(id:string)=>go({view:'account',account:id});
 const pick=(r:{symbol?:string;id?:string})=>{setQuery('');setResults(null);if(r.symbol)openCoin(r.symbol);else if(r.id)openAccount(r.id);};
 return <section className={`${styles.panel} ${styles.explorer}`} aria-label="Crypto research workspace">
  <header className={styles.wsHeader}>
   <div className={styles.wsBrand}><h2>Crypto Research</h2><nav className={styles.wsNav} aria-label="Workspace">{NAV.map(([id,label])=><button key={id} aria-pressed={state.view===id||(id==='people'&&state.view==='account')} onClick={()=>go({view:id,...(id==='people'?{account:null}:{})})}>{label}</button>)}</nav></div>
   <div className={styles.wsSearch} ref={searchBox}><label style={{display:'flex',alignItems:'center',gap:8,fontSize:13}} className={styles.muted}><span>Search</span><input aria-label="Search accounts, coins or contracts" placeholder="@handle, coin or contract" value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){const first=results?.coins[0]??results?.accounts[0];if(first)pick(first);}}}/></label>{updated&&<span className={styles.wsUpdated}>Updated {updated}</span>}
    {results&&<div className={styles.wsResults} role="listbox" aria-label="Search results">{results.coins.map(c=><button key={c.symbol} role="option" aria-selected="false" onClick={()=>pick(c)}><strong>{c.symbol}</strong> · {c.name}</button>)}{results.accounts.map(a=><button key={a.id} role="option" aria-selected="false" onClick={()=>pick(a)}>@{a.handle}<small className={styles.muted}> · {a.followers==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact'}).format(a.followers)+' followers'}</small></button>)}{results.untrackedContract&&<button role="option" aria-selected="false" onClick={()=>{setResults(null);go({view:'data'});}}>Contract not tracked · open the registry</button>}{!results.coins.length&&!results.accounts.length&&!results.untrackedContract&&<span className={styles.muted} style={{padding:'10px 12px',fontSize:13}}>No saved coin or account matches.</span>}</div>}
   </div>
  </header>
  <div style={{paddingTop:20}}>
   {state.view==='signals'&&<CryptoSignalsView onCoin={c=>openCoin(c)} onAccount={openAccount} onPeople={()=>go({view:'people'})} onData={()=>go({view:'data'})}/>}
   {state.view==='people'&&<CryptoPeopleView initialCoin={null} onAccount={openAccount}/>}
   {state.view==='coins'&&<CryptoCoinView coin={state.coin} tab={state.tab} from={state.from} to={state.to} day={state.day} highlight={state.highlight} onChange={p=>go(p)} onAccount={openAccount} onData={()=>go({view:'data'})}/>}
   {state.view==='account'&&state.account&&<CryptoAccountView accountId={state.account} onBack={()=>go({view:'people',account:null})} onCoin={c=>openCoin(c)} onMark={(coin,id)=>go({view:'coins',coin,tab:'timeline',highlight:id,day:null,from:null,to:null})}/>}
   {state.view==='account'&&!state.account&&<CryptoPeopleView initialCoin={null} onAccount={openAccount}/>}
   {state.view==='data'&&<CryptoDataView onCoin={c=>openCoin(c)}/>}
  </div>
  <p className={styles.muted} style={{marginTop:24,fontSize:12}}>Research context only, not investment advice. Viewing spends no provider credits; collection runs on schedule. Missing coverage never means silence.</p>
 </section>;
}
