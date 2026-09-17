"use client";
import Link from 'next/link';
import {usePathname,useRouter} from 'next/navigation';
import {useEffect,useRef,useState,type ReactNode} from 'react';
import type {Route} from 'next';
import {paths} from '@/lib/crypto-workspace';
import styles from './crypto-research.module.css';
type SearchResult={coins:{symbol:string;name:string}[];accounts:{id:string;handle:string;followers:number|null}[];untrackedContract:string|null};
const ICONS={
 signals:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>,
 people:<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><circle cx="17" cy="9" r="2.5"/><path d="M15.5 14.5a5 5 0 0 1 6 5"/></svg>,
 coins:<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v10M9.5 9.5h3.5a1.75 1.75 0 0 1 0 3.5h-2a1.75 1.75 0 0 0 0 3.5h4"/></svg>,
 data:<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>,
};
const NAV=[
 {id:'signals',href:paths.signals,label:'Signals',purpose:'What moved',match:(p:string)=>p===paths.signals||p===paths.signals+'/'},
 {id:'people',href:paths.people,label:'People',purpose:'Who has a record',match:(p:string)=>p.startsWith(paths.people)||p.startsWith('/market/crypto/accounts')},
 {id:'coins',href:'/market/crypto/coins' as Route,label:'Coins',purpose:'Investigate a move',match:(p:string)=>p.startsWith('/market/crypto/coins')},
 {id:'data',href:paths.data,label:'Data',purpose:'Coverage and credits',match:(p:string)=>p.startsWith(paths.data)},
] as const;
export function CryptoShell({children}:{children:ReactNode}){
 const pathname=usePathname()??'';const router=useRouter();
 const [query,setQuery]=useState(''),[results,setResults]=useState<SearchResult|null>(null),[updated,setUpdated]=useState<string|null>(null);
 const box=useRef<HTMLDivElement>(null);
 useEffect(()=>{const c=new AbortController();fetch('/api/market/crypto/status',{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok&&b.data.lastCollection)setUpdated(String(b.data.lastCollection).slice(11,16)+' UTC');}).catch(()=>{});return()=>c.abort();},[]);
 useEffect(()=>{if(query.trim().length<2){setResults(null);return;}const c=new AbortController();const t=setTimeout(()=>{fetch(`/api/market/crypto/search?q=${encodeURIComponent(query.trim())}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok)setResults(b.data);}).catch(()=>{});},250);return()=>{clearTimeout(t);c.abort();};},[query]);
 useEffect(()=>{const close=(e:MouseEvent)=>{if(box.current&&!box.current.contains(e.target as Node))setResults(null);};document.addEventListener('mousedown',close);return()=>document.removeEventListener('mousedown',close);},[]);
 useEffect(()=>{setResults(null);setQuery('');},[pathname]);
 const pick=(r:{symbol?:string;id?:string})=>{setQuery('');setResults(null);if(r.symbol)router.push(paths.coin(r.symbol));else if(r.id)router.push(paths.account(r.id));};
 return <div className={styles.shell}>
  <nav className={styles.rail} aria-label="Crypto research">
   <div className={styles.railBrand}><h2>Crypto Research</h2><small>Saved X posts against pool prices</small></div>
   {NAV.map(n=><Link key={n.id} href={n.href} aria-current={n.match(pathname)?'page':undefined}>{ICONS[n.id]}<span><strong>{n.label}</strong><small>{n.purpose}</small></span></Link>)}
   <p className={styles.railFoot}>Research context only, not investment advice. Viewing spends no provider credits.</p>
  </nav>
  <div style={{minWidth:0}}>
   <div className={styles.shellTop}>
    <div className={styles.wsSearch} ref={box}><label style={{display:'flex',alignItems:'center',gap:8,fontSize:13}} className={styles.muted}><span>Search</span><input aria-label="Search accounts, coins or contracts" placeholder="@handle, coin or contract" value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){const first=results?.coins[0]??results?.accounts[0];if(first)pick(first);}}}/></label>
     {results&&<div className={styles.wsResults} role="listbox" aria-label="Search results">{results.coins.map(c=><button key={c.symbol} role="option" aria-selected="false" onClick={()=>pick(c)}><strong>{c.symbol}</strong> · {c.name}</button>)}{results.accounts.map(a=><button key={a.id} role="option" aria-selected="false" onClick={()=>pick(a)}>@{a.handle}<small className={styles.muted}> · {a.followers==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact'}).format(a.followers)+' followers'}</small></button>)}{results.untrackedContract&&<button role="option" aria-selected="false" onClick={()=>{setResults(null);router.push(paths.data);}}>Contract not tracked · open the registry</button>}{!results.coins.length&&!results.accounts.length&&!results.untrackedContract&&<span className={styles.muted} style={{padding:'10px 12px',fontSize:13}}>No saved coin or account matches.</span>}</div>}
    </div>
    {updated&&<span className={styles.wsUpdated}>Updated {updated}</span>}
   </div>
   {children}
  </div>
 </div>;
}
export function PageHeader({eyebrow,title,children,crumbs}:{eyebrow:string;title:string;children?:ReactNode;crumbs?:{href?:string;label:string}[]}){
 return <header className={styles.pageHeader}>{crumbs&&<nav className={styles.crumbs} aria-label="Breadcrumb" style={{marginBottom:10}}>{crumbs.map((c,i)=><span key={i} style={{display:'contents'}}>{i>0&&<span aria-hidden="true">›</span>}{c.href?<Link href={c.href as Route}>{c.label}</Link>:<span aria-current="page">{c.label}</span>}</span>)}</nav>}<span className={styles.eyebrow}>{eyebrow}</span><h2>{title}</h2>{children&&<p>{children}</p>}</header>;
}
