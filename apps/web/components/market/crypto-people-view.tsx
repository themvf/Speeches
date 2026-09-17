"use client";
import {useEffect,useMemo,useState} from 'react';
import {pct} from '@/lib/crypto-impact';
import {whyLeader,type Leader} from '@/lib/crypto-leaders';
import {COIN_SYMBOLS} from '@/lib/crypto-coins';
import styles from './crypto-research.module.css';
type Row=Leader&{last_at:string|null;posts:number};
const compact=(n:number|null)=>n==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n)+' followers';
const confidence=(r:Row)=>r.episodes>=10&&r.coins.length>=2?'Repeated across coins':r.episodes>=10?'Repeated evidence':r.episodes>=3?'Limited sample':r.roles.length?'Roles only':'Single observations';
const roleText=(r:Row)=>{const parts=new Set<string>();for(const x of r.roles){if(x.early)parts.add('Early discoverer');if(x.analysis)parts.add('Original analysis');if(x.amplifier)parts.add('Amplifier');if(x.role==='Project / platform')parts.add('Project / platform');if(x.role==='Reporting feeds')parts.add('Reporting feed');}if(r.watcher)parts.add(r.watcher);return [...parts].join(' · ')||'—';};
export function CryptoPeopleView({initialCoin,onAccount}:{initialCoin:string|null;onAccount:(id:string)=>void}){
 const [rows,setRows]=useState<Row[]|null>(null),[error,setError]=useState('');
 const [coin,setCoin]=useState(initialCoin??'ALL'),[rank,setRank]=useState('early'),[role,setRole]=useState('any'),[minEp,setMinEp]=useState(0),[audience,setAudience]=useState(0),[active,setActive]=useState(0),[search,setSearch]=useState('');
 useEffect(()=>{const c=new AbortController();fetch('/api/market/crypto/leaders?all=1',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setRows(b.data.leaders);}).catch(()=>{if(!c.signal.aborted)setError('People could not be loaded.');});return()=>c.abort();},[]);
 const shown=useMemo(()=>{const since=active?Date.now()-active*86400000:0;
  return (rows??[]).filter(r=>(coin==='ALL'||r.coins.includes(coin))&&r.episodes>=minEp&&(r.followers??0)>=audience&&(!since||(r.last_at&&Date.parse(r.last_at)>=since))&&r.handle.toLowerCase().includes(search.toLowerCase())
   &&(role==='any'||(role==='early'&&r.early_coins>0)||(role==='analysis'&&r.roles.some(x=>x.analysis))||(role==='amplifier'&&r.roles.some(x=>x.amplifier))||(role==='watcher'&&!!r.watcher)||(role==='feed'&&r.roles.some(x=>x.role==='Reporting feeds'))||(role==='project'&&r.roles.some(x=>x.role==='Project / platform'))))
   .sort((a,b)=>rank==='excess'?(b.median_excess_24h??-Infinity)-(a.median_excess_24h??-Infinity)||b.episodes-a.episodes:rank==='episodes'?b.episodes-a.episodes:rank==='audience'?(b.followers??-1)-(a.followers??-1):rank==='recent'?String(b.last_at??'').localeCompare(String(a.last_at??'')):b.early_coins-a.early_coins||b.coins.length-a.coins.length||(b.median_excess_24h??-Infinity)-(a.median_excess_24h??-Infinity)||b.episodes-a.episodes);},[rows,coin,rank,role,minEp,audience,active,search]);
 return <div style={{display:'flex',flexDirection:'column',gap:18}}>
  <div className={styles.wsTitle}><h2>Who has a track record?</h2><p>One list. Every account with saved evidence, ranked by how early they were and what happened after they posted, across coins.</p></div>
  <div className={styles.wsChips} role="group" aria-label="Coin">{['ALL',...COIN_SYMBOLS].map(s=><button key={s} aria-pressed={coin===s} onClick={()=>setCoin(s)}>{s==='ALL'?'All coins':s}</button>)}</div>
  <div className={styles.wsFilters}>
   <label>Rank by<select value={rank} onChange={e=>setRank(e.target.value)}><option value="early">Early on most coins</option><option value="excess">Best +24h vs drift</option><option value="episodes">Most price-linked episodes</option><option value="audience">Largest audience</option><option value="recent">Most recently active</option></select></label>
   <label>Role<select value={role} onChange={e=>setRole(e.target.value)}><option value="any">Any role</option><option value="early">Early discoverer</option><option value="analysis">Original analysis</option><option value="amplifier">Amplifier</option><option value="watcher">Whale &amp; volume watcher</option><option value="feed">Reporting feed</option><option value="project">Project / platform</option></select></label>
   <label>Minimum episodes<select value={minEp} onChange={e=>setMinEp(Number(e.target.value))}><option value={0}>Any</option><option value={3}>3+</option><option value={5}>5+</option><option value={10}>10+</option></select></label>
   <label>Audience<select value={audience} onChange={e=>setAudience(Number(e.target.value))}><option value={0}>Any size</option><option value={10000}>10K+</option><option value={100000}>100K+</option></select></label>
   <label>Active<select value={active} onChange={e=>setActive(Number(e.target.value))}><option value={0}>Any time</option><option value={7}>Last 7 days</option><option value={30}>Last 30 days</option></select></label>
   <label>Handle<input type="search" placeholder="@handle" value={search} onChange={e=>setSearch(e.target.value)}/></label>
   <span className={styles.muted} style={{fontSize:12,paddingBottom:10}}>{rows?`${shown.length} of ${rows.length} accounts with evidence`:''}</span>
  </div>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!rows?<p role="status" className={styles.muted}>Reading saved rankings…</p>:!shown.length?<p className={styles.empty}>No accounts meet these filters. Rankings appear after the next collection run; price-linked episodes need a day of hourly candles.</p>:
  <section className={styles.wsSection}><div className={styles.table}><table><thead><tr><th>Account</th><th>Coins</th><th>Early on</th><th>Episodes</th><th>+24h vs drift</th><th>Roles</th><th>Confidence</th></tr></thead><tbody>{shown.slice(0,100).map((r,i)=><tr key={r.account_id}>
   <td><div className={styles.accountIdentity}><span className={styles.rankNumber}>{i+1}</span><div><button className={styles.link} onClick={()=>onAccount(r.account_id)}>@{r.handle}</button><small>{compact(r.followers)}{r.last_at?' · active '+String(r.last_at).slice(0,10):''}</small></div></div></td>
   <td><small>{r.coins.join(' · ')}</small></td><td>{r.early_coins||'—'}</td><td>{r.episodes}</td>
   <td className={r.median_excess_24h==null?'':r.median_excess_24h>0?styles.positive:styles.negative} style={{fontWeight:600}}>{pct(r.median_excess_24h)}</td>
   <td><small>{roleText(r)}</small></td><td><span className={`${styles.wsPill} ${styles.wsPillMuted}`}>{confidence(r)}</span></td>
  </tr>)}</tbody></table></div>
  <small className={styles.muted}>Early = first seven days after a coin&#39;s first saved contract post. Episodes = an author&#39;s first post on a coin in 24 hours with a complete price window. +24h vs drift subtracts the coin&#39;s own median move. Roles are text heuristics; watchers report wallets and volume. Nothing here verifies influence or causation. {shown.length>100?`Showing 100 of ${shown.length}.`:''}</small>
  <details><summary className={styles.muted}>Where did the old views go?</summary><p className={styles.muted} style={{margin:'8px 0 0',lineHeight:1.6,fontSize:13}}>Largest audiences = rank by Largest audience. Voice roles = the Role filter. Price after posting = the +24h column with Minimum episodes. Watcher rankings = Role: Whale &amp; volume watcher. Every row opens the same account page.</p></details>
  <p className={styles.muted} style={{fontSize:12}}>Why: {shown.slice(0,3).map(r=>`@${r.handle} · ${whyLeader(r)}`).join('  |  ')}</p></section>}
 </div>;
}
