"use client";
import {useEffect,useState} from 'react';
import {priceLabel} from '@/lib/crypto-run';
import {pct} from '@/lib/crypto-impact';
import {whyLeader,type Leader} from '@/lib/crypto-leaders';
import type {Signal,boardRows} from '@/lib/crypto-signals';
import type {CoinFinding} from '@/lib/crypto-coin-discovery';
import styles from './crypto-research.module.css';
type Board=ReturnType<typeof boardRows>[number];
type Data={status:string;hours:number;board:Board[];signals:Signal[];rules:Record<string,string>;watched:number;asOf?:string};
const tone=(v:number|null)=>v==null?'':v>0?styles.positive:v<0?styles.negative:'';
const fmtPct=(v:number|null)=>v==null?'—':`${v>0?'+':''}${(v*100).toFixed(1)}%`;
const time=(iso:string)=>iso.slice(11,16);
export function CryptoSignalsView({onCoin,onAccount,onPeople,onData}:{onCoin:(coin:string)=>void;onAccount:(id:string)=>void;onPeople:()=>void;onData:()=>void}){
 const [data,setData]=useState<Data|null>(null),[leaders,setLeaders]=useState<Leader[]|null>(null),[discovery,setDiscovery]=useState<CoinFinding[]|null>(null),[error,setError]=useState('');
 useEffect(()=>{const c=new AbortController();const from=new Date(Date.now()-7*86400000).toISOString().slice(0,10);
  const load=async(path:string)=>{const r=await fetch(path,{signal:c.signal});const b=await r.json();if(!r.ok||!b.ok)throw Error();return b.data;};
  load('/api/market/crypto/signals').then(setData).catch(()=>{if(!c.signal.aborted)setError('Signals could not be loaded.');});
  load('/api/market/crypto/leaders').then(d=>setLeaders(d.leaders)).catch(()=>setLeaders([]));
  load(`/api/market/crypto/coin-discovery?from=${from}`).then(d=>setDiscovery(d.coins)).catch(()=>setDiscovery([]));return()=>c.abort();},[]);
 const surfaced=(discovery??[]).filter(c=>!c.tracked).slice(0,6);
 return <div style={{display:'flex',flexDirection:'column',gap:20}}>
  <div className={styles.wsTitle}><h2>What moved, and did attention lead it?</h2><p>Saved X posts against archived pool prices for every tracked coin. Association only, never attribution.</p></div>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!data?<p role="status" className={styles.muted}>Loading the board…</p>:<>
  <div className={styles.wsGrid}>
   <section className={styles.wsSection}><h3>Board · sorted by 24h volume ratio</h3>
    <div className={styles.table}><table><thead><tr><th>Coin</th><th>Price 24h</th><th>Volume vs 24h before</th><th>Posts 24h</th><th>Watched accounts posting</th></tr></thead><tbody>{data.board.map(c=><tr key={c.symbol}>
     <td><button className={styles.link} onClick={()=>onCoin(c.symbol)}><strong>{c.symbol}</strong></button><small className={styles.cellNote}>{c.name}{c.price_now!=null?' · '+priceLabel(c.price_now):''}</small></td>
     <td className={tone(c.price_change)} style={{fontWeight:600}}>{c.hourly_hours?fmtPct(c.price_change):<small>no hourly archive</small>}</td>
     <td>{c.volume_ratio==null?'—':c.volume_ratio.toFixed(1)+'×'}</td>
     <td>{c.posts_24h}<small className={styles.cellNote}>{c.posts_change==null?(c.posts_24h?'new':'—'):fmtPct(c.posts_change)+' vs prior 24h'} · {c.authors_24h} accounts</small></td>
     <td>{c.watched_posting_24h?<span className={styles.wsPink} style={{fontWeight:600}}>{c.watched_posting_24h} tracked</span>:<small>none</small>}</td>
    </tr>)}</tbody></table></div>
    <small className={styles.muted}>Price and volume come from the pinned pool&#39;s hourly candles; a coin without an indexed pool shows posts only. {data.watched} watched accounts: the ten reviewed watchers plus every account with a track record.</small></section>
   <section className={styles.wsSection}><h3>Signals · last {data.hours}h</h3>
    {!data.signals.length?<p className={styles.empty}>No rule fired in this window. That is a finding, not an error.</p>:<div className={styles.wsFeed}>{data.signals.slice(0,25).map((s,i)=><article key={i}><small>{time(s.at)}</small><span><button className={`${styles.wsPill} ${s.severity==='high'?styles.wsPillPink:s.severity==='medium'?styles.wsPillAmber:''}`} onClick={()=>onCoin(s.coin)}>{s.coin||'—'}</button></span><div><strong>{s.account_id?<button className={styles.link} onClick={()=>onAccount(s.account_id!)}>{s.title}</button>:s.title}</strong><div className={styles.muted}>{s.detail}{s.post_url&&<> · <a href={s.post_url} target="_blank" rel="noreferrer">source ↗</a></>}</div></div></article>)}</div>}
    <details><summary className={styles.muted}>Rules</summary><ul className={styles.muted} style={{margin:'8px 0 0',paddingLeft:18,fontSize:12,lineHeight:1.6}}>{Object.entries(data.rules).map(([k,v])=><li key={k}>{v}</li>)}</ul></details></section>
  </div>
  <div className={styles.wsGrid}>
   <section className={styles.wsSection}><h3>Accounts with a track record · top 5 <button className={styles.link} style={{fontSize:12,fontWeight:500}} onClick={onPeople}>All people →</button></h3>
    {!leaders?<p role="status" className={styles.muted}>Reading saved rankings…</p>:!leaders.length?<p className={styles.empty}>No cross-coin evidence yet. Voice rankings appear after the next collection; price-linked episodes need a day of hourly candles.</p>:<div className={styles.table}><table><thead><tr><th>Account</th><th>Early on</th><th>Episodes</th><th>+24h vs drift</th><th>Why</th></tr></thead><tbody>{leaders.slice(0,5).map(l=><tr key={l.account_id}><td><button className={styles.link} onClick={()=>onAccount(l.account_id)}>@{l.handle}</button><small className={styles.cellNote}>{l.followers==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(l.followers)+' followers'}</small></td><td>{l.early_coins?`${l.early_coins} coin${l.early_coins>1?'s':''}`:'—'}</td><td>{l.episodes}</td><td className={tone(l.median_excess_24h)} style={{fontWeight:600}}>{pct(l.median_excess_24h)}</td><td><small>{whyLeader(l)}</small></td></tr>)}</tbody></table></div>}</section>
   <section className={styles.wsSection}><h3>New coins named by watched accounts · 7d</h3>
    {!discovery?<p role="status" className={styles.muted}>Reading watcher posts…</p>:!surfaced.length?<p className={styles.empty}>No untracked coins named by watcher accounts in the last seven days.</p>:<div className={styles.table}><table><thead><tr><th>Coin</th><th>Accounts</th><th>Identity</th><th></th></tr></thead><tbody>{surfaced.map(c=><tr key={c.key}><td><strong>{c.symbol}</strong><small className={styles.cellNote}>{c.network}</small></td><td>{c.accounts}</td><td><small>{c.identity}{c.address?<span className={`${styles.cellNote} ${styles.wsMono}`}>{c.address}</span>:null}</small></td><td><button className={styles.link} onClick={onData}>Registry →</button></td></tr>)}</tbody></table></div>}
    <small className={styles.muted}>From the ten reviewed watcher accounts. Candidates, not endorsements.</small></section>
  </div></>}
 </div>;
}
