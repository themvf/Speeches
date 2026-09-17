"use client";
import Link from 'next/link';
import {useEffect,useMemo,useState} from 'react';
import {daysBetween,scopePosts,peopleIn,largestDailyGain,priceLabel,type RunData,type RunMarket} from '@/lib/crypto-run';
import {voiceEvidence} from '@/lib/crypto-voices';
import {COINS,coinConfig} from '@/lib/crypto-coins';
import {paths,type CoinQuery} from '@/lib/crypto-workspace';
import {CryptoRunChart} from './crypto-run-chart';
import {CryptoBeforeMove} from './crypto-before-move';
import {CryptoPostBrowser} from './crypto-post-browser';
import {CryptoConnections} from './crypto-connections';
import styles from './crypto-research.module.css';
type Props={coin:string;query:CoinQuery;onQuery:(patch:Partial<CoinQuery>)=>void;onAccount:(id:string)=>void;onData:()=>void};
// One scrolling page per coin: timeline, before the move, posts, connections. No sub-tabs.
export function CryptoCoinView({coin,query,onQuery,onAccount,onData}:Props){
 const {from,to,day,highlight}=query;
 const [run,setRun]=useState<RunData|null>(null),[market,setMarket]=useState<RunMarket|null>(null),[marketLoading,setMarketLoading]=useState(true),[error,setError]=useState(''),[pool,setPool]=useState('');
 const today=new Date().toISOString().slice(0,10);const start=coinConfig(coin).archiveStart;const fromDay=from??start,toDay=to??today;
 useEffect(()=>{const c=new AbortController();setRun(null);setError('');fetch(`/api/market/crypto/run?coin=${coin}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setRun(b.data);}).catch(()=>{if(!c.signal.aborted)setError('Saved posts could not be loaded. Reload to try again.');});return()=>c.abort();},[coin]);
 useEffect(()=>{const c=new AbortController();setMarketLoading(true);fetch(`/api/market/crypto/history?coin=${coin}${pool?'&pool='+encodeURIComponent(pool):''}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();if(!c.signal.aborted){setMarket(b.data);setMarketLoading(false);}}).catch(()=>{if(!c.signal.aborted){setMarket(null);setMarketLoading(false);}});return()=>c.abort();},[coin,pool]);
 const days=useMemo(()=>daysBetween(run?.start??start,run?.end??today),[run?.start,run?.end,start,today]);
 const evidence=useMemo(()=>voiceEvidence(run?.posts??[],coin),[run,coin]);const eligible=evidence.eligible;
 const posts=useMemo(()=>scopePosts(eligible,fromDay,toDay),[eligible,fromDay,toDay]);
 const series=marketLoading?[]:market?.points??[];const gain=largestDailyGain(series.filter(p=>p.day>=fromDay&&p.day<=toDay));
 const searched=(run?.days??[]).filter(d=>d.searched>0).length,total=days.length;
 const beforeCount=useMemo(()=>{if(!day)return null;const s=Date.parse(day+'T00:00:00Z')-86400000,e=Date.parse(day+'T00:00:00Z');return new Set(eligible.filter(p=>{const t=Date.parse(p.posted_at);return t>=s&&t<e&&p.kind!=='repost';}).map(p=>p.author_id)).size;},[eligible,day]);
 const range=(a:string,b:string)=>onQuery({from:a,to:b});
 return <div style={{display:'flex',flexDirection:'column',gap:18}}>
  <nav className={styles.coinChips} aria-label="Tracked coins">{COINS.map(c=><Link key={c.symbol} href={paths.coin(c.symbol)} aria-current={c.symbol===coin?'page':undefined}>{c.symbol}</Link>)}</nav>
  <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap'}}><strong style={{fontSize:16}}>{coinConfig(coin).name}</strong><span className={styles.wsPill}>{coinConfig(coin).networkLabel}</span>{coinConfig(coin).address&&<span className={`${styles.wsPill} ${styles.wsPillMuted} ${styles.wsMono}`} style={{overflowWrap:'anywhere'}}>{coinConfig(coin).address}</span>}{market?.selected&&<span className={`${styles.wsPill} ${styles.wsPillMuted}`}>pinned pool {market.selected.id.slice(0,6)}…{market.selected.id.slice(-4)}</span>}</div>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!run?<p role="status" className={styles.muted}>Loading saved posts…</p>:<>
  {run.total>run.limit&&<p className={styles.coverageNote}>Showing the earliest {run.limit.toLocaleString()} of {run.total.toLocaleString()} saved posts.</p>}
  <div className={styles.wsStats}>
   <button className={styles.wsStat} disabled={!gain} onClick={()=>gain&&onQuery({day:gain.day})}><span>Largest daily gain</span><strong className={styles.positive}>{gain?`+${gain.percent.toLocaleString(undefined,{maximumFractionDigits:0})}% · ${gain.day.slice(5)}`:'—'}</strong><small>{gain?'tap to inspect':'needs consecutive daily closes'}</small></button>
   <div className={styles.wsStat}><span>Posts before it · 24h</span><strong>{beforeCount==null?'—':`${beforeCount} accounts`}</strong><small>{day?`before ${day}`:'choose a day on the chart'}</small></div>
   <div className={styles.wsStat}><span>Earliest contract post</span><strong>{evidence.anchor?evidence.anchor.slice(0,16).replace('T',' '):coinConfig(coin).address?'none saved yet':'native coin'}</strong><small>{evidence.anchor?'UTC · not a verified launch date':coinConfig(coin).originFrom?'origin search in progress':'—'}</small></div>
   <button className={styles.wsStat} onClick={onData}><span>Coverage</span><strong className={searched<total?styles.negative:''}>{searched} / {total} days</strong><small>{total-searched} days unsearched · Data →</small></button>
  </div>
  <section className={styles.wsSection}><h3>Price &amp; attention · {fromDay} → {toDay} {(from||to)&&<button className={styles.control} style={{marginLeft:'auto',minHeight:34,padding:'6px 10px'}} onClick={()=>onQuery({from:null,to:null})}>Full period</button>}</h3>
   {marketLoading&&<p className={styles.muted}>Loading market history…</p>}
   <CryptoRunChart days={days} market={series} posts={eligible} coverage={run.days??[]} from={fromDay} to={toDay} onRange={range} onInspect={d=>onQuery({day:d})} highlight={highlight} inspecting={day}/>
   <div className={styles.marketSource}>{!marketLoading&&market?.selected&&<label>Market source<select aria-label="Trading pool" value={market.selected.id} onChange={e=>setPool(e.target.value)}>{market.pools.map(p=><option key={p.id} value={p.id}>{p.name} · {p.created.slice(0,10)} · {p.id.slice(0,6)}</option>)}</select></label>}<div><a className={styles.link} href={market?.sourceUrl??'https://www.geckoterminal.com/'} target="_blank" rel="noreferrer">{market?.source??'Market source'} ↗</a><p className={styles.muted}>{market?.note??'Market history is unavailable; saved social evidence remains accessible.'}{market?.points?.length?` Latest close ${priceLabel(market.points[market.points.length-1].close)}.`:''}</p></div></div>
   <details><summary className={styles.muted}>Dates · {fromDay} → {toDay} · {posts.length.toLocaleString()} posts · {peopleIn(posts).filter(p=>p.posts.length).length} authors</summary><div className={styles.rangeBar}><div className="flex flex-wrap items-end gap-2"><label>From (UTC)<input type="date" value={fromDay} min={days[0]} max={toDay} onChange={e=>{if(e.target.value>=days[0]&&e.target.value<=toDay)range(e.target.value,toDay);}}/></label><span className="pb-3">→</span><label>Through (UTC)<input type="date" value={toDay} min={fromDay} max={days.at(-1)} onChange={e=>{if(e.target.value>=fromDay&&e.target.value<=days[days.length-1])range(fromDay,e.target.value);}}/></label></div></div></details>
  </section>
  {day&&<CryptoBeforeMove day={day} posts={eligible} market={series} highlight={highlight} onHighlight={h=>onQuery({highlight:h})} onSelect={onAccount} onClose={()=>onQuery({day:null,highlight:null})}/>}
  <section className={styles.flatSection}><h3 style={{margin:'0 0 4px'}}>Posts</h3><p className={styles.muted} style={{margin:'0 0 8px',fontSize:13}}>Every saved post that mentions {coin}. Filters are in the sheet; sentiment lives under Insights below the list.</p><CryptoPostBrowser key={coin} coin={coin} run={run}/></section>
  <details className={styles.flatSection}><summary className={styles.muted} style={{cursor:'pointer'}}>Connections in this period · who quotes, replies to and mentions whom</summary><div style={{marginTop:8}}><CryptoConnections posts={posts} onSelect={onAccount}/></div></details>
  </>}
 </div>;
}
