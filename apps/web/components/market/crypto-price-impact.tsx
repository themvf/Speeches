"use client";
import {useEffect,useState} from 'react';
import {rankImpact,confidence,pct,share,MIN_EPISODES,type ImpactBaseline,type ImpactRow} from '@/lib/crypto-impact';
import styles from './crypto-research.module.css';
type Data={status:string;coin:string;accounts:ImpactRow[];baselines:ImpactBaseline[];events:number;episodes:number;asOf?:string};
const compact=(n:number|null)=>n==null?'Unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n);
const tone=(v:number|null)=>v==null?'':Number(v)>0?styles.positive:Number(v)<0?styles.negative:'';
export function CryptoPriceImpact({coin,onSelect}:{coin:string;onSelect:(id:string)=>void}){
 const [scope,setScope]=useState<'coin'|'all'>('coin'),[min,setMin]=useState(MIN_EPISODES),[search,setSearch]=useState('');
 const [data,setData]=useState<Data|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0);
 const requested=scope==='all'?'ALL':coin;
 useEffect(()=>{const c=new AbortController();setData(null);setError('');fetch(`/api/market/crypto/impact?coin=${requested}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setData(b.data);}).catch(()=>{if(!c.signal.aborted)setError('Could not load price impact evidence.');});return()=>c.abort();},[requested,retry]);
 const rows=rankImpact(data?.accounts??[],min,requested).filter(r=>r.handle.toLowerCase().includes(search.toLowerCase())).slice(0,50);
 const baseline=data?.baselines.find(b=>b.coin===coin);
 return <section aria-label="Price after posting">
  <div className={styles.sectionHeading}><div><p className={styles.label}>Price linkage</p><h3>What happened after they posted?</h3><p className={styles.muted}>Median move of the pinned pool price in the 24 hours after each account’s posts, read against the coin’s own drift over the same archive.</p></div><span className={styles.savedBadge}>Saved evidence · no live queries</span></div>
  <div className={styles.rankingControls}>
   <div className={styles.audienceChips} role="group" aria-label="Scope"><button aria-pressed={scope==='coin'} onClick={()=>setScope('coin')}>{coin} only</button><button aria-pressed={scope==='all'} onClick={()=>setScope('all')}>All tracked coins</button></div>
   <div className={styles.rankingSearch}><input type="search" aria-label="Search accounts by handle" placeholder="Search accounts" value={search} onChange={e=>setSearch(e.target.value)}/><select aria-label="Minimum episodes" value={min} onChange={e=>setMin(Number(e.target.value))}><option value={1}>1+ episodes</option><option value={3}>3+ episodes</option><option value={5}>5+ episodes</option><option value={10}>10+ episodes</option></select></div>
  </div>
  {error?<p role="alert">{error} <button className={styles.control} onClick={()=>setRetry(n=>n+1)}>Retry</button></p>:!data?<p role="status">Reading the saved event study…</p>:<>
   {data.status!=='ready'?<div className={styles.empty}><h3>No price-linked posts yet</h3><p>{data.status==='not_configured'||data.status==='not_started'?'The event study table has not been created. It fills after the market-history job archives hourly candles and links saved posts.':'Hourly candles are archived, but no saved post has a complete 24-hour window yet. Rows appear one day after each post once coverage exists.'}</p></div>:<>
   <div className={styles.cards}>
    <div className={styles.card}><div className={styles.label}>Coin drift · any hour</div><div className={`${styles.value} ${tone(baseline?.median_24h??null)}`}>{pct(baseline?.median_24h)}</div><p className={styles.muted}>{baseline?`Median 24h move across ${baseline.hours.toLocaleString()} archived hours for ${coin}; ${share(baseline.share_up_24h)} of hours ended higher. Accounts are compared with this, not with zero.`:`No hourly archive for ${coin} yet.`}</p></div>
    <div className={styles.card}><div className={styles.label}>Linked posts</div><div className={styles.value}>{data.episodes.toLocaleString()}</div><p className={styles.muted}>Episodes with complete 24h price windows · {data.events.toLocaleString()} posts in total ({scope==='all'?'all coins':coin}). One episode per author per 24 hours.</p></div>
    <div className={styles.card}><div className={styles.label}>Accounts with {min}+ episodes</div><div className={styles.value}>{rankImpact(data.accounts,min,requested).length}</div><p className={styles.muted}>Ranked by median excess 24h move. Small samples move a lot with one post; the minimum-episodes control matters.</p></div>
   </div>
   {!rows.length?<div className={styles.empty}><h3>No accounts meet this threshold</h3><p>Lower the minimum episodes or widen the scope.</p></div>:<div className={`${styles.table} ${styles.reachDesktop}`}><table><thead><tr><th>Account</th><th>Episodes</th><th>+1h</th><th>+6h</th><th>+24h</th><th>vs drift</th><th>Up after</th><th>Volume ×</th><th>Evidence</th></tr></thead><tbody>{rows.map((r,i)=><tr key={r.account_id}>
    <td><div className={styles.accountIdentity}><span className={styles.rankNumber}>{i+1}</span><div><button className={styles.link} onClick={()=>onSelect(r.account_id)}>@{r.handle}</button><small className={styles.cellNote}>{compact(r.followers)} followers · {r.coins.join(', ')} · {confidence(r)}</small></div></div></td>
    <td>{r.episodes}<small className={styles.cellNote}>{r.posts} linked posts</small></td>
    <td className={tone(r.median_1h)}>{pct(r.median_1h)}</td><td className={tone(r.median_6h)}>{pct(r.median_6h)}</td><td className={tone(r.median_24h)}>{pct(r.median_24h)}</td>
    <td className={tone(r.median_excess_24h)}><strong>{pct(r.median_excess_24h)}</strong><small className={styles.cellNote}>beat drift {share(r.share_beat_24h)}</small></td>
    <td>{share(r.share_up_24h)}<small className={styles.cellNote}>best {pct(r.best_return_24h)} · worst {pct(r.worst_return_24h)}</small></td>
    <td>{r.median_volume_ratio==null?'—':Number(r.median_volume_ratio).toLocaleString(undefined,{maximumFractionDigits:2})+'×'}<small className={styles.cellNote}>24h after ÷ before</small></td>
    <td>{r.best_post_url?<a className={styles.link} href={r.best_post_url} target="_blank" rel="noreferrer">Best post ↗</a>:'—'}<small className={styles.cellNote}>{String(r.first_at).slice(0,10)} → {String(r.last_at).slice(0,10)}</small></td>
   </tr>)}</tbody></table></div>}
   </>}
  </>}
  <details className={styles.methodology}><summary>How to read these numbers</summary><p className={styles.muted}>Each saved non-repost post that mentions the coin is placed in its UTC hour on the pinned pool’s hourly candles. Returns compare that hour’s close with the close 1, 6 and 24 hours later; “vs drift” subtracts the coin’s median 24h move over every archived hour, so a rising coin does not make every poster look prescient. An episode is an author’s first post on a coin in 24 hours, so a burst during one move counts once. Volume × divides the 24 hours of volume after the post by the 24 hours before and needs at least half of those hours archived. Pool prices are one pool, not the whole market; thin pools move on small trades. Sequence is not causation: a post can follow a move, react to news, or be one of hundreds in the same hour. Missing search coverage means missing posts. Nothing here is investment advice.</p></details>
 </section>;
}
