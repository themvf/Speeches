"use client";
import {useEffect,useState} from 'react';
import type {CoinOverview} from '@/app/api/market/crypto/overview/route';
import type {CoinFinding} from '@/lib/crypto-coin-discovery';
import {priceLabel} from '@/lib/crypto-run';
import styles from './crypto-research.module.css';
type Overview={status:string;coins:CoinOverview[];asOf?:string};
type Discovery={coins:CoinFinding[];loaded:number;unfinished:number;from:string;to:string};
const ago=(iso:string|null)=>{if(!iso)return 'no saved posts';const h=(Date.now()-Date.parse(iso))/3600000;return h<1?'under an hour ago':h<48?`${Math.round(h)}h ago`:`${Math.round(h/24)}d ago`;};
const change=(now:number|null,before:number|null)=>now==null||before==null||before<=0?null:(now/before-1)*100;
export function CryptoOverview({onOpen}:{onOpen:(coin:string,view:string)=>void}){
 const [data,setData]=useState<Overview|null>(null),[discovery,setDiscovery]=useState<Discovery|null>(null),[error,setError]=useState('');
 useEffect(()=>{const c=new AbortController();const from=new Date(Date.now()-7*86400000).toISOString().slice(0,10);
  fetch('/api/market/crypto/overview',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setData(b.data);}).catch(()=>{if(!c.signal.aborted)setError('The overview could not be loaded.');});
  fetch(`/api/market/crypto/coin-discovery?from=${from}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok)setDiscovery(b.data);}).catch(()=>{});return()=>c.abort();},[]);
 const surfaced=(discovery?.coins??[]).filter(c=>!c.tracked).slice(0,8);
 return <section aria-label="Crypto research overview">
  <div className={styles.sectionHeading}><div><p className={styles.label}>Start here</p><h3>What moved, who is talking</h3><p className={styles.muted}>Saved X posts and archived pool prices for every tracked coin. Pick a coin to open its posts, accounts and timing evidence.</p></div><span className={styles.savedBadge}>Saved evidence · no live queries</span></div>
  {error?<p role="alert">{error}</p>:!data?<p role="status">Loading the overview…</p>:<div className={styles.overviewGrid}>{data.coins.map(c=>{const pct=change(c.price_now,c.price_24h_ago);return <article key={c.symbol} className={styles.overviewCard}>
   <header><button className={styles.link} onClick={()=>onOpen(c.symbol,'run')}><strong>{c.symbol}</strong> · {c.name}</button><span className={styles.muted}>{c.networkLabel}</span></header>
   <div className={styles.overviewStats}>
    <div><span>Price · 24h</span><strong className={pct==null?'':pct>0?styles.positive:pct<0?styles.negative:''}>{pct==null?'—':`${pct>0?'+':''}${pct.toFixed(1)}%`}</strong><small>{c.price_now!=null?priceLabel(c.price_now)+(c.price_observed_at?' · '+ago(c.price_observed_at):''):'No hourly archive yet'}</small></div>
    <div><span>Posts · 24h</span><strong>{c.posts_24h.toLocaleString()}</strong><small>{c.authors_24h} accounts · {c.posts_7d.toLocaleString()} posts / 7d</small></div>
    <div><span>New voices · 7d</span><strong>{c.new_authors_7d}</strong><small>first saved post on this coin this week</small></div>
    <div><span>Price-linked posts</span><strong>{c.linked_posts.toLocaleString()}</strong><small>episodes with a complete 24h window</small></div>
   </div>
   <footer><span className={styles.muted}>Latest saved post {ago(c.last_post_at)} · archive from {c.archiveStart}</span><div className={styles.overviewLinks}><button className={styles.control} onClick={()=>onOpen(c.symbol,'posts')}>Posts</button><button className={styles.control} onClick={()=>onOpen(c.symbol,'impact')}>Price after posting</button><button className={styles.control} onClick={()=>onOpen(c.symbol,'voices')}>Voices</button></div></footer>
  </article>;})}</div>}
  <div className={styles.sectionHeading}><div><p className={styles.label}>Untracked</p><h3>Coins surfacing in watcher posts this week</h3><p className={styles.muted}>Tickers and contracts named by the ten reviewed watcher accounts that are not yet tracked. Candidates to add to the registry, not endorsements.</p></div></div>
  {!discovery?<p role="status">Reading watcher posts…</p>:!surfaced.length?<p className={styles.empty}>No untracked coins named by watcher accounts in the last seven days.</p>:<div className={styles.table}><table><thead><tr><th>Coin</th><th>Identity</th><th>Accounts</th><th>Posts</th><th>Reported actions</th><th>First · last</th></tr></thead><tbody>{surfaced.map(c=><tr key={c.key}><td><strong>{c.symbol}</strong><small className={styles.cellNote}>{c.network}</small></td><td className={styles.muted}>{c.identity}{c.address?<small className={styles.cellNote}>{c.address}</small>:null}</td><td>{c.accounts}</td><td>{c.posts}</td><td>{[c.buys&&`${c.buys} buy`,c.sells&&`${c.sells} sell`,c.accumulation&&`${c.accumulation} accumulation`,c.volume&&`${c.volume} volume`,c.launches&&`${c.launches} launch`].filter(Boolean).join(' · ')||'mentions only'}</td><td>{c.first.slice(0,10)} · {c.last.slice(0,10)}</td></tr>)}</tbody></table></div>}
  <p className={styles.muted}>Watcher coverage: {discovery?.unfinished??0} search windows still unfinished this week. Reported amounts and transactions are unverified text.</p>
 </section>;
}
