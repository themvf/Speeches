"use client";
import {useEffect,useState} from 'react';
import type {CoinOverview} from '@/app/api/market/crypto/overview/route';
import type {CoinFinding} from '@/lib/crypto-coin-discovery';
import {priceLabel} from '@/lib/crypto-run';
import {pct} from '@/lib/crypto-impact';
import {whyLeader,type Leader} from '@/lib/crypto-leaders';
import styles from './crypto-research.module.css';
type Overview={status:string;coins:CoinOverview[];asOf?:string};
type Discovery={coins:CoinFinding[];loaded:number;unfinished:number;from:string;to:string};
const ago=(iso:string|null)=>{if(!iso)return 'no saved posts';const h=(Date.now()-Date.parse(iso))/3600000;return h<1?'under an hour ago':h<48?`${Math.round(h)}h ago`:`${Math.round(h/24)}d ago`;};
const change=(now:number|null,before:number|null)=>now==null||before==null||before<=0?null:(now/before-1)*100;
export function CryptoOverview({onOpen,onSelect}:{onOpen:(coin:string,view:string)=>void;onSelect:(id:string)=>void}){
 const [data,setData]=useState<Overview|null>(null),[discovery,setDiscovery]=useState<Discovery|null>(null),[leaders,setLeaders]=useState<{status:string;leaders:Leader[]}|null>(null),[error,setError]=useState('');
 useEffect(()=>{const c=new AbortController();const from=new Date(Date.now()-7*86400000).toISOString().slice(0,10);
  fetch('/api/market/crypto/overview',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setData(b.data);}).catch(()=>{if(!c.signal.aborted)setError('The overview could not be loaded.');});
  fetch('/api/market/crypto/leaders',{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok)setLeaders(b.data);}).catch(()=>{});
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
  <div className={styles.sectionHeading}><div><p className={styles.label}>Across coins</p><h3>Accounts to watch</h3><p className={styles.muted}>Accounts that were early, wrote original analysis, or have price-linked episodes on tracked coins, ranked by how many coins they were early on. Candidates for review, not proven market movers.</p></div></div>
  {!leaders?<p role="status">Reading saved rankings…</p>:!leaders.leaders.length?<p className={styles.empty}>No cross-coin evidence yet. Voice rankings appear after the next rolling collection; price-linked episodes need a day of hourly candles.</p>:<div className={styles.table}><table><thead><tr><th>Account</th><th>Coins</th><th>Early on</th><th>Episodes</th><th>+24h vs drift</th><th>Why</th></tr></thead><tbody>{leaders.leaders.slice(0,15).map((l,i)=><tr key={l.account_id}><td><div className={styles.accountIdentity}><span className={styles.rankNumber}>{i+1}</span><div><button className={styles.link} onClick={()=>onSelect(l.account_id)}>@{l.handle}</button><small>{l.followers==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(l.followers)+' followers'}</small></div></div></td><td>{l.coins.join(', ')}</td><td>{l.early_coins}</td><td>{l.episodes}</td><td className={l.median_excess_24h==null?'':l.median_excess_24h>0?styles.positive:styles.negative}>{pct(l.median_excess_24h)}</td><td className={styles.muted}>{whyLeader(l)}</td></tr>)}</tbody></table></div>}
  <div className={styles.sectionHeading}><div><p className={styles.label}>Untracked</p><h3>Coins surfacing in watcher posts this week</h3><p className={styles.muted}>Tickers and contracts named by the ten reviewed watcher accounts that are not yet tracked. Candidates to add to the registry, not endorsements.</p></div></div>
  {!discovery?<p role="status">Reading watcher posts…</p>:!surfaced.length?<p className={styles.empty}>No untracked coins named by watcher accounts in the last seven days.</p>:<div className={styles.table}><table><thead><tr><th>Coin</th><th>Identity</th><th>Accounts</th><th>Posts</th><th>Reported actions</th><th>First · last</th></tr></thead><tbody>{surfaced.map(c=><tr key={c.key}><td><strong>{c.symbol}</strong><small className={styles.cellNote}>{c.network}</small></td><td className={styles.muted}>{c.identity}{c.address?<small className={styles.cellNote}>{c.address}</small>:null}</td><td>{c.accounts}</td><td>{c.posts}</td><td>{[c.buys&&`${c.buys} buy`,c.sells&&`${c.sells} sell`,c.accumulation&&`${c.accumulation} accumulation`,c.volume&&`${c.volume} volume`,c.launches&&`${c.launches} launch`].filter(Boolean).join(' · ')||'mentions only'}</td><td>{c.first.slice(0,10)} · {c.last.slice(0,10)}</td></tr>)}</tbody></table></div>}
  <p className={styles.muted}>Watcher coverage: {discovery?.unfinished??0} search windows still unfinished this week. Reported amounts and transactions are unverified text.</p>
 </section>;
}
