"use client";
import {useEffect,useState} from 'react';
import styles from './crypto-research.module.css';
import {standing,runs,extremes,relativeActivity,byEasternHour,type HourProfile} from '@/lib/crypto-hour-profile';
type Status={status:string;coverage:{symbol:string;days_total:number;days_searched:number;unfinished:number;first_window:string|null;searched_through:string|null;origin:{pages:number;status:string;start_at:string|null;end_at:string|null}|null;hourly_hours:number;hourly_latest:string|null;source_id:string|null}[];ledgers:{name:string;used:number;ceiling:number;ends:string|null}[];registry:{symbol:string;name:string;network:string;address:string|null;archiveStart:string;originFrom:string|null;official:string[];matches:string[]}[];watchers:{id:string;handle:string}[];lastCollection:string|null;outstanding?:number;hourProfile?:HourProfile};

const METRIC:{key:'returns'|'volume_share'|'volatility';label:string;asks:string}[]=[
 {key:'returns',label:'Direction',asks:'do they rise or fall at set hours'},
 {key:'volume_share',label:'Activity',asks:'when trading happens'},
 {key:'volatility',label:'Volatility',asks:'how big the moves are'}];
const STANDING_NOTE:Record<string,string>={
 'recurring':'clears the strictest test and the shape returns in a held-out half',
 'this window only':'significant in this window, but the shape did not return',
 'recurs, unproven':'the shape returns out of sample, but a dozen coins cannot clear the strictest test',
 'none':'no time-of-day pattern','not tested':'not enough history in this window'};

function HourProfilePanel({p}:{p:HourProfile}){
 const activity=p.pooled.volume_share,rows=activity?byEasternHour(activity.table):[];
 const {quiet,busy}=activity?extremes(activity.table):{quiet:[],busy:[]};
 const peak=rows.reduce<typeof rows[number]|null>((a,r)=>r.mean!=null&&(!a||a.mean==null||r.mean>a.mean)?r:a,null);
 return <section className={styles.wsSection}>
  <h3>Time of day · {p.window_days} days to {String(p.until).slice(0,10)}</h3>
  <div className={styles.table}><table><thead><tr><th>Question</th><th>Answer</th><th>Strictest test</th><th>Holds out of sample</th></tr></thead><tbody>
   {METRIC.map(m=>{const b=p.pooled[m.key],st=standing(b);return <tr key={m.key}>
    <td><strong>{m.label}</strong><small className={styles.cellNote}>{m.asks}</small></td>
    <td className={st==='none'||st==='not tested'?styles.muted:''}>{st}<small className={styles.cellNote}>{STANDING_NOTE[st]}</small></td>
    <td>{b?.p_global?<>p&nbsp;=&nbsp;{b.p_global.coin.toFixed(2)}<small className={styles.cellNote}>{b.coins} coins, one vote each</small></>:<small>—</small>}</td>
    <td>{b?.split_half?<>r&nbsp;=&nbsp;{b.split_half.r>=0?'+':''}{b.split_half.r.toFixed(2)}<small className={styles.cellNote}>p&nbsp;=&nbsp;{b.split_half.p.toFixed(3)}</small></>:<small>—</small>}</td>
   </tr>;})}
  </tbody></table></div>
  {activity&&<><p className={styles.muted} style={{marginTop:10}}>
   Quietest {runs(quiet).join(', ')||'no hour is notably quiet'} ET · busiest {runs(busy).join(', ')||'no hour is notably busy'} ET
   {peak&&peak.mean!=null?` · peak ${peak.et_hour}:00 ET at ${relativeActivity(peak)!.toFixed(2)}× an even hour`:''}</p>
  <div className={styles.table}><table><thead><tr><th>ET</th><th>UTC</th><th>Share of the coin&rsquo;s day</th><th></th></tr></thead><tbody>
   {rows.map(r=>{const v=relativeActivity(r);return <tr key={r.et_hour}>
    <td>{r.et_hour}:00</td><td><small>{r.utc_hour}:00</small></td>
    <td>{r.mean==null?<small>no data</small>:<>{(r.mean*100).toFixed(2)}%<small className={styles.cellNote}>{v!.toFixed(2)}× even</small></>}</td>
    <td><div className={styles.wsBar}><div className={v!=null&&v>1.2?'warn':''} style={{width:Math.min(100,Math.round((v??0)*40))+'%'}}></div></div></td></tr>;})}
  </tbody></table></div></>}
  <small className={styles.muted}>Every p-value is one test across all 24 hours at once, under the null that gives each coin a single vote, so a striking hour is not a finding on its own. Activity and volatility are one phenomenon, not two: on a pool the size of a move follows the flow through it. {p.volume_excluded.length?`Left out of activity: ${p.volume_excluded.map(e=>e.coin).join(', ')} — the provider reports a rolling window, not this hour's trading.`:''} Research context only; this says when people trade, never what a price will do next.</small>
 </section>;
}
export function CryptoDataView({onCoin}:{onCoin:(coin:string)=>void}){
 const [data,setData]=useState<Status|null>(null),[error,setError]=useState('');
 useEffect(()=>{const c=new AbortController();fetch('/api/market/crypto/status',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setData(b.data);}).catch(()=>{if(!c.signal.aborted)setError('Status could not be loaded.');});return()=>c.abort();},[]);
 return <div style={{display:'flex',flexDirection:'column',gap:18}}>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!data?<p role="status" className={styles.muted}>Loading status…</p>:<>
  <div className={styles.wsGrid}>
   <section className={styles.wsSection}><h3>Coverage per coin</h3><div className={styles.table}><table><thead><tr><th>Coin</th><th>Days searched</th><th>Origin search</th><th>Hourly prices</th><th>Gaps</th></tr></thead><tbody>{data.coverage.map(c=><tr key={c.symbol}><td><button className={styles.link} onClick={()=>onCoin(c.symbol)}><strong>{c.symbol}</strong></button></td><td className={c.days_searched<c.days_total?styles.negative:''}>{c.days_searched} / {c.days_total}<small className={styles.cellNote}>{c.searched_through?'through '+String(c.searched_through).slice(0,10):'nothing searched yet'}</small></td><td>{!c.origin?<small>n/a</small>:c.origin.status==='search_exhausted'?'complete':c.origin.status==='pending'&&c.origin.pages===0?'queued':`${c.origin.pages} page${c.origin.pages===1?'':'s'} so far`}<small className={styles.cellNote}>{c.origin?.start_at?'from '+String(c.origin.start_at).slice(0,10):''}</small></td><td>{c.hourly_hours?`${c.hourly_hours.toLocaleString()} h`:<small>none yet</small>}<small className={styles.cellNote}>{c.hourly_latest?'latest '+String(c.hourly_latest).slice(0,13).replace('T',' ')+'h':c.source_id?'':'no indexed pool'}</small></td><td><small>{c.unfinished?`${c.unfinished} windows unfinished`:'—'}</small></td></tr>)}</tbody></table></div><small className={styles.muted}>Unsearched days never mean silence. Origin = one contract-only search back to the registry date for coins added after launch.</small></section>
   <section className={styles.wsSection}><h3>Credits · TwitterAPI.io, 100,000 per USD</h3><div className={styles.table}><table><thead><tr><th>Campaign</th><th>Used</th><th>Ceiling</th><th></th><th>Ends</th></tr></thead><tbody>{data.ledgers.map(l=>{const p=l.ceiling?Math.min(100,Math.round(l.used/l.ceiling*100)):0;return <tr key={l.name}><td>{l.name}</td><td>{l.used.toLocaleString()}</td><td>{l.ceiling.toLocaleString()}</td><td><div className={styles.wsBar}><div className={p>70?'warn':''} style={{width:p+'%'}}></div></div></td><td><small>{l.ends?String(l.ends).slice(0,10):'fixed'}</small></td></tr>;})}</tbody></table></div><small className={styles.muted}>Ceilings are ours; the provider balance is not visible here. Last saved request {data.lastCollection?String(data.lastCollection).slice(0,16).replace('T',' ')+' UTC':'none'}{data.outstanding?` · ${data.outstanding} request awaiting ledger review`:' · no outstanding requests'}.</small></section>
  </div>
  <section className={styles.wsSection}><h3>Registry · {data.registry.length} coins</h3><div className={styles.table}><table><thead><tr><th>Coin</th><th>Contract</th><th>Matches on</th><th>Official handles</th><th>Since</th></tr></thead><tbody>{data.registry.map(r=><tr key={r.symbol}><td><strong>{r.symbol}</strong><small className={styles.cellNote}>{r.name} · {r.network}</small></td><td><span className={styles.wsMono} style={{overflowWrap:'anywhere'}}>{r.address??'native'}</span></td><td><small>{r.matches.join(' · ')}</small></td><td><small>{r.official.length?r.official.map(h=>'@'+h).join(', '):'none configured'}</small></td><td><small>{r.archiveStart}{r.originFrom?' · origin '+r.originFrom:''}</small></td></tr>)}</tbody></table></div><small className={styles.muted}>Adding a coin is one registry entry (apps/web/lib/crypto-coins.json). Official handles mark interested sources so they never rank as independent voices. Watched accounts: {data.watchers.map(w=>'@'+w.handle).join(', ')} plus every account with a track record.</small></section>
  {data.hourProfile&&<HourProfilePanel p={data.hourProfile}/>}
  </>}
 </div>;
}
