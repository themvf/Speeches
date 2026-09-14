"use client";
import {useMemo,useState} from 'react';
import type {RunPost} from '@/lib/crypto-run';
import {largeAccounts} from '@/lib/crypto-run-reach';
import styles from './crypto-research.module.css';
const number=(n:number|null)=>n==null?'—':n.toLocaleString(undefined,{maximumFractionDigits:1});
const compact=(n:number|null)=>n==null?'Not captured':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n);
export function CryptoLargeAccounts({posts,suggestedDay,onSelect}:{posts:RunPost[];suggestedDay:string;onSelect:(id:string)=>void}){
 const [threshold,setThreshold]=useState(10000),[search,setSearch]=useState(''),[sort,setSort]=useState('followers');
 const [move,setMove]=useState<string|null>(null),[beforeOnly,setBeforeOnly]=useState(false);
 const day=move??suggestedDay;
 const accounts=useMemo(()=>largeAccounts(posts,day),[posts,day]);
 const filtered=accounts.filter(a=>(threshold===0||(a.followers??-1)>=threshold)&&a.handle.toLowerCase().includes(search.toLowerCase())&&(!beforeOnly||(a.before??0)>0));
 if(sort==='likes')filtered.sort((a,b)=>(b.medianLikes??-1)-(a.medianLikes??-1));
 if(sort==='before')filtered.sort((a,b)=>(b.before??-1)-(a.before??-1));
 const top=accounts.find(a=>a.followers!=null),large=accounts.filter(a=>(a.followers??0)>=100000),early=large.filter(a=>(a.before??0)>0);
 const unknown=accounts.filter(a=>a.followers==null).length;
 const shown=filtered.slice(0,100);
 return <div>
  <div className={styles.sectionHeading}><div><p className={styles.label}>Audience & attention</p><h3>Who has the reach?</h3><p className={styles.muted}>Large audiences discussing this coin in your selected period.</p></div><span className={styles.savedBadge}>Saved evidence · no live queries</span></div>
  <div className={styles.reachHighlights}>
   <button disabled={!top} onClick={()=>top&&onSelect(top.id)}><span>Largest saved audience</span><strong>{compact(top?.followers??null)}</strong><small>{top?'@'+top.handle+' · inspect posts':'No follower counts captured yet'}</small></button>
   <button onClick={()=>{setThreshold(100000);setBeforeOnly(false);setSearch('');}}><span>Accounts with 100K+</span><strong>{large.length}</strong><small>Potential reach, not verified impact</small></button>
   <button disabled={!day} onClick={()=>{setThreshold(100000);setBeforeOnly(true);setSearch('');}}><span>100K+ accounts before move</span><strong>{day?early.length:'—'}</strong><small>{day?'Posted before '+day+' UTC':'Choose a comparison date below'}</small></button>
  </div>
  <div className={styles.rankingControls}>
   <div className={styles.audienceChips} role="group" aria-label="Minimum followers">{[[0,'All'],[10000,'10K+'],[100000,'100K+'],[1000000,'1M+']].map(([value,label])=><button key={value} aria-pressed={threshold===value} onClick={()=>setThreshold(Number(value))}>{label}</button>)}</div>
   <div className={styles.rankingSearch}><input type="search" aria-label="Search large accounts" placeholder="Search accounts" value={search} onChange={e=>setSearch(e.target.value)}/><select aria-label="Large account sort" value={sort} onChange={e=>setSort(e.target.value)}><option value="followers">Most followers</option><option value="likes">Most median likes</option><option value="before">Most posts before move</option></select></div>
  </div>
  <details className={styles.timingControls}><summary>Timing comparison <span>{day||'Choose a date'}{beforeOnly?' · Before move only':''}</span></summary><div className={styles.toolbar}><label>Price-move date (UTC)<input type="date" aria-label="Price move comparison date" value={day} onChange={e=>{setMove(e.target.value);if(!e.target.value)setBeforeOnly(false);}}/></label><label className={styles.checkLabel}><input type="checkbox" checked={beforeOnly} disabled={!day} onChange={e=>setBeforeOnly(e.target.checked)}/> Only accounts that posted before this day</label></div><p className={styles.muted}>Before means earlier than 00:00 UTC on this date.{move===null&&day?' The suggested date is the largest saved daily gain within your selection.':''} This compares timing, not causation.</p></details>
  <div className={styles.resultsCaption}><span><strong>{filtered.length}</strong> accounts match{beforeOnly?' · before move only':''}</span><span>{unknown>0?`${unknown} accounts lack follower counts`:'Latest saved follower counts'} · select an account for evidence</span></div>
  {shown.length?<>
   <div className={`${styles.table} ${styles.reachDesktop}`}><table><thead><tr><th>Account</th><th>Followers</th><th>First mention in selection</th><th>Before move</th><th>Median likes</th><th>Quotes + reposts</th></tr></thead><tbody>{shown.map((a,i)=><tr key={a.id}>
    <td><div className={styles.accountIdentity}><span className={styles.rankNumber}>{i+1}</span><span className={styles.accountAvatar} aria-hidden="true">{a.handle.slice(0,2).toUpperCase()}</span><div><button className={styles.link} onClick={()=>onSelect(a.id)}>@{a.handle}</button><small>{a.posts} coin posts · <a href={a.first.url} target="_blank" rel="noreferrer">first source ↗</a></small></div></div></td>
    <td><strong className={styles.followerValue}>{compact(a.followers)}</strong><small className={styles.cellNote}>{a.observed?'Observed '+a.observed.slice(0,10):'Date unavailable'}</small></td>
    <td>{a.first.posted_at.slice(0,10)}<small className={styles.cellNote}>{a.first.posted_at.slice(11,16)} UTC</small></td>
    <td><span className={(a.before??0)>0?styles.beforeBadge:styles.muted}>{a.before==null?'Choose date':a.before+' posts'}</span></td>
    <td>{number(a.medianLikes)}<small className={styles.cellNote}>{a.likeSamples}/{a.posts} measured</small></td>
    <td>{number(a.amplification)}<small className={styles.cellNote}>{a.amplificationSamples}/{a.posts} measured</small></td>
   </tr>)}</tbody></table></div>
   <div className={styles.reachMobile}>{shown.map((a,i)=><article key={a.id} className={styles.mobileAccount}><div className={styles.mobileAccountHeader}><div className={styles.accountIdentity}><span className={styles.accountAvatar} aria-hidden="true">{a.handle.slice(0,2).toUpperCase()}</span><div><span className={styles.label}>#{i+1} in results</span><button className={styles.link} onClick={()=>onSelect(a.id)}>@{a.handle}</button></div></div><div><strong className={styles.followerValue}>{compact(a.followers)}</strong><small className={styles.cellNote}>followers</small></div></div><div className={styles.mobileMetrics}><div><span>Coin posts</span><strong>{a.posts}</strong></div><div><span>Before move</span><strong>{a.before??'—'}</strong></div><div><span>Median likes</span><strong>{number(a.medianLikes)}</strong></div><div><span>Quotes + reposts</span><strong>{number(a.amplification)}</strong></div></div><p className={styles.muted}>First in selection: {a.first.posted_at.slice(0,16).replace('T',' ')} UTC<br/>Followers observed: {a.observed?.slice(0,10)??'unavailable'} · Likes measured on {a.likeSamples}/{a.posts} posts; quotes/reposts on {a.amplificationSamples}/{a.posts}.</p><button className={styles.evidenceButton} onClick={()=>onSelect(a.id)}>Inspect posts & connections <span>→</span></button></article>)}</div>
  </>:<div className={styles.empty}><h3>No accounts match this view</h3><p>Try a wider date range or lower audience threshold.</p><button className={styles.control} onClick={()=>{setThreshold(0);setSearch('');setBeforeOnly(false);}}>Show all saved accounts</button></div>}
  <p className={styles.muted}>Showing {shown.length} of {filtered.length} matches. Reach is potential audience, not proof of influence.</p>
  <details className={styles.methodology}><summary>How to read these numbers</summary><p className={styles.muted}>Followers are the latest saved counts, not audience size when an old post was published. Likes, quotes and reposts are cumulative counts when captured, including attention that may have arrived after the price move. Quotes + reposts is not unique people. A dash means not captured. Rankings respect the selected dates and evidence filter; missing search coverage can omit influential accounts.</p></details>
 </div>;
}
