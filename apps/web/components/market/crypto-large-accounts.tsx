"use client";
import {useMemo,useState} from 'react';
import type {RunPost} from '@/lib/crypto-run';
import {largeAccounts} from '@/lib/crypto-run-reach';
import styles from './crypto-research.module.css';

const number=(n:number|null)=>n==null?'Not captured':n.toLocaleString(undefined,{maximumFractionDigits:1});
export function CryptoLargeAccounts({posts,suggestedDay,onSelect}:{posts:RunPost[];suggestedDay:string;onSelect:(id:string)=>void}){
 const [threshold,setThreshold]=useState(10000),[search,setSearch]=useState(''),[sort,setSort]=useState('followers');
 const [move,setMove]=useState<string|null>(null),[beforeOnly,setBeforeOnly]=useState(false);
 const day=move??suggestedDay;
 const accounts=useMemo(()=>largeAccounts(posts,day),[posts,day]);
 const filtered=accounts.filter(a=>(threshold===0||(a.followers??-1)>=threshold)&&a.handle.toLowerCase().includes(search.toLowerCase())&&(!beforeOnly||(a.before??0)>0));
 if(sort==='likes')filtered.sort((a,b)=>(b.medianLikes??-1)-(a.medianLikes??-1));
 if(sort==='before')filtered.sort((a,b)=>(b.before??-1)-(a.before??-1));
 const unknown=accounts.filter(a=>a.followers==null).length;
 return <div>
  <div className={styles.toolbar}><div><h3>Large accounts discussing this coin</h3><p className={styles.muted}>Start with audience size, then inspect the posts and attention they received.</p></div></div>
  <div className={styles.toolbar}>
   <label>Followers <select aria-label="Minimum followers" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}><option value={0}>Any / not captured</option><option value={10000}>10K+</option><option value={100000}>100K+</option><option value={1000000}>1M+</option></select></label>
   <label>Sort by <select aria-label="Large account sort" value={sort} onChange={e=>setSort(e.target.value)}><option value="followers">Most followers</option><option value="likes">Median likes</option><option value="before">Posts before move</option></select></label>
   <input type="search" aria-label="Search large accounts" placeholder="Find an account…" value={search} onChange={e=>setSearch(e.target.value)}/>
  </div>
  <div className={styles.toolbar}><label>Compare with price move (UTC) <input type="date" aria-label="Price move comparison date" value={day} onChange={e=>{setMove(e.target.value);if(!e.target.value)setBeforeOnly(false);}}/></label><label><input type="checkbox" style={{width:'auto',minHeight:0,marginRight:8}} checked={beforeOnly} disabled={!day} onChange={e=>setBeforeOnly(e.target.checked)}/> Posted before this day</label></div>
  <p className={styles.muted}>{day?`Before = earlier than ${day} at 00:00 UTC. ${move===null?'Suggested date is the largest saved daily gain in your selection.':''}`:'Choose a date to compare posting before a price move.'} Counts respect the date range and evidence filter above.</p>
  <p className={styles.muted}>{filtered.length} matching accounts · {unknown} with follower counts not captured</p>
  {filtered.length?<div className={styles.table}><table><thead><tr><th className="text-left">Account / evidence</th><th>Followers</th><th>First in selection</th><th>Coin posts</th><th>Before move</th><th>Median likes</th><th>Quotes + reposts</th></tr></thead><tbody>{filtered.slice(0,100).map(a=><tr key={a.id}>
   <td><button className={styles.link} onClick={()=>onSelect(a.id)}>@{a.handle}</button><br/><a className={styles.muted} href={a.first.url} target="_blank" rel="noreferrer">First saved post ↗</a></td>
   <td><strong style={{color:'#7dd3fc'}}>{number(a.followers)}</strong><br/><small className={styles.muted}>{a.observed?'Observed '+a.observed.slice(0,10):'Date unavailable'}</small></td>
   <td>{a.first.posted_at.slice(0,16).replace('T',' ')} UTC</td><td>{a.posts}</td><td>{a.before??'Choose date'}</td>
   <td>{number(a.medianLikes)}<br/><small className={styles.muted}>{a.likeSamples}/{a.posts} posts measured</small></td>
   <td>{number(a.amplification)}<br/><small className={styles.muted}>{a.amplificationSamples}/{a.posts} posts measured</small></td>
  </tr>)}</tbody></table></div>:<p className={styles.empty}>No saved accounts match these filters. Try a lower follower threshold or a wider date range.</p>}
  <p className={styles.coverageNote}>Up to 100 matching accounts. Followers are the latest saved counts, not audience size when an old post was published. Likes, quotes and reposts are cumulative counts when captured; they may have accrued after the price move. Quotes + reposts is not a count of unique people. A large audience or an early post does not establish market impact.</p>
 </div>;
}
