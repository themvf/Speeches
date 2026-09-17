"use client";
import {useMemo} from 'react';
import type {MarketPoint,RunPost} from '@/lib/crypto-run';
import {priceLabel} from '@/lib/crypto-run';
import styles from './crypto-research.module.css';
const compact=(n:number|null|undefined)=>n==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n)+' followers';
// Posts in the 24 hours before a chosen UTC day, one row per account, largest audience first. Timing only, never attribution.
export function CryptoBeforeMove({day,posts,market,highlight,onHighlight,onSelect,onClose}:{day:string;posts:RunPost[];market:MarketPoint[];highlight:string|null;onHighlight:(id:string|null)=>void;onSelect:(id:string)=>void;onClose:()=>void}){
 const start=Date.parse(day+'T00:00:00Z')-86400000,end=Date.parse(day+'T00:00:00Z');
 const rows=useMemo(()=>{const by=new Map<string,{id:string;handle:string;followers:number|null;posts:RunPost[]}>();
  for(const p of posts){const t=Date.parse(p.posted_at);if(t<start||t>=end||p.kind==='repost')continue;const r=by.get(p.author_id)??{id:p.author_id,handle:p.handle,followers:p.followers??null,posts:[]};r.posts.push(p);if((p.followers??-1)>(r.followers??-1))r.followers=p.followers??null;by.set(p.author_id,r);}
  return [...by.values()].map(r=>({...r,posts:r.posts.sort((a,b)=>a.posted_at.localeCompare(b.posted_at))})).sort((a,b)=>(b.followers??-1)-(a.followers??-1)||a.posts[0].posted_at.localeCompare(b.posts[0].posted_at));},[posts,start,end]);
 const sorted=[...market].sort((a,b)=>a.day.localeCompare(b.day));const i=sorted.findIndex(p=>p.day===day);const point=i>=0?sorted[i]:null,prev=i>0?sorted[i-1]:null;
 const move=point&&prev&&prev.close>0&&Date.parse(point.day)-Date.parse(prev.day)===86400000?(point.close/prev.close-1)*100:null;
 const during=posts.filter(p=>p.posted_at.slice(0,10)===day&&p.kind!=='repost').length;
 return <section className={styles.beforeMove} aria-label={`Posts before ${day}`}>
  <div className={styles.toolbar}><div><p className={styles.label}>Before the move</p><h3>{day} UTC {move!=null?<span className={move>0?styles.positive:styles.negative}>{move>0?'+':''}{move.toFixed(1)}% daily close</span>:<span className={styles.muted}>no comparable daily close</span>}</h3><p className={styles.muted}>{rows.length} accounts posted in the 24 hours before this day ({rows.reduce((n,r)=>n+r.posts.length,0)} posts){point?` · close ${priceLabel(point.close)}`:''} · {during} posts during the day itself.</p></div><button className={styles.control} onClick={onClose}>Close ✕</button></div>
  {!rows.length?<p className={styles.empty}>No saved non-repost posts in the preceding 24 hours. Unsearched windows do not mean silence.</p>:<div className={styles.accountRows}>{rows.slice(0,40).map(r=><article key={r.id} className={highlight===r.id?styles.beforeMoveActive:''}>
   <div className={styles.mobileAccountHeader}><div><button className={styles.link} onClick={()=>onSelect(r.id)}>@{r.handle}</button><small className={styles.cellNote}>{compact(r.followers)} · {r.posts.length} post{r.posts.length>1?'s':''} · first {r.posts[0].posted_at.slice(11,16)} UTC</small></div><button className={styles.control} aria-pressed={highlight===r.id} onClick={()=>onHighlight(highlight===r.id?null:r.id)}>{highlight===r.id?'Marked on chart':'Mark on chart'}</button></div>
   <p className={styles.postExcerpt}>{r.posts[0].text}</p><a className={styles.muted} href={r.posts[0].url} target="_blank" rel="noreferrer">Source ↗</a>
  </article>)}</div>}
  {rows.length>40&&<p className={styles.muted}>Showing the 40 largest audiences of {rows.length} accounts.</p>}
  <p className={styles.muted}>Posting before a move is timing, not cause: the same 24 hours can hold news, listings and hundreds of unrelated posts, and thin pools move on small trades.</p>
 </section>;
}
