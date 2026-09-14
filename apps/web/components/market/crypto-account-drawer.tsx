"use client";
import {useEffect,useRef} from 'react';
import type {Person} from '@/lib/crypto-run';
import type {Tracking} from '@/lib/crypto-social';
import {formatCount,growthLabel} from '@/lib/crypto-social';
import styles from './crypto-research.module.css';
export function CryptoAccountDrawer({person,tracking,onClose}:{person:Person;tracking:Tracking|null;onClose:()=>void}){
 const ref=useRef<HTMLDivElement>(null);const close=useRef(onClose);close.current=onClose;
 useEffect(()=>{const previous=document.activeElement as HTMLElement|null,overflow=document.body.style.overflow;document.body.style.overflow='hidden';ref.current?.focus();
  const key=(e:KeyboardEvent)=>{if(e.key==='Escape')close.current();if(e.key==='Tab'){const items=ref.current?.querySelectorAll<HTMLElement>('button,a[href],input,select,summary');if(!items?.length)return;const first=items[0],last=items[items.length-1];if(e.shiftKey&&(document.activeElement===first||document.activeElement===ref.current)){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}};
  document.addEventListener('keydown',key);return()=>{document.removeEventListener('keydown',key);document.body.style.overflow=overflow;previous?.focus();};},[]);
 const profile=tracking?.accounts.find(a=>a.id===person.id);const history=tracking?.history.filter(h=>h.account_id===person.id)??[];
 const savedProfile=[...person.posts].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
 const useTracked=(profile?.profile_observed_at??'')>=(savedProfile?.followers_observed_at??'');
 const followers=useTracked?profile?.followers:savedProfile?.followers;
 const observed=useTracked?profile?.profile_observed_at:savedProfile?.followers_observed_at;
 const evidence=Array.from(new Map([...person.posts,...person.incoming].map(p=>[p.id,p])).values()).sort((a,b)=>a.posted_at.localeCompare(b.posted_at));
 return <div className={styles.drawerBackdrop} onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><div ref={ref} tabIndex={-1} className={styles.drawer} role="dialog" aria-modal="true" aria-labelledby="account-title">
  <div className="flex justify-between items-start gap-4"><div><p className={styles.label}>Account evidence</p><h2 id="account-title" className="text-xl break-all">@{person.handle}</h2></div><button onClick={onClose} className={styles.control}>Close ✕</button></div>
  <p className={`${styles.muted} mt-2`}>First seen in this selection: {person.first.slice(0,16).replace('T',' ')} UTC</p>
  <p className="my-5 text-sm whitespace-pre-wrap break-words">{profile?.bio??'No bio saved for this account yet.'}</p>
  <div className={styles.drawerStats}><div><p className={styles.label}>Posts in selection</p><strong>{person.posts.length}</strong></div><div><p className={styles.label}>People interacting</p><strong>{person.participants}</strong></div><div><p className={styles.label}>Latest followers</p><strong>{formatCount(followers)}</strong></div><div><p className={styles.label}>7-day growth</p><strong className={Number(profile?.growth_7d)>0?styles.positive:Number(profile?.growth_7d)<0?styles.negative:''}>{growthLabel(profile?.growth_7d)}</strong></div></div>
  <p className={`${styles.muted} my-3`}>Profile observed {observed?.slice(0,10)??'not yet'}. Follower figures describe recent snapshots, not historical audience size.</p>
  <a href={`https://x.com/i/user/${person.id}`} target="_blank" rel="noreferrer" className={styles.link}>Open X profile ↗</a>
  <h3 className="mt-6 mb-3">Posts & amplification</h3><div className={styles.eventList}>{evidence.slice(0,30).map(p=><article key={p.id}><time>{p.posted_at.slice(0,16).replace('T',' ')} UTC</time><a href={p.url} target="_blank" rel="noreferrer" className={styles.link}>@{p.handle} · {p.kind} ↗</a><p className="text-sm whitespace-pre-wrap break-words">{p.text}</p></article>)}</div>
  {evidence.length>30&&<p className={styles.muted}>Showing the earliest 30 of {evidence.length} source posts in this selection.</p>}
  <details><summary>Follower observations ({history.length})</summary><table><thead><tr><th>Date (UTC)</th><th>Followers</th></tr></thead><tbody>{history.map(p=><tr key={p.observed_at}><td>{p.observed_at.slice(0,10)}</td><td>{p.available?formatCount(p.followers):'Unavailable'}</td></tr>)}</tbody></table></details>
 </div></div>;
}
