"use client";
import {useEffect,useRef,useState} from 'react';
import {pct,share} from '@/lib/crypto-impact';
import {whyLeader,type Leader} from '@/lib/crypto-leaders';
import type {Person} from '@/lib/crypto-run';
import type {Tracking} from '@/lib/crypto-social';
import {formatCount,growthLabel} from '@/lib/crypto-social';
import styles from './crypto-research.module.css';
type AccountData={status:string;account?:{id:string;handle:string;name:string;followers:number|null};latest_profile?:{bio:string|null;followers:number|null;observed_at:string}|null;activity?:{coin:string;posts:number;originals:number;first_at:string;last_at:string}[];posts?:{id:string;text:string;url:string;posted_at:string;kind:string;coins:string[]|null;return_24h:number|null}[];summary?:Leader|null};
export function CryptoAccountDrawer({person,accountId,tracking,onClose,onOpenCoin}:{person:Person|null;accountId:string;tracking:Tracking|null;onClose:()=>void;onOpenCoin?:(coin:string,view:string)=>void}){
 const [across,setAcross]=useState<AccountData|null>(null);
 useEffect(()=>{const c=new AbortController();setAcross(null);fetch(`/api/market/crypto/account?id=${accountId}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(r.ok&&b.ok)setAcross(b.data);}).catch(()=>{});return()=>c.abort();},[accountId]);
 const ref=useRef<HTMLDivElement>(null);const close=useRef(onClose);close.current=onClose;
 useEffect(()=>{const previous=document.activeElement as HTMLElement|null,overflow=document.body.style.overflow;document.body.style.overflow='hidden';ref.current?.focus();
  const key=(e:KeyboardEvent)=>{if(e.key==='Escape')close.current();if(e.key==='Tab'){const items=ref.current?.querySelectorAll<HTMLElement>('button,a[href],input,select,summary');if(!items?.length)return;const first=items[0],last=items[items.length-1];if(e.shiftKey&&(document.activeElement===first||document.activeElement===ref.current)){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}};
  document.addEventListener('keydown',key);return()=>{document.removeEventListener('keydown',key);document.body.style.overflow=overflow;previous?.focus();};},[]);
 const handle=person?.handle??across?.account?.handle??accountId;
 const profile=tracking?.accounts.find(a=>a.id===accountId);const history=tracking?.history.filter(h=>h.account_id===accountId)??[];
 const savedProfile=[...(person?.posts??[])].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
 const useTracked=(profile?.profile_observed_at??'')>=(savedProfile?.followers_observed_at??'');
 const followers=(useTracked?profile?.followers:savedProfile?.followers)??across?.latest_profile?.followers??across?.account?.followers??null;
 const observed=(useTracked?profile?.profile_observed_at:savedProfile?.followers_observed_at)??across?.latest_profile?.observed_at??null;
 const evidence=Array.from(new Map([...(person?.posts??[]),...(person?.incoming??[])].map(p=>[p.id,p])).values()).sort((a,b)=>a.posted_at.localeCompare(b.posted_at));
 const summary=across?.summary??null;
 return <div className={styles.drawerBackdrop} onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><div ref={ref} tabIndex={-1} className={styles.drawer} role="dialog" aria-modal="true" aria-labelledby="account-title">
  <div className="flex justify-between items-start gap-4"><div><p className={styles.label}>Account evidence</p><h2 id="account-title" className="text-xl break-all">@{handle}</h2></div><button onClick={onClose} className={styles.control}>Close ✕</button></div>
  <p className={`${styles.muted} mt-2`}>{person?`First seen in this selection: ${person.first.slice(0,16).replace('T',' ')} UTC`:'No posts in the current coin and date selection; showing saved evidence across coins.'}</p>
  <p className="my-5 text-sm whitespace-pre-wrap break-words">{profile?.bio??across?.latest_profile?.bio??'No bio saved for this account yet.'}</p>
  <div className={styles.drawerStats}><div><p className={styles.label}>Posts in selection</p><strong>{person?.posts.length??0}</strong></div><div><p className={styles.label}>People interacting</p><strong>{person?.participants??0}</strong></div><div><p className={styles.label}>Latest followers</p><strong>{formatCount(followers)}</strong></div><div><p className={styles.label}>7-day growth</p><strong className={Number(profile?.growth_7d)>0?styles.positive:Number(profile?.growth_7d)<0?styles.negative:''}>{growthLabel(profile?.growth_7d)}</strong></div></div>
  <p className={`${styles.muted} my-3`}>Profile observed {observed?.slice(0,10)??'not yet'}. Follower figures describe recent snapshots, not historical audience size.</p>
  <a href={`https://x.com/i/user/${accountId}`} target="_blank" rel="noreferrer" className={styles.link}>Open X profile ↗</a>
  <h3 className="mt-6 mb-2">Across tracked coins</h3>
  {!across?<p role="status" className={styles.muted}>Loading cross-coin evidence…</p>:across.status!=='ready'?<p className={styles.muted}>Cross-coin evidence is unavailable without saved data.</p>:<>
   {summary&&<p className={styles.insight}>{whyLeader(summary)}. Roles come from saved text heuristics; price figures are the pinned pool's move after posts, not attribution.</p>}
   {across.activity?.length?<div className={styles.table}><table><thead><tr><th>Coin</th><th>Posts</th><th>First · last</th><th>Role</th><th>Episodes</th><th>+24h vs drift</th></tr></thead><tbody>{across.activity.map(a=>{const role=summary?.roles.find(r=>r.coin===a.coin),imp=summary?.impact.find(i=>i.coin===a.coin);return <tr key={a.coin}><td>{onOpenCoin?<button className={styles.link} onClick={()=>onOpenCoin(a.coin,'posts')}>{a.coin}</button>:a.coin}</td><td>{a.posts}<small className={styles.cellNote}>{a.originals} original</small></td><td>{String(a.first_at).slice(0,10)}<small className={styles.cellNote}>{String(a.last_at).slice(0,10)}</small></td><td>{role?role.role+(role.early?' · early':''):'—'}</td><td>{imp?.episodes??0}</td><td className={imp?.median_excess_24h==null?'':Number(imp.median_excess_24h)>0?styles.positive:styles.negative}>{pct(imp?.median_excess_24h)}<small className={styles.cellNote}>{imp?`up ${share(imp.share_up_24h)} of episodes`:'no complete window'}</small></td></tr>;})}</tbody></table></div>:<p className={styles.muted}>No saved non-repost posts on any tracked coin.</p>}
   {summary?.watcher&&<p className={styles.muted}>Watcher ranking: {summary.watcher}.</p>}
   {across.posts?.length?<details><summary>Latest saved posts across coins ({across.posts.length})</summary><div className={styles.eventList}>{across.posts.map(p=><article key={p.id}><time>{String(p.posted_at).slice(0,16).replace('T',' ')} UTC · {(p.coins??[]).join(', ')}{p.return_24h!=null?` · pool ${pct(p.return_24h)} after 24h`:''}</time><a href={p.url} target="_blank" rel="noreferrer" className={styles.link}>{p.kind} ↗</a><p className="text-sm whitespace-pre-wrap break-words">{p.text}</p></article>)}</div></details>:null}
  </>}
  <h3 className="mt-6 mb-3">Posts & amplification in this selection</h3>{!evidence.length&&<p className={styles.muted}>None in the current selection.</p>}<div className={styles.eventList}>{evidence.slice(0,30).map(p=><article key={p.id}><time>{p.posted_at.slice(0,16).replace('T',' ')} UTC</time><a href={p.url} target="_blank" rel="noreferrer" className={styles.link}>@{p.handle} · {p.kind} ↗</a><p className="text-sm whitespace-pre-wrap break-words">{p.text}</p></article>)}</div>
  {evidence.length>30&&<p className={styles.muted}>Showing the earliest 30 of {evidence.length} source posts in this selection.</p>}
  <details><summary>Follower observations ({history.length})</summary><table><thead><tr><th>Date (UTC)</th><th>Followers</th></tr></thead><tbody>{history.map(p=><tr key={p.observed_at}><td>{p.observed_at.slice(0,10)}</td><td>{p.available?formatCount(p.followers):'Unavailable'}</td></tr>)}</tbody></table></details>
 </div></div>;
}
