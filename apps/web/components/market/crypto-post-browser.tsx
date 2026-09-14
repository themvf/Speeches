"use client";
import {useEffect,useMemo,useState} from 'react';
import {filterEvidence,type RunData,type RunPost} from '@/lib/crypto-run';
import {defaultPostFilters,filterSavedPosts,type PostFilters} from '@/lib/crypto-post-filter';
import styles from './crypto-research.module.css';
type TaggedPost=RunPost&{coins:string[]};
const coins=['ZCAT','PONS','ZEC'];
const count=(n:number|null|undefined)=>n==null?'—':n.toLocaleString();
export function CryptoPostBrowser({coin,run}:{coin:string;run:RunData}){
 const [scope,setScope]=useState('current'),[evidence,setEvidence]=useState('words'),[filters,setFilters]=useState<PostFilters>({...defaultPostFilters}),[page,setPage]=useState(0);
 const [extra,setExtra]=useState<Record<string,RunData>>({}),[loading,setLoading]=useState(false),[error,setError]=useState(''),[retry,setRetry]=useState(0);
 useEffect(()=>{if(scope!=='all')return;const controller=new AbortController();setLoading(true);setError('');
  Promise.all(coins.filter(c=>c!==coin).map(async c=>{const response=await fetch(`/api/market/crypto/run?coin=${c}`,{signal:controller.signal});const body=await response.json();if(!response.ok||!body.ok)throw Error();return [c,body.data] as [string,RunData];})).then(rows=>{if(!controller.signal.aborted){setExtra(Object.fromEntries(rows));setLoading(false);}}).catch(()=>{if(!controller.signal.aborted){setLoading(false);setError('Could not load all searches. Retry to avoid reviewing an incomplete combined result.');}});return()=>controller.abort();
 },[scope,coin,retry]);
 const datasets=useMemo(()=>scope==='all'?{...extra,[coin]:run}:{[coin]:run},[scope,extra,coin,run]);
 const posts=useMemo(()=>{const map=new Map<string,TaggedPost>();for(const [symbol,data] of Object.entries(datasets)){const eligible=evidence==='contract'&&symbol==='ZEC'?[]:filterEvidence(data.posts,symbol,evidence);for(const p of eligible){const existing=map.get(p.id);if(existing)existing.coins.push(symbol);else map.set(p.id,{...p,coins:[symbol]});}}return [...map.values()];},[datasets,evidence]);
 const results=useMemo(()=>filterSavedPosts(posts,filters),[posts,filters]);
 const pages=Math.max(1,Math.ceil(results.length/25)),currentPage=Math.min(page,pages-1),shown=results.slice(currentPage*25,(currentPage+1)*25);
 const update=(patch:Partial<PostFilters>)=>{setFilters(f=>({...f,...patch}));setPage(0);};
 const reset=()=>{setFilters({...defaultPostFilters});setEvidence('words');setPage(0);};
 const active=Object.entries(filters).filter(([k,v])=>v!==defaultPostFilters[k as keyof PostFilters]).length+(evidence!=='words'?1:0);
 return <div className={styles.postBrowser}>
  <div className={styles.sectionHeading}><div><p className={styles.label}>Saved search results</p><h3>Find the posts that matter</h3><p className={styles.muted}>Search one coin or compare all three. Filters below apply to this post view.</p></div><span className={styles.savedBadge}>No X credits used</span></div>
  <div className={styles.postSearchRow}><input type="search" aria-label="Search post text" placeholder={'Search words or "exact phrases"'} value={filters.query} onChange={e=>update({query:e.target.value})}/><select aria-label="Post search scope" value={scope} onChange={e=>{setScope(e.target.value);setPage(0);}}><option value="current">{coin} search</option><option value="all">All searches · ZCAT, PONS, ZEC</option></select></div>
  <div className={styles.toolbar}><div className={styles.audienceChips} role="group" aria-label="Post shortcuts"><button onClick={()=>update({followers:100000})}>100K+ audience</button><button onClick={()=>update({likes:100,sort:'likes'})}>100+ likes</button><button onClick={()=>update({kind:'original'})}>Original posts</button><button onClick={()=>update({sort:'oldest'})}>Earliest first</button></div><button className={styles.control} onClick={reset}>Reset filters{active?` (${active})`:''}</button></div>
  <details className={styles.scopeControls}><summary><span>Refine results</span><span className={styles.scopeSummary}>{active?`${active} settings changed`:'Account, dates, exclusions & engagement'}</span></summary><div className={styles.postFilterGrid}>
   <label>Account contains<input aria-label="Post author" placeholder="@handle" value={filters.author} onChange={e=>update({author:e.target.value})}/></label>
   <label>Exclude words / phrases<input aria-label="Exclude post text" placeholder={'airdrop "join my telegram"'} value={filters.exclude} onChange={e=>update({exclude:e.target.value})}/></label>
   <label>From (UTC)<input type="date" value={filters.from} max={filters.to||undefined} onChange={e=>update({from:e.target.value})}/></label>
   <label>Through (UTC)<input type="date" value={filters.to} min={filters.from||undefined} onChange={e=>update({to:e.target.value})}/></label>
   <label>Minimum followers<input type="number" min="0" value={filters.followers} onChange={e=>update({followers:Math.max(0,Number(e.target.value))})}/></label>
   <label>Minimum likes<input type="number" min="0" value={filters.likes} onChange={e=>update({likes:Math.max(0,Number(e.target.value))})}/></label>
   <label>Minimum reposts<input type="number" min="0" value={filters.reposts} onChange={e=>update({reposts:Math.max(0,Number(e.target.value))})}/></label>
   <label>Post type<select value={filters.kind} onChange={e=>update({kind:e.target.value})}>{['all','original','reply','quote','repost'].map(k=><option key={k} value={k}>{k==='all'?'All types':k}</option>)}</select></label>
   <label>Coin evidence<select value={evidence} onChange={e=>{setEvidence(e.target.value);setPage(0);}}><option value="words">Coin mentions</option><option value="contract">Exact contract · ZCAT / PONS</option><option value="all">All saved search results</option></select></label>
  </div><p className={styles.muted}>All included terms must match; any excluded term removes a post. Use quotes for phrases. Exact-contract mode excludes Zcash, which has no token contract.</p></details>
  <div className={styles.toolbar}><label className={styles.checkLabel}><input type="checkbox" checked={filters.onePerAuthor} onChange={e=>update({onePerAuthor:e.target.checked})}/> One post per account, using the selected sort</label><label>Sort <select aria-label="Post sort" value={filters.sort} onChange={e=>update({sort:e.target.value})}><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="followers">Largest audience</option><option value="likes">Most likes</option><option value="reposts">Most reposts</option></select></label></div>
  <p className={styles.muted}>Audience ≥ {count(filters.followers)} · Likes ≥ {count(filters.likes)} · Reposts ≥ {count(filters.reposts)} · {filters.kind==='all'?'All post types':filters.kind}{filters.exclude?` · Excluding: ${filters.exclude}`:''}{filters.from||filters.to?` · ${filters.from||'Start'} → ${filters.to||'Latest'} UTC`:''}</p>
  {Object.entries(datasets).filter(([,d])=>d.total>d.posts.length).map(([c,d])=><p key={c} className={styles.coverageNote}>{c}: searching the earliest {d.posts.length.toLocaleString()} of {d.total.toLocaleString()} saved posts returned by the timeline API.</p>)}
  {loading?<p role="status">Loading saved searches…</p>:error?<p role="alert">{error} <button className={styles.control} onClick={()=>setRetry(n=>n+1)}>Retry</button></p>:<>
   <p role="status" className={styles.resultsCaption}>{results.length.toLocaleString()} matches · {new Set(results.map(p=>p.author_id)).size.toLocaleString()} accounts · {posts.length.toLocaleString()} unique posts searched</p>
   {!shown.length?<div className={styles.empty}>No posts match. Lower the thresholds or remove a filter. <button className={styles.control} onClick={reset}>Reset filters</button></div>:<div className={styles.postResults}>{shown.map(p=><article key={p.id}><header><a href={`https://x.com/i/user/${p.author_id}`} target="_blank" rel="noreferrer" className={styles.link}>@{p.handle}</a><span>{p.coins.join(' · ')} · {p.kind}</span><time>{p.posted_at.slice(0,16).replace('T',' ')} UTC</time></header><p className={styles.postFullText}>{p.text}</p><footer><span>{count(p.followers)} followers</span><span>{count(p.likes)} likes</span><span>{count(p.reposts)} reposts</span><span>{count(p.quotes)} quotes</span><a className={styles.link} href={p.url} target="_blank" rel="noreferrer">Open post ↗</a></footer><small className={styles.muted}>Audience observed {p.followers_observed_at?.slice(0,10)??'unknown'} · Engagement observed {p.metrics_observed_at?.slice(0,10)??'unknown'}</small></article>)}</div>}
   <div className={styles.toolbar}><button className={styles.control} disabled={!currentPage} onClick={()=>setPage(currentPage-1)}>Previous</button><span className={styles.muted}>Page {currentPage+1} of {pages} · 25 posts per page</span><button className={styles.control} disabled={currentPage>=pages-1} onClick={()=>setPage(currentPage+1)}>Next</button></div>
  </>}
  <p className={styles.muted}>Duplicate post IDs across searches appear once. Missing metrics show — and do not meet positive thresholds. Counts are saved observations, not historical reach or proof of market impact. Search coverage is incomplete.</p>
 </div>;
}
