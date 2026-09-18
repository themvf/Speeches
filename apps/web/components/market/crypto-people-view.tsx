"use client";
import {useEffect,useMemo,useState,type ReactNode} from 'react';
import {createColumnHelper,flexRender,getCoreRowModel,getSortedRowModel,useReactTable,type SortingState} from '@tanstack/react-table';
import {afterPosting,earlyCoins,postingStyles,whyLeader,MIN_LINKED_POSTS,STYLE_LABEL,type Leader,type PostingStyle} from '@/lib/crypto-leaders';
import {COIN_SYMBOLS} from '@/lib/crypto-coins';
import styles from './crypto-research.module.css';
type Row=Leader&{last_at:string|null;posts:number;days:number};
const compact=(n:number|null)=>n==null?'audience unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n)+' followers';
const ago=(iso:string|null)=>{if(!iso)return 'no saved posts';const d=Math.floor((Date.now()-Date.parse(iso))/86400000);return d<=0?'active today':d===1?'active yesterday':`active ${d}d ago`;};
// Inline stroke icons, same convention as the rail. Each stands for one posting style; the text label sits beside it.
const ICON:Record<PostingStyle,ReactNode>={
 first:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 21V4M5 4h12l-3 4 3 4H5"/></svg>,
 early:<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>,
 analysis:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h7a2 2 0 0 1 2 2v13a2 2 0 0 0-2-2H4zM20 5h-7a2 2 0 0 0-2 2v13a2 2 0 0 1 2-2h7z"/></svg>,
 amplifier:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 10v4h3l6 4V6l-6 4zM16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/></svg>,
 feed:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 19a1 1 0 1 0 0-.1M5 12a7 7 0 0 1 7 7M5 5a14 14 0 0 1 14 14"/></svg>,
 project:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 21V5l8-3 8 3v16M9 9h2M13 9h2M9 13h2M13 13h2M9 17h2M13 17h2"/></svg>,
 watcher:<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>,
};
const STYLE_FILTER:Record<string,(r:Row)=>boolean>={any:()=>true,first:r=>postingStyles(r).includes('first'),early:r=>r.early_coins>0,analysis:r=>postingStyles(r).includes('analysis'),amplifier:r=>postingStyles(r).includes('amplifier'),watcher:r=>!!r.watcher,feed:r=>postingStyles(r).includes('feed'),project:r=>postingStyles(r).includes('project')};
const sampled=(r:Row)=>r.episodes>=MIN_LINKED_POSTS;
// "Rank by" presets map onto column sorting; clicking any header overrides them.
const PRESET:Record<string,SortingState>={early:[{id:'early',desc:true}],hit:[{id:'after',desc:true}],move:[{id:'move',desc:true}],evidence:[{id:'evidence',desc:true}],audience:[{id:'audience',desc:true}],recent:[{id:'recent',desc:true}]};
const col=createColumnHelper<Row>();
const num=(v:number|null|undefined)=>v==null?-Infinity:v;
const COLUMNS=[
 col.accessor(r=>r.handle.toLowerCase(),{id:'who',header:'Who'}),
 col.accessor(r=>r.early_coins,{id:'early',header:'Where early',sortingFn:(a,b)=>a.original.early_coins-b.original.early_coins||Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.hit_rate)-num(b.original.hit_rate)||num(a.original.median_excess_24h)-num(b.original.median_excess_24h)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>r.hit_rate,{id:'after',header:'After they post',sortingFn:(a,b)=>Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.hit_rate)-num(b.original.hit_rate)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>postingStyles(r).length,{id:'how',header:'How they post'}),
 col.accessor(r=>r.episodes,{id:'evidence',header:'Evidence',sortingFn:(a,b)=>a.original.episodes-b.original.episodes||a.original.coins.length-b.original.coins.length}),
 // Sort-only columns behind the "Rank by" presets; never rendered.
 col.accessor(r=>r.median_excess_24h,{id:'move',sortingFn:(a,b)=>Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.median_excess_24h)-num(b.original.median_excess_24h)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>r.followers,{id:'audience',sortingFn:(a,b)=>num(a.original.followers)-num(b.original.followers)}),
 col.accessor(r=>r.last_at??'',{id:'recent'}),
];
const HIDDEN={move:false,audience:false,recent:false};
export function CryptoPeopleView({initialCoin,onAccount}:{initialCoin:string|null;onAccount:(id:string)=>void}){
 const [rows,setRows]=useState<Row[]|null>(null),[error,setError]=useState('');
 const [coin,setCoin]=useState(initialCoin??'ALL'),[rank,setRank]=useState('early'),[sorting,setSorting]=useState<SortingState>(PRESET.early),[style,setStyle]=useState('any'),[minEp,setMinEp]=useState(0),[audience,setAudience]=useState(0),[active,setActive]=useState(0),[search,setSearch]=useState('');
 useEffect(()=>{const c=new AbortController();fetch('/api/market/crypto/leaders?all=1',{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setRows(b.data.leaders);}).catch(()=>{if(!c.signal.aborted)setError('People could not be loaded.');});return()=>c.abort();},[]);
 const shown=useMemo(()=>{const since=active?Date.now()-active*86400000:0;
  return (rows??[]).filter(r=>(coin==='ALL'||r.coins.includes(coin))&&r.episodes>=minEp&&(r.followers??0)>=audience&&(!since||(r.last_at&&Date.parse(r.last_at)>=since))&&r.handle.toLowerCase().includes(search.toLowerCase())&&(STYLE_FILTER[style]??STYLE_FILTER.any)(r))
   ;},[rows,coin,style,minEp,audience,active,search]);
 const table=useReactTable({data:shown,columns:COLUMNS,state:{sorting,columnVisibility:HIDDEN},onSortingChange:setSorting,getCoreRowModel:getCoreRowModel(),getSortedRowModel:getSortedRowModel(),sortDescFirst:true});
 const pickPreset=(k:string)=>{setRank(k);setSorting(PRESET[k]??PRESET.early);};
 return <div style={{display:'flex',flexDirection:'column',gap:18}}>
  <div className={styles.wsChips} role="group" aria-label="Coin">{['ALL',...COIN_SYMBOLS].map(s=><button key={s} aria-pressed={coin===s} onClick={()=>setCoin(s)}>{s==='ALL'?'All coins':s}</button>)}</div>
  <div className={styles.wsFilters}>
   <label>Rank by<select value={rank} onChange={e=>pickPreset(e.target.value)}><option value="early">Early on most coins</option><option value="hit">Most often up after posting</option><option value="move">Biggest move after posting</option><option value="evidence">Most price-linked posts</option><option value="audience">Largest audience</option><option value="recent">Most recently active</option></select></label>
   <label>How they post<select value={style} onChange={e=>setStyle(e.target.value)}><option value="any">Any way</option><option value="first">First to share a contract</option><option value="early">Among the first to post</option><option value="analysis">Explains rather than hypes</option><option value="amplifier">Mostly quotes others</option><option value="watcher">Reports whale trades</option><option value="feed">Alert feed</option><option value="project">Project account</option></select></label>
   <label>Price-linked posts<select value={minEp} onChange={e=>setMinEp(Number(e.target.value))}><option value={0}>Any</option><option value={3}>3+</option><option value={5}>5+</option><option value={10}>10+</option></select></label>
   <label>Audience<select value={audience} onChange={e=>setAudience(Number(e.target.value))}><option value={0}>Any size</option><option value={10000}>10K+</option><option value={100000}>100K+</option></select></label>
   <label>Active<select value={active} onChange={e=>setActive(Number(e.target.value))}><option value={0}>Any time</option><option value={7}>Last 7 days</option><option value={30}>Last 30 days</option></select></label>
   <label>Handle<input type="search" placeholder="@handle" value={search} onChange={e=>setSearch(e.target.value)}/></label>
   <span className={styles.muted} style={{fontSize:12,paddingBottom:10}}>{rows?`${shown.length} of ${rows.length} accounts with evidence`:''}</span>
  </div>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!rows?<p role="status" className={styles.muted}>Reading saved rankings…</p>:!shown.length?<p className={styles.empty}>No accounts meet these filters. Rankings appear after the next collection run; price-linked posts need a day of hourly candles.</p>:
  <section className={styles.wsSection}><div className={`${styles.table} ${styles.peopleScroll}`}><table className={styles.peopleTable}><thead>{table.getHeaderGroups().map(g=><tr key={g.id}>{g.headers.map(h=>{const dir=h.column.getIsSorted();return <th key={h.id} aria-sort={dir==='asc'?'ascending':dir==='desc'?'descending':'none'}><button type="button" className={styles.sortHeader} onClick={h.column.getToggleSortingHandler()}>{flexRender(h.column.columnDef.header,h.getContext())}<span aria-hidden="true">{dir==='asc'?'▲':dir==='desc'?'▼':''}</span></button></th>;})}</tr>)}</thead><tbody>{table.getRowModel().rows.slice(0,100).map((row,i)=>{const r=row.original,early=earlyCoins(r),after=afterPosting(r),ways=postingStyles(r);return <tr key={r.account_id} className={sampled(r)?'':styles.dimRow}>
   <td><div className={styles.accountIdentity}><span className={styles.rankNumber}>{i+1}</span><div><button className={styles.link} onClick={()=>onAccount(r.account_id)}>@{r.handle}</button><small>{compact(r.followers)} · {ago(r.last_at)}</small></div></div></td>
   <td>{early.length?<span className={styles.coinDays}>{early.map(e=><span key={e.coin} className={styles.wsPill}>{e.coin}{e.day?<small> day {e.day}</small>:null}</span>)}</span>:<span className={styles.muted}>—</span>}</td>
   <td>{after?<span className={r.hit_rate!=null&&r.hit_rate>=.5?styles.positive:styles.negative}>{after}</span>:<span className={styles.muted}>{r.episodes?`${r.episodes} price-linked post${r.episodes>1?'s':''} · not enough to say`:'no price-linked posts'}</span>}</td>
   <td>{ways.length?<span className={styles.howChips}>{ways.map(w=><span key={w} className={styles.howChip} title={STYLE_LABEL[w]}>{ICON[w]}<span>{STYLE_LABEL[w]}</span></span>)}</span>:<span className={styles.muted}>saved posts only</span>}</td>
   <td style={{minWidth:230}}><span style={{whiteSpace:'nowrap'}}>{r.episodes} price-linked · {r.coins.length} coin{r.coins.length===1?'':'s'} · {r.days} day{r.days===1?'':'s'}</span><small className={styles.cellNote}>{whyLeader(r)}</small></td>
  </tr>;})}</tbody></table></div>
  <small className={styles.muted}>Click a column heading to sort by it; the Rank by menu is a preset.</small>
  <small className={styles.muted}>Where early = the calendar day of the account&#39;s first post on that coin, counted from the earliest saved post of its contract (day 1 = same day; only within the first week). After they post = the pinned pool 24 hours after the account&#39;s first post on a coin in a day, compared with the coin&#39;s own median move; shown only with {MIN_LINKED_POSTS}+ such posts, and rows with fewer are dimmed. How they post is a text heuristic. Nothing here verifies influence or causation. {shown.length>100?`Showing 100 of ${shown.length}.`:''}</small></section>}
 </div>;
}
