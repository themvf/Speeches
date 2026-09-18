"use client";
import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {useRouter,useSearchParams} from 'next/navigation';
import {createColumnHelper,flexRender,getCoreRowModel,getSortedRowModel,useReactTable,type SortingState,type ColumnDef} from '@tanstack/react-table';
import {COINS,coinConfig} from '@/lib/crypto-coins';
import {afterPosting,earlyCoins,postingStyles,whyLeader,MIN_LINKED_POSTS,STYLE_LABEL,type Leader} from '@/lib/crypto-leaders';
import {pct} from '@/lib/crypto-impact';
import type {Signal,boardRows} from '@/lib/crypto-signals';
import type {CoinFinding} from '@/lib/crypto-coin-discovery';
import {daysBetween,largestDailyGain,networkEdges,scopePosts,type RunData,type RunMarket} from '@/lib/crypto-run';
import {voiceEvidence} from '@/lib/crypto-voices';
import {computeRings,RING_LABEL,RING_DESCRIPTION,type Ring} from '@/lib/crypto-rings';
import {describeScope,matchesQuery,parseQuery,readState,writeState,legacyState,TABS,type State,type Tab} from '@/lib/crypto-workbench';
import {CryptoRunChart} from './crypto-run-chart';
import {CryptoBeforeMove} from './crypto-before-move';
import {CryptoDataView} from './crypto-data-view';
import s from './crypto-workbench.module.css';
type Row=Leader&{last_at:string|null;posts:number;days:number};
type Board=ReturnType<typeof boardRows>[number];
type Signals={status:string;board:Board[];signals:Signal[];asOf?:string};
type Status={coverage:{symbol:string;hourly_hours:number;hourly_latest:string|null;source_id:string|null;origin:{status:string}|null;days_total:number;days_searched:number}[];ledgers:{name:string;used:number;ceiling:number}[];lastCollection:string|null};
type Account={status:string;account?:{id:string;handle:string;followers:number|null};latest_profile?:{bio:string|null;followers:number|null}|null;activity?:{coin:string;posts:number;originals:number;first_at:string;last_at:string}[];posts?:{id:string;text:string;url:string;posted_at:string;kind:string;coins:string[]|null;return_24h:number|null}[];summary?:Leader|null};
const aud=(n:number|null|undefined)=>n==null?'—':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?Math.round(n/1e3)+'K':String(n);
const ago=(iso:string|null)=>{if(!iso)return '—';const d=Math.floor((Date.now()-Date.parse(iso))/86400000);return d<=0?'today':d===1?'1d':d+'d';};
const hm=(iso:string)=>iso.slice(11,16);
const sampled=(r:Leader)=>r.episodes>=MIN_LINKED_POSTS;
const tone=(v:number|null|undefined)=>v==null?s.faint:v>0?s.pos:v<0?s.neg:'';
async function load<T>(path:string,signal:AbortSignal):Promise<T>{const r=await fetch(path,{signal});const b=await r.json();if(!r.ok||!b.ok)throw Error(b.error??'load failed');return b.data as T;}
const num=(v:number|null|undefined)=>v==null?-Infinity:v;
const col=createColumnHelper<Row>();
const COLUMNS=[
 col.accessor(r=>r.handle.toLowerCase(),{id:'who',header:'Account'}),
 col.accessor(r=>r.followers,{id:'aud',header:'Aud',sortingFn:(a,b)=>num(a.original.followers)-num(b.original.followers)}),
 col.accessor(r=>r.early_coins,{id:'early',header:'Early',sortingFn:(a,b)=>a.original.early_coins-b.original.early_coins||Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.hit_rate)-num(b.original.hit_rate)||num(a.original.median_excess_24h)-num(b.original.median_excess_24h)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>r.hit_rate,{id:'hit',header:'Up',sortingFn:(a,b)=>Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.hit_rate)-num(b.original.hit_rate)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>r.median_excess_24h,{id:'move',header:'Excess 24h',sortingFn:(a,b)=>Number(sampled(a.original))-Number(sampled(b.original))||num(a.original.median_excess_24h)-num(b.original.median_excess_24h)||a.original.episodes-b.original.episodes}),
 col.accessor(r=>r.episodes,{id:'linked',header:'Linked'}),
 col.accessor(r=>r.coins.length,{id:'coins',header:'Coins'}),
 col.accessor(r=>r.days,{id:'days',header:'Days'}),
 col.accessor(r=>postingStyles(r).length,{id:'style',header:'Style'}),
 col.accessor(r=>r.last_at??'',{id:'active',header:'Active'}),
];
const RIGHT=new Set(['aud','hit','move','linked','coins','days']);
const RINGS:Ring[]=[1,2,3,4,5,6,7,8];
export function CryptoWorkbench(){
 const router=useRouter(),params=useSearchParams();
 const state=useMemo(()=>readState(params),[params]);
 // Latest state lives in a ref so two updates in quick succession compose instead of the second reading a stale URL.
 const stateRef=useRef(state);useEffect(()=>{stateRef.current=state;},[state]);
 const set=useCallback((patch:Partial<State>)=>{const next={...stateRef.current,...patch};stateRef.current=next;router.replace(writeState(next),{scroll:false});},[router]);
 // Legacy ?view= links land here; the old sub-paths are redirected by next.config.
 useEffect(()=>{if(params.get('view')){const l=legacyState(window.location.pathname,params);if(l)router.replace(writeState(l));}},[params,router]);
 const [qText,setQText]=useState(state.q);useEffect(()=>{setQText(state.q);},[state.q]);
 useEffect(()=>{if(qText===state.q)return;const t=setTimeout(()=>set({q:qText}),300);return()=>clearTimeout(t);},[qText,state.q,set]);
 const query=useMemo(()=>parseQuery(qText),[qText]);
 const coin=query.coin??state.coin;
 const focusCoin=coin??COINS[0].symbol; // coin-shaped panels need one coin; the scope strip says when it is implicit
 const [sig,setSig]=useState<Signals|null>(null),[leaders,setLeaders]=useState<Row[]|null>(null),[disc,setDisc]=useState<CoinFinding[]|null>(null),[status,setStatus]=useState<Status|null>(null),[err,setErr]=useState('');
 useEffect(()=>{const c=new AbortController();const from=new Date(Date.now()-7*86400000).toISOString().slice(0,10);
  load<Signals>('/api/market/crypto/signals',c.signal).then(setSig).catch(()=>{if(!c.signal.aborted)setErr('Signals could not be loaded.');});
  load<{leaders:Row[]}>('/api/market/crypto/leaders?all=1',c.signal).then(d=>setLeaders(d.leaders)).catch(()=>{if(!c.signal.aborted)setLeaders([]);});
  load<{coins:CoinFinding[]}>(`/api/market/crypto/coin-discovery?from=${from}`,c.signal).then(d=>setDisc(d.coins)).catch(()=>setDisc([]));
  load<Status>('/api/market/crypto/status',c.signal).then(setStatus).catch(()=>{});return()=>c.abort();},[]);
 const [run,setRun]=useState<RunData|null>(null),[market,setMarket]=useState<RunMarket|null>(null);
 useEffect(()=>{const c=new AbortController();setRun(null);setMarket(null);
  load<RunData>(`/api/market/crypto/run?coin=${focusCoin}`,c.signal).then(setRun).catch(()=>{if(!c.signal.aborted)setRun({status:'error',posts:[],days:[],total:0,limit:0,start:coinConfig(focusCoin).archiveStart,end:new Date().toISOString().slice(0,10)});});
  load<RunMarket>(`/api/market/crypto/history?coin=${focusCoin}`,c.signal).then(setMarket).catch(()=>{});return()=>c.abort();},[focusCoin]);
 const [acct,setAcct]=useState<Account|null>(null);
 useEffect(()=>{if(!state.account){setAcct(null);return;}const c=new AbortController();setAcct(null);load<Account>(`/api/market/crypto/account?id=${encodeURIComponent(state.account)}`,c.signal).then(setAcct).catch(()=>{if(!c.signal.aborted)setAcct({status:'error'});});return()=>c.abort();},[state.account]);
 // Coin-shaped panels
 const today=new Date().toISOString().slice(0,10);const start=coinConfig(focusCoin).archiveStart;const fromDay=state.from??start,toDay=state.to??today;
 const days=useMemo(()=>daysBetween(run?.start??start,run?.end??today),[run?.start,run?.end,start,today]);
 const eligible=useMemo(()=>voiceEvidence(run?.posts??[],focusCoin),[run,focusCoin]);
 const scoped=useMemo(()=>scopePosts(eligible.eligible,fromDay,toDay),[eligible,fromDay,toDay]);
 // Rings: who posts with whom on the focus coin over the current window.
 const rings=useMemo(()=>computeRings(scoped,focusCoin,[coinConfig(focusCoin).name]),[scoped,focusCoin]);
 const ringCounts=useMemo(()=>{const c=new Map<Ring,number>();for(const r of rings.values())if(r.ring)c.set(r.ring,(c.get(r.ring)??0)+1);return c;},[rings]);
 // People
 const people=useMemo(()=>(leaders??[]).filter(l=>matchesQuery(l,query,coin,rings)),[leaders,query,coin,rings]);
 const columns=useMemo<ColumnDef<Row,unknown>[]>(()=>[...COLUMNS.slice(0,3),col.accessor(r=>rings.get(r.account_id)?.ring??9,{id:'ring',header:'Ring',sortingFn:(a,b)=>(rings.get(b.original.account_id)?.ring??9)-(rings.get(a.original.account_id)?.ring??9)}) as ColumnDef<Row,unknown>,...COLUMNS.slice(3)] as ColumnDef<Row,unknown>[],[rings]);
 const [sorting,setSorting]=useState<SortingState>([{id:'early',desc:true}]);
 const table=useReactTable({data:people,columns,state:{sorting},onSortingChange:setSorting,getCoreRowModel:getCoreRowModel(),getSortedRowModel:getSortedRowModel(),sortDescFirst:true});
 const rows=table.getRowModel().rows;
 const [cursor,setCursor]=useState(0);useEffect(()=>{setCursor(0);},[people,sorting]);
 const [help,setHelp]=useState(false);
 const series=market?.points??[];const gain=largestDailyGain(series.filter(p=>p.day>=fromDay&&p.day<=toDay));
 const closes=useMemo(()=>new Map(series.map(p=>[p.day,p.close])),[series]);
 const nextClose=(iso:string)=>{const d=iso.slice(0,10);const a=closes.get(d),b=closes.get(new Date(Date.parse(d+'T00:00:00Z')+86400000).toISOString().slice(0,10));return a&&b?b/a-1:null;};
 const feed=useMemo(()=>scoped.filter(p=>p.kind!=='repost'&&(!query.handle||p.handle.toLowerCase().includes(query.handle))).sort((a,b)=>b.posted_at.localeCompare(a.posted_at)).slice(0,400),[scoped,query.handle]);
 const beforeCount=useMemo(()=>{if(!state.day)return null;const e=Date.parse(state.day+'T00:00:00Z'),st=e-86400000;return new Set(eligible.eligible.filter(p=>{const t=Date.parse(p.posted_at);return t>=st&&t<e&&p.kind!=='repost';}).map(p=>p.author_id)).size;},[eligible,state.day]);
 const edges=useMemo(()=>{const e=networkEdges(scoped);const names=new Map<string,string>();for(const p of scoped){names.set(p.author_id,p.handle);for(const x of p.edges)names.set(x.target_id,x.target);}return e.sort((a,b)=>b.weight-a.weight).slice(0,200).map(x=>({...x,from:names.get(x.source)??x.source,to:names.get(x.target)??x.target}));},[scoped]);
 const searched=(run?.days??[]).filter(d=>d.searched>0).length;
 // Keyboard: / focuses the command line, arrows move the cursor, Enter opens, c pins the row's first coin, Esc clears.
 const cmd=useRef<HTMLInputElement>(null);
 useEffect(()=>{const h=(e:KeyboardEvent)=>{const typing=(e.target as HTMLElement)?.tagName==='INPUT'||(e.target as HTMLElement)?.tagName==='TEXTAREA'||(e.target as HTMLElement)?.tagName==='SELECT';
  if(e.key==='Escape'){if(typing)(e.target as HTMLElement).blur();else if(state.account)set({account:null});else{setQText('');set({q:'',coin:null});}return;}
  if(typing)return;
  if(e.key==='/'){e.preventDefault();cmd.current?.focus();return;}
  if(state.tab!=='people'||!rows.length)return;
  if(e.key==='ArrowDown'){e.preventDefault();setCursor(c=>Math.min(rows.length-1,c+1));}
  else if(e.key==='ArrowUp'){e.preventDefault();setCursor(c=>Math.max(0,c-1));}
  else if(e.key==='Enter'){set({account:rows[cursor]?.original.account_id??null});}
  else if(e.key==='c'){const r=rows[cursor]?.original;if(r)set({coin:state.coin===r.coins[0]?null:r.coins[0]});}};
  document.addEventListener('keydown',h);return()=>document.removeEventListener('keydown',h);},[rows,cursor,state.tab,state.account,state.coin,set]);
 const pinCoin=(c:string)=>{setQText(q=>q.replace(/\b[A-Za-z]{2,10}\b/g,t=>t.toUpperCase()===c?'':t).replace(/\s+/g,' ').trim());set({coin:state.coin===c?null:c});};
 const cov=status?.coverage??[];const noPool=cov.filter(c=>!c.source_id).map(c=>c.symbol);const latest=cov.map(c=>c.hourly_latest).filter(Boolean).sort().at(-1);
 const credits=status?.ledgers.reduce((n,l)=>n+Math.max(0,l.ceiling-l.used),0);
 const summary=acct?.summary??null;
 return <div className={s.wb}>
  <div className={s.top}><h1>Crypto workbench</h1><label className={s.cmdLabel} htmlFor="wb-cmd">Command</label>
   <input id="wb-cmd" ref={cmd} className={s.cmd} value={qText} onChange={e=>setQText(e.target.value)} placeholder="ZCAT · @handle · contract · early day<3 · hit>0.7 posts>5 · watcher   (press / to focus)" spellCheck={false} aria-label="Command line"/>
   <button className={s.btn} onClick={()=>{setQText('');set({q:'',coin:null,account:null,day:null,highlight:null,from:null,to:null});}}>Clear</button>
   <span className={s.status}>{sig?.asOf?`Updated ${hm(sig.asOf)} UTC`:''}{leaders?` · ${leaders.length} accounts · ${leaders.filter(sampled).length} with a sample`:''}{credits!=null?` · ${credits.toLocaleString()} credits left`:''}</span></div>
  <div className={s.scope}><span>Scope</span><span className={`${s.mono}`} style={{color:'#7dd3fc'}}>{describeScope(query,coin)}{!coin&&state.tab!=='people'&&state.tab!=='data'?` · showing ${focusCoin}`:''}</span>{query.unknown.length>0&&<span className={s.warn}>ignored: {query.unknown.join(' ')}</span>}<span style={{marginLeft:'auto'}}>{leaders?`${people.length} of ${leaders.length} accounts`:''}</span></div>
  <div className={s.body}>
   {/* Left: coins, signals, discovery */}
   <div className={`${s.col} ${s.left}`}>
    <div className={s.h}>Coins · 24h</div>
    <div className={s.coinsBox}><table className={s.t}><thead><tr><th>Coin</th><th className={s.r}>Px</th><th className={s.r}>Vol×</th><th className={s.r}>Posts</th><th className={s.r}>Sig</th></tr></thead><tbody>
     {(sig?.board??COINS.map(c=>({symbol:c.symbol,price_change:null,volume_ratio:null,posts_24h:0}))).map(b=>{const n=(sig?.signals??[]).filter(x=>x.coin===b.symbol).length;return <tr key={b.symbol} className={`${s.row} ${coin===b.symbol?s.on:''}`} onClick={()=>pinCoin(b.symbol)}><td style={{fontWeight:600}}>{b.symbol}</td><td className={`${s.r} ${tone(b.price_change)}`}>{b.price_change==null?'—':pct(b.price_change,0)}</td><td className={`${s.r} ${s.soft}`}>{b.volume_ratio==null?'—':b.volume_ratio.toFixed(1)}</td><td className={s.r}>{b.posts_24h}</td><td className={s.r}>{n||'—'}</td></tr>;})}
    </tbody></table></div>
    <div className={s.h}>Signals · last 24h {err&&<span className={s.warn}>{err}</span>}</div>
    <div className={s.scroll}>{!sig?<div className={s.empty}>Reading signals…</div>:!sig.signals.filter(x=>!coin||x.coin===coin).length?<div className={s.empty}>No signals in scope.</div>:sig.signals.filter(x=>!coin||x.coin===coin).map((x,i)=><div key={i} className={s.feedRow} style={{gridTemplateColumns:'40px 56px minmax(0,1fr)'}} onClick={()=>set({coin:x.coin,account:x.account_id??state.account})} title={x.detail}><span className={`${s.mono} ${s.faint}`}>{hm(x.at)}</span><span style={{fontWeight:600,color:'#7dd3fc'}}>{x.coin}</span><span className={s.clip}>{x.title}{x.handle?' · @'+x.handle:''}</span></div>)}</div>
    <div className={s.h} style={{borderTop:'1px solid var(--line-soft)'}}>Discovery <span>· untracked coins named by watchers · 7d</span></div>
    <div className={s.fixed}>{!disc?<div className={s.empty}>Reading watcher posts…</div>:!disc.filter(c=>!c.tracked).length?<div className={s.empty}>None in the last seven days.</div>:<table className={s.t}><tbody>{disc.filter(c=>!c.tracked).slice(0,8).map(c=><tr key={c.key}><td style={{fontWeight:600}}>{c.symbol}<span className={`${s.mono} ${s.faint}`} style={{marginLeft:6,fontSize:10.5}}>{c.address?c.address.slice(0,4)+'…'+c.address.slice(-3):c.network}</span></td><td className={s.r}>{c.accounts}</td><td className={s.soft}>{c.identity}</td><td><button className={`${s.btn} ${s.btnSm}`} onClick={()=>set({tab:'data'})}>registry</button></td></tr>)}</tbody></table>}</div>
   </div>
   {/* Centre: tabs */}
   <div className={s.col}>
    <div className={s.tabs} role="tablist">{TABS.map(t=><button key={t.id} role="tab" aria-selected={state.tab===t.id} className={s.tab} onClick={()=>set({tab:t.id as Tab})}>{t.label}</button>)}
     {state.tab==='people'&&<span className={s.hint}>headings sort · <span className={s.key}>↑</span> <span className={s.key}>↓</span> move · <span className={s.key}>⏎</span> open · <span className={s.key}>c</span> pin coin · <span className={s.key}>/</span> command</span>}
     <button className={s.link} style={{marginLeft:'auto',fontSize:11,fontWeight:500}} onClick={()=>setHelp(h=>!h)}>{help?'hide definitions':'definitions'}</button></div>
    {(state.tab==='people'||state.tab==='posts')&&ringCounts.size>0&&<div className={s.ringRow}><span className={s.faint}>Rings · {focusCoin} · {fromDay}→{toDay}</span>{RINGS.filter(r=>ringCounts.get(r)).map(r=><button key={r} className={s.ringTag} data-ring={r} title={RING_DESCRIPTION[r]} aria-pressed={query.ring===r} onClick={()=>setQText(q=>(q.replace(/\bring:?\d\b/gi,'').trim()+(query.ring===r?'':' ring:'+r)).trim())}>{r} {RING_LABEL[r]} · {ringCounts.get(r)}</button>)}</div>}
    {help&&<div className={s.help}>Early = day of the account&#39;s first post on a coin counted from the coin&#39;s first saved contract post (d1 = same day, within the first week). Up = how many of the account&#39;s price-linked posts saw the pinned pool higher 24h later; Excess = the median move minus the coin&#39;s own median. Both need {MIN_LINKED_POSTS}+ linked posts; thinner rows are dimmed. Style is a text heuristic. Feed &quot;next close&quot; is the daily close after the post&#39;s day versus its own day, not the hourly event study. Association only, never attribution.<div style={{marginTop:6}}><b>Rings</b> (computed for {focusCoin} over {fromDay}→{toDay}; type <span className={s.key}>ring:N</span> to filter): {RINGS.map(r=><span key={r}><span className={s.ringTag} data-ring={r}>{r} {RING_LABEL[r]}</span> {RING_DESCRIPTION[r]}{ringCounts.get(r)?` (${ringCounts.get(r)})`:''}. </span>)}</div></div>}
    {state.tab==='people'&&<div className={s.scroll}>{!leaders?<div className={s.empty}>Reading saved rankings…</div>:!rows.length?<div className={s.empty}>No accounts match. Rankings appear after the next collection run; price-linked posts need a day of hourly candles.</div>:
     <table className={s.t}><thead>{table.getHeaderGroups().map(g=><tr key={g.id}><th style={{width:26}}>#</th>{g.headers.map(h=>{const d=h.column.getIsSorted();return <th key={h.id} className={RIGHT.has(h.id)?s.r:''} aria-sort={d==='asc'?'ascending':d==='desc'?'descending':'none'}><button onClick={h.column.getToggleSortingHandler()}>{flexRender(h.column.columnDef.header,h.getContext())}{d?(d==='asc'?' ▲':' ▼'):''}</button></th>;})}</tr>)}</thead>
     <tbody>{rows.slice(0,300).map((row,i)=>{const r=row.original,early=earlyCoins(r),ok=sampled(r),styles=postingStyles(r);return <tr key={r.account_id} className={`${s.row} ${ok?'':s.dim} ${state.account===r.account_id?s.on:''} ${i===cursor?s.cursor:''}`} onClick={()=>{setCursor(i);set({account:r.account_id});}} title={whyLeader(r)}>
      <td className={s.faint}>{i+1}</td><td><span className={s.link}>@{r.handle}</span></td><td className={`${s.r} ${s.soft}`}>{aud(r.followers)}</td>
      <td>{(()=>{const rg=rings.get(r.account_id);return rg?.ring?<span className={s.ringTag} data-ring={rg.ring} title={rg.evidence}>{rg.ring} {RING_LABEL[rg.ring]}</span>:<span className={s.faint}>{rg?'—':''}</span>;})()}</td>
      <td className={`${s.mono} ${s.soft}`}>{early.length?early.map(e=>e.coin+(e.day?' d'+e.day:'')).join(' '):'—'}</td>
      <td className={`${s.r} ${ok?(r.hit_rate!=null&&r.hit_rate>=.5?s.pos:s.neg):s.faint}`}>{ok?`${r.up}/${r.episodes}`:r.episodes?`${r.episodes} · thin`:'—'}</td>
      <td className={`${s.r} ${ok?tone(r.median_excess_24h):s.faint}`} style={{fontWeight:600}}>{ok?pct(r.median_excess_24h,0):'—'}</td>
      <td className={s.r}>{r.episodes}</td><td className={s.r}>{r.coins.length}</td><td className={s.r}>{r.days}</td>
      <td className={s.soft} title={styles.map(k=>STYLE_LABEL[k]).join(' · ')}>{styles.join(' · ')||'—'}</td><td className={s.faint}>{ago(r.last_at)}</td></tr>;})}</tbody></table>}</div>}
    {state.tab==='posts'&&<><div className={`${s.pad} ${s.faint}`} style={{fontSize:11,borderBottom:'1px solid var(--line-soft)',display:'flex'}}><span>X posts on {focusCoin} · {fromDay} → {toDay} · newest first · reposts hidden{query.handle?` · @${query.handle}`:''}</span><span style={{marginLeft:'auto'}}>{run?`${feed.length}${feed.length===400?'+':''} posts · next close vs post day`:''}</span></div>
     <div className={s.scroll}>{!run?<div className={s.empty}>Loading saved posts…</div>:!feed.length?<div className={s.empty}>No saved posts in scope.</div>:feed.map(p=><div key={p.id} className={s.feedRow} style={{gridTemplateColumns:'86px 128px minmax(0,1fr) 64px'}} onClick={()=>set({account:p.author_id})} title={p.text}>
      <span className={`${s.mono} ${s.faint}`}>{p.posted_at.slice(5,16).replace('T',' ')}</span><span className={s.clip} style={{fontWeight:600,color:'#bae6fd'}}>{rings.get(p.author_id)?.ring?<span className={s.ringDot} data-ring={rings.get(p.author_id)!.ring} title={RING_LABEL[rings.get(p.author_id)!.ring!]}>{rings.get(p.author_id)!.ring}</span>:null}@{p.handle}</span><span className={s.clip} style={{color:'var(--ink)'}}>{p.kind==='quote'?'↳ ':p.kind==='reply'?'↩ ':''}{p.text}</span><span className={`${s.r} ${tone(nextClose(p.posted_at))}`} style={{textAlign:'right'}}>{nextClose(p.posted_at)==null?'—':pct(nextClose(p.posted_at),0)}</span></div>)}</div></>}
    {state.tab==='timeline'&&<div className={s.scroll}>
     <div className={s.stats}><strong>{coinConfig(focusCoin).name} · {focusCoin}</strong><span>pinned pool {market?.selected?market.selected.id.slice(0,6)+'…':'—'}</span>
      <button disabled={!gain} onClick={()=>gain&&set({day:gain.day})}>largest daily gain <strong className={s.pos}>{gain?`+${gain.percent.toFixed(0)}% · ${gain.day.slice(5)}`:'—'}</strong></button>
      <span>accounts before it <strong>{beforeCount==null?'—':beforeCount}</strong></span>
      <span>earliest contract post <strong className={s.mono}>{eligible.anchor?eligible.anchor.slice(0,16).replace('T',' ')+'Z':coinConfig(focusCoin).address?'none saved':'native'}</strong></span>
      <button onClick={()=>set({tab:'data'})}>coverage <strong className={searched<days.length?s.neg:''}>{searched}/{days.length} days</strong></button>
      {(state.from||state.to)&&<button onClick={()=>set({from:null,to:null})}><strong>full period</strong></button>}</div>
     <div className={s.pad}>{!run?<div className={s.empty}>Loading saved posts…</div>:<CryptoRunChart days={days} market={series} posts={eligible.eligible} coverage={run.days??[]} from={fromDay} to={toDay} onRange={(a,b)=>set({from:a,to:b})} onInspect={d=>set({day:d})} highlight={state.highlight} inspecting={state.day}/>}
      {market&&<p className={s.faint} style={{fontSize:11,margin:'4px 0 0'}}>{market.source} · {market.note}</p>}</div>
     {state.day&&run&&<div className={s.pad}><CryptoBeforeMove day={state.day} posts={eligible.eligible} market={series} highlight={state.highlight} onHighlight={h=>set({highlight:h})} onSelect={id=>set({account:id})} onClose={()=>set({day:null,highlight:null})}/></div>}
     {!state.day&&<div className={s.empty}>Click a day on the chart, or the largest gain above, to list who posted in the 24 hours before it.</div>}
    </div>}
    {state.tab==='connections'&&<><div className={`${s.pad} ${s.faint}`} style={{fontSize:11,borderBottom:'1px solid var(--line-soft)'}}>Who quoted, replied to or mentioned whom on {focusCoin} · {fromDay} → {toDay} · by edge count · click a handle to open it</div>
     <div className={s.scroll}>{!run?<div className={s.empty}>Loading saved posts…</div>:!edges.length?<div className={s.empty}>No quote, reply or mention edges in scope.</div>:<table className={s.t}><thead><tr><th>From</th><th></th><th>To</th><th>Kind</th><th className={s.r}>Edges</th><th>Evidence</th></tr></thead><tbody>{edges.map((e,i)=><tr key={i}><td><button className={s.link} onClick={()=>set({account:e.source})}>@{e.from}</button></td><td className={s.faint}>→</td><td><button className={s.link} onClick={()=>set({account:e.target})}>@{e.to}</button></td><td className={s.soft}>{e.kind}</td><td className={s.r}>{e.weight}</td><td><a href={e.evidence} target="_blank" rel="noreferrer" style={{fontSize:11}}>post ↗</a></td></tr>)}</tbody></table>}</div></>}
    {state.tab==='data'&&<div className={`${s.scroll} ${s.pad}`}><CryptoDataView onCoin={c=>set({coin:c,tab:'timeline'})}/></div>}
   </div>
   {/* Right: account pane */}
   <div className={`${s.col} ${s.right}`}>
    {!state.account?<div className={s.empty}>No account selected. Click a row, a signal, a post or a handle, or type <span className={s.key}>@handle</span> in the command line. The pane opens here; the table stays put.</div>
    :!acct?<div className={s.empty}>Loading account…</div>:acct.status!=='ready'?<div className={s.empty}>No saved evidence for this account. <button className={s.link} onClick={()=>set({account:null})}>close</button></div>:<>
     <div className={s.paneHead}><strong>@{acct.account?.handle}</strong><span className={s.faint} style={{fontSize:11}}>{aud(acct.latest_profile?.followers??acct.account?.followers)} followers</span><a href={`https://x.com/i/user/${acct.account?.id}`} target="_blank" rel="noreferrer" style={{marginLeft:'auto',fontSize:11}}>x.com ↗</a><button className={s.link} style={{fontSize:11,fontWeight:500}} onClick={()=>set({account:null})}>close</button></div>
     <div className={s.paneWhy}>{(()=>{const rg=acct.account?rings.get(acct.account.id):undefined;return rg?.ring?<div style={{marginBottom:4}}><span className={s.ringTag} data-ring={rg.ring}>ring {rg.ring} · {RING_LABEL[rg.ring]}</span> <span className={s.faint}>on {focusCoin} {fromDay}→{toDay}: {rg.evidence}{rg.tags.length>1?' · also '+rg.tags.filter(t=>t!==rg.ring).map(t=>RING_LABEL[t]).join(', '):''}</span></div>:null;})()}{summary?whyLeader(summary):'saved posts only'}{acct.latest_profile?.bio?<span className={s.faint} style={{display:'block',marginTop:4,whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{acct.latest_profile.bio}</span>:null}</div>
     <div className={s.h}>Per coin</div>
     <div className={s.fixed}><table className={s.t}><thead><tr><th>Coin</th><th>Early</th><th className={s.r}>Posts</th><th className={s.r}>Up</th><th className={s.r}>Excess</th></tr></thead><tbody>{(acct.activity??[]).map(a=>{const role=summary?.roles.find(r=>r.coin===a.coin),imp=summary?.impact.find(i=>i.coin===a.coin);const ok=(imp?.episodes??0)>=MIN_LINKED_POSTS;return <tr key={a.coin} className={s.row} onClick={()=>set({coin:a.coin,tab:'timeline',highlight:acct.account?.id??null})}><td style={{fontWeight:600,color:'#7dd3fc'}}>{a.coin}</td><td className={`${s.mono} ${s.soft}`}>{role?.early&&role.day?'d'+role.day+(role.contract?' ·ca':''):'—'}</td><td className={s.r}>{a.posts}</td><td className={s.r}>{imp?`${Math.round((imp.share_up_24h??0)*imp.episodes)}/${imp.episodes}`:'—'}</td><td className={`${s.r} ${ok?tone(imp?.median_excess_24h):s.faint}`}>{ok?pct(imp?.median_excess_24h,0):'—'}</td></tr>;})}</tbody></table>{summary&&afterPosting(summary)==null&&summary.episodes>0&&<div className={`${s.pad} ${s.faint}`} style={{fontSize:11}}>Too few price-linked posts to read the outcome.</div>}</div>
     <div className={s.h}>Recent posts <span>· pool 24h after</span></div>
     <div className={s.scroll}>{!(acct.posts??[]).length?<div className={s.empty}>No saved posts.</div>:acct.posts!.map(p=><div key={p.id} className={s.post}><div className={s.meta}><span className={s.mono}>{String(p.posted_at).slice(5,16).replace('T',' ')}</span>{(p.coins??[]).map(c=><button key={c} className={s.link} style={{fontSize:11}} onClick={()=>pinCoin(c)}>{c}</button>)}<span style={{marginLeft:'auto'}} className={tone(p.return_24h)}>{p.return_24h==null?'':pct(p.return_24h,0)}</span></div><details><summary style={{listStyle:'none',cursor:'pointer'}}><p>{p.text}</p></summary><p>{p.text}</p><a href={p.url} target="_blank" rel="noreferrer" style={{fontSize:11}}>Open source ↗</a></details></div>)}</div>
    </>}
   </div>
  </div>
  <div className={s.bottom}><span>Data</span>
   {status?<span>coverage <b>{cov.filter(c=>c.hourly_hours>0).length}/{cov.length} coins</b> · candles to <b className={s.mono}>{latest?String(latest).slice(5,13).replace('T',' ')+'h':'—'}</b> · posts to <b className={s.mono}>{status.lastCollection?String(status.lastCollection).slice(5,16).replace('T',' ')+'Z':'—'}</b>{noPool.length?<> · <span className={s.warn}>{noPool.join(', ')} no pool yet</span></>:null} · origin <b>{cov.filter(c=>c.origin?.status==='search_exhausted').length}/{cov.filter(c=>c.origin).length} done</b></span>:<span className={s.faint}>reading status…</span>}
   <button className={`${s.btn} ${s.btnSm}`} onClick={()=>set({tab:'data'})}>registry · ledgers · coverage</button>
   <span style={{marginLeft:'auto'}} className={s.faint}>Research context only, not investment advice.</span></div>
 </div>;
}
