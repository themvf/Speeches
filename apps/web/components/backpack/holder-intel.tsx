'use client';
import {useEffect,useMemo,useRef,useState} from 'react';
import {numeric} from '@/lib/backpack';
import {rawToDecimal,share,type IntelPayload,type IntelRow} from '@/lib/bp-intel';
import s from './monitor.module.css';

const money=(v:unknown)=>numeric(v)===null?'N/A':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',notation:'compact',maximumFractionDigits:2}).format(Number(v));
const count=(v:unknown)=>numeric(v)===null?'N/A':new Intl.NumberFormat('en-US').format(Number(v));
const pct=(v:number|null)=>v===null?'N/A':v>0&&v<0.001?'<0.1%':`${(v*100).toFixed(1)}%`;
const stamp=(v:unknown)=>v?String(v).replace('T',' ').slice(0,16)+' UTC':'—';
const short=(a:unknown)=>{const t=String(a??'');return t.length>12?`${t.slice(0,4)}…${t.slice(-4)}`:t;};
const amount=(raw:unknown,decimals:unknown)=>{const d=rawToDecimal(raw,decimals);if(d===null)return 'N/A';const n=Number(d);return n===0||Math.abs(n)>=0.01?new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:2}).format(n):d;};
const signed=(raw:unknown,decimals:unknown)=>{const text=amount(raw,decimals);return text!=='N/A'&&!String(text).startsWith('-')&&String(raw)!=='0'?`+${text}`:text;};
const TIER:Record<string,string>={parsed_swap:'Parsed swap',inferred_swap:'Inferred swap',unclassified:'Unclassified',not_applicable:'—'};
const RULE:Record<string,string>={new_position:'New position',multiple_buyers:'Multiple buyers',accumulation:'Accumulation',major_sale:'Major sale'};
const tokenName=(r:IntelRow,side?:'input'|'output')=>{const mint=String(side?r[`${side}_mint`]??'':r.mint??'');const symbol=side?r[`${side}_symbol`]:r.symbol;return mint==='native'?'SOL':String(symbol??short(mint));};
type Tab='holdings'|'purchases'|'roster'|'alerts'|'social';
const COVERAGE:Record<string,string>={complete:'Searched',partial:'Partly searched',earlier_query:'Earlier search only',not_searched:'Not searched'};
const intelUrl=(scope:string,section:string,extra:Record<string,string>={})=>{const p=new URLSearchParams(scope);p.set('section',section);for(const [k,v] of Object.entries(extra))p.set(k,v);return `/api/market/crypto/backpack?${p}`;};
type Detail={kind:'token'|'wallet';key:string;data?:IntelPayload;error?:string};

export function HolderIntel(){
 const [cohort,setCohort]=useState(''),[run,setRun]=useState(''),[tab,setTab]=useState<Tab>('holdings');
 const [data,setData]=useState<IntelPayload|null>(null),[roster,setRoster]=useState<IntelPayload|null>(null),[error,setError]=useState('');
 const [social,setSocial]=useState<IntelPayload|null>(null);
 const [detail,setDetail]=useState<Detail|null>(null),[show,setShow]=useState({sol:false,stable:false,spam:false,bp:false});
 // Latest detail request wins; a scope change or Close invalidates anything still in flight.
 const detailRequest=useRef(0);
 const [activityAll,setActivityAll]=useState(false),[holdingFilter,setHoldingFilter]=useState('all');
 const scope=useMemo(()=>{const p=new URLSearchParams();if(cohort)p.set('cohort',cohort);if(run)p.set('run',run);return p.toString();},[cohort,run]);
 const url=(section:string,extra:Record<string,string>={})=>intelUrl(scope,section,extra);
 useEffect(()=>{let live=true;setData(null);setRoster(null);setSocial(null);setError('');detailRequest.current++;setDetail(null);
  fetch(intelUrl(scope,'overview')).then(async r=>{if(!r.ok)throw new Error((await r.json()).error??'Unavailable');return r.json();}).then(d=>{if(live)setData(d);}).catch(e=>{if(live)setError(e.message);});
  return()=>{live=false;};
 },[scope]);
 useEffect(()=>{if(tab!=='roster'||roster||!data?.meta?.version)return;let live=true;
  fetch(intelUrl(scope,'roster')).then(r=>r.json()).then(d=>{if(live)setRoster(d);}).catch(()=>{if(live)setError('Roster unavailable');});return()=>{live=false;};
 },[tab,roster,data,scope]);
 useEffect(()=>{if(tab!=='social'||social||!data?.meta?.version)return;let live=true;
  fetch(intelUrl(scope,'social',{days:'30'})).then(r=>r.json()).then(d=>{if(live)setSocial(d);}).catch(()=>{if(live)setError('Posts on X unavailable');});return()=>{live=false;};
 },[tab,social,data,scope]);
 async function open(kind:'token'|'wallet',key:string){
  const request=++detailRequest.current;
  setDetail({kind,key});
  try{const r=await fetch(url(kind,kind==='token'?{mint:key,days:'30'}:{wallet:key,days:'30'}));const d=await r.json();
   if(request===detailRequest.current)setDetail(r.ok?{kind,key,data:d}:{kind,key,error:d.error});}
  catch{if(request===detailRequest.current)setDetail({kind,key,error:'Could not load detail.'});}
 }
 const meta=data?.meta,version=meta?.version,cov:IntelRow=meta?.coverage??{};
 const members=numeric(cov.members),read=(numeric(cov.read_complete)??0)+(numeric(cov.read_partial)??0);
 const incomplete=members!==null&&read<members;
 const warnings=[
  !meta?.run&&'No stored portfolio read for this cohort yet.',
  incomplete&&`Portfolio read covers ${read} of ${members} wallets: holder counts are observed lower bounds; a missing wallet is not a non-holder.`,
  (numeric(cov.stale_prices)??0)>0&&`${count(cov.stale_prices)} holdings carry prices older than one hour; their values are withheld.`,
  (numeric(cov.history_gaps)??0)>0&&`${count(cov.history_gaps)} wallets have capped or gapped transaction history.`,
  (numeric(cov.history_pending)??0)>0&&`${count(cov.history_pending)} wallets have not finished the 30-day history backfill.`,
  (numeric(cov.reconciliation_discrepancies)??0)>0&&`${count(cov.reconciliation_discrepancies)} balance changes in the last 7 days are not explained by classified events.`,
 ].filter(Boolean) as string[];
 const overlap=(data?.rows??[]).filter(r=>(show.bp||!r.is_bp)&&(show.sol||!r.is_sol)&&(show.stable||!r.is_stable)&&(show.spam||r.spam_class!=='suspected_spam')&&r.asset_class!=='nft');
 const activity=(data?.extra.activity??[]).filter(r=>activityAll||r.kind==='swap');
 const csv=(section:string,extra:Record<string,string>={})=><a className={s.csv} href={url(section,{...extra,format:'csv'})}>CSV</a>;
 const WalletLink=({address}:{address:unknown})=><span className={s.mono}><button className={s.linkButton} onClick={()=>open('wallet',String(address))}>{short(address)}</button> <a href={`https://solscan.io/account/${address}`} target="_blank" rel="noreferrer" aria-label="Open wallet on Solscan">↗</a></span>;
 const TokenLink=({row,side}:{row:IntelRow;side?:'input'|'output'})=>{const mint=String(side?row[`${side}_mint`]:row.mint);return <button className={s.linkButton} onClick={()=>open('token',mint)} title={mint}>{tokenName(row,side)}</button>;};
 const Tx=({sig}:{sig:unknown})=><a className={s.mono} href={`https://solscan.io/tx/${sig}`} target="_blank" rel="noreferrer">{short(sig)}</a>;
 const Flags=({r}:{r:IntelRow})=><>{r.new_position?<span className={s.badge}>new position</span>:null}{r.first_observed_purchase?<span className={s.badge}>first observed</span>:null}{r.re_entry?<span className={s.badge}>re-entry</span>:null}{r.recently_launched?<span className={s.badge}>recent launch</span>:null}{r.pre_membership?<span className={s.badgeMuted}>before tracking</span>:null}</>;
 return <section className={`${s.panel} ${s.intel}`} id="bp-holder-intelligence">
  <div className={s.sectionHead}><div><span className={s.eyebrow}>BP HOLDER INTELLIGENCE</span><h2>What the largest BP holders own and buy</h2><p>Wallets, not people: one person can control several wallets and one custody wallet can hold for many. Direct Solana token balances only; DeFi positions, exchange accounts and other chains are outside scope.</p></div>
  {meta&&<div className={s.intelControls}><label>Cohort<select value={cohort||String(version?.version_id??'')} onChange={e=>{setCohort(e.target.value);setRun('');setDetail(null);}}>{meta.versions.map(v=><option key={String(v.version_id)} value={String(v.version_id)}>{v.kind==='original'?'Original':'Current'} · v{String(v.version_id)} · {String(v.source_date)} · {String(v.size)} wallets</option>)}</select></label>
   <label>Portfolio read<select value={run||String(meta.run?.run_id??'')} onChange={e=>setRun(e.target.value)}>{meta.runs.map(r=><option key={String(r.run_id)} value={String(r.run_id)}>{stamp(r.started_at)} · {String(r.status)}</option>)}</select></label></div>}</div>
  {error&&<div className={s.notice} role="alert">{error}</div>}
  {!data&&!error&&<p className={s.muted}>Loading stored holder intelligence…</p>}
  {data&&data.status!=='ready'&&<div className={s.notice}><strong>{data.status==='schema_pending'?'Collector schema awaiting migration':data.status==='no_cohort'?'No cohort yet':'Data connection not configured'}</strong><p>{data.status==='no_cohort'?'A cohort version is created after the daily capture completes a supply-reconciled BP holder enumeration.':'Nothing is shown until stored observations exist.'}</p></div>}
  {data?.status==='ready'&&meta&&<>
   <div className={s.strip}>
    <span><i className={meta.run?s.dot:s.offDot}/> {version?.kind==='original'?'Original':'Current'} cohort v{String(version?.version_id)} · capture {String(version?.source_date)} · {String(version?.size)} wallets{version?.entrant_cap_bound?' · entrant cap bound':''}</span>
    <span>Portfolio read {stamp(meta.run?.started_at)}</span>
    <span>Reads: {count(cov.read_complete)} complete · {count(cov.read_partial)} partial · {count(cov.read_unavailable)} unavailable · {count(cov.read_oversized)} oversized · {count(cov.read_missing)} not read</span>
    <span>History: {count(cov.history_complete)} complete · last poll {stamp(cov.newest_poll)}</span>
    <span title={meta.live.detail}>Live monitoring: {meta.live.enabled?'on':'off (hourly polling)'}</span>
   </div>
   {warnings.length>0&&<div className={s.notice} role="status"><strong>Collection is incomplete or stale</strong><ul>{warnings.map(w=><li key={w}>{w}</li>)}</ul></div>}
   <div className={s.ranges} role="tablist" aria-label="Holder intelligence views">{([['holdings','Common holdings'],['purchases','Recent purchases'],['roster','Holder roster'],['alerts',`Alerts (${data.extra.alerts?.length??0})`],['social','On X']] as const).map(([key,label])=><button key={key} role="tab" aria-selected={tab===key} aria-pressed={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>

   {tab==='holdings'&&<div><div className={s.intelBar}><p>Holders counted from the selected portfolio read. Meaningful means at least {money(meta.meaningfulUsd)} at a fresh price. Weight is the share of each holder&apos;s priced, in-scope portfolio. Common ownership is not evidence of buying; flows below come from classified swaps after each wallet joined tracking.</p>
    <div className={s.toggles}>{([['bp','BP'],['sol','SOL'],['stable','Stablecoins'],['spam','Suspected spam']] as const).map(([k,label])=><label key={k}><input type="checkbox" checked={show[k]} onChange={e=>setShow({...show,[k]:e.target.checked})}/>{label}</label>)}{csv('overlap')}</div></div>
    <div className={s.tableWrap}><table><thead><tr><th>Token</th><th>Holders</th><th>Meaningful</th><th>Combined value</th><th>Median weight</th><th>Largest owner</th><th>New-position buyers 1h / 24h / 7d</th><th>Buyers / sellers 7d</th><th>Net flow 7d</th><th>Price · liquidity</th><th>Quality</th></tr></thead>
    <tbody>{overlap.slice(0,200).map(r=><tr key={String(r.mint)}><td><TokenLink row={r}/> <span className={s.muted}>{String(r.name??'').slice(0,28)}</span>{r.asset_class==='position'&&<span className={s.badgeMuted} title={String(r.class_reason??'')}>position</span>}{r.spam_class==='suspected_spam'&&<span className={s.badgeWarn} title={String(r.spam_reason??'')}>spam?</span>}</td>
     <td>{count(r.holders)} <span className={s.muted}>{pct(share(r.holders,r.cohort_size))}</span></td><td>{count(r.meaningful_holders)}</td><td>{money(r.combined_value_usd)}</td><td>{pct(numeric(r.median_weight))}</td><td>{pct(numeric(r.largest_owner_share))}</td>
     <td>{count(r.new_buyers_1h??0)} / {count(r.new_buyers_24h??0)} / {count(r.new_buyers_7d??0)}</td><td>{count(r.buyers_7d??0)} / {count(r.sellers_7d??0)}{numeric(r.inferred_purchases_7d)?<span className={s.muted}> · {count(r.inferred_purchases_7d)} inferred</span>:null}</td>
     <td>{signed(r.net_raw_7d,r.decimals)}{(numeric(r.unpriced_purchases_7d)??0)+(numeric(r.unpriced_sales_7d)??0)>0&&<span className={s.muted}> · some unpriced</span>}</td>
     <td>{money(r.price)} · {money(r.liquidity_usd)}</td><td>{r.stale_price?'Stale price':''}{numeric(r.unpriced_holders)?` ${count(r.unpriced_holders)} unpriced`:''}{numeric(r.partial_visibility)?` · ${count(r.partial_visibility)} confidential`:''}{!r.stale_price&&!numeric(r.unpriced_holders)&&!numeric(r.partial_visibility)?'Priced':''}</td></tr>)}</tbody></table>
    {!overlap.length&&<p className={s.tableEmpty}>No holdings in this read with the current filters.</p>}</div>
    <p className={s.muted}>Showing {Math.min(overlap.length,200)} of {overlap.length} tokens · {count(data.rows[0]?.wallets_read)} of {count(data.rows[0]?.cohort_size??cov.members)} wallets read. NFTs are excluded from totals.</p></div>}

   {tab==='purchases'&&<div><div className={s.intelBar}><p>Swaps are a disposal of the input and an acquisition of the output. Parsed swaps come from Helius&apos;s swap parser; inferred swaps are balance changes through a known DEX program. USD is an execution-time estimate from the trade&apos;s own stablecoin or SOL leg; other trades stay unpriced. Transfers and airdrops are never purchases.</p>
    <div className={s.toggles}><label><input type="checkbox" checked={activityAll} onChange={e=>setActivityAll(e.target.checked)}/>Transfers, wraps and unclassified</label>{csv('activity')}</div></div>
    <div className={s.tableWrap}><table><thead><tr><th>Time</th><th>Wallet</th><th>Kind · tier</th><th>Sold</th><th>Bought</th><th>USD</th><th>Flags</th><th>Evidence</th></tr></thead><tbody>{activity.slice(0,300).map(r=><tr key={`${r.signature}-${r.wallet_address}-${r.event_index}`}>
     <td>{stamp(r.block_time)}</td><td><WalletLink address={r.wallet_address}/></td><td>{String(r.kind).replace('_',' ')} · <span className={r.tier==='inferred_swap'?s.badgeWarn:r.tier==='parsed_swap'?s.badge:s.badgeMuted}>{TIER[String(r.tier)]}</span></td>
     <td>{r.input_mint?<>{amount(r.input_raw,r.input_decimals)} <TokenLink row={r} side="input"/></>:'—'}</td><td>{r.output_mint?<>{amount(r.output_raw,r.output_decimals)} <TokenLink row={r} side="output"/></>:'—'}</td>
     <td title={String(r.valuation_source??'')}>{r.valuation_status==='estimated'?money(r.usd_value):r.valuation_status==='unpriced'?'Unpriced':'—'}</td><td><Flags r={r}/></td><td><Tx sig={r.signature}/> <details><summary>why</summary><p>{String(r.detail)}</p><p>{String(r.venue??'')} · pre-balance from {String(r.pre_balance_source??'n/a')} · {String(r.finality)} · {String(r.parser_version)}</p></details></td></tr>)}</tbody></table>
    {!activity.length&&<p className={s.tableEmpty}>No classified activity for this cohort in the last 7 days.</p>}</div></div>}

   {tab==='roster'&&<div><div className={s.intelBar}><p>Filtered ranking of owner-aggregated BP balances from capture {String(version?.source_date)}. Only confirmed/high-confidence system labels exclude a wallet; unlabelled large wallets are not assumed to be exchanges. Wallets enter at rank {String(version?.target_size)} or better and leave after {String(version?.exit_runs)} consecutive captures below rank {String(version?.exit_rank)}.</p>
    <div className={s.toggles}>{csv('roster')}{csv('roster',{ranking:'raw'})}<span className={s.muted}>raw ranking</span></div></div>
    {!roster&&<p className={s.muted}>Loading roster…</p>}
    {roster&&<div className={s.tableWrap}><table><thead><tr><th>Rank</th><th>Wallet</th><th>BP</th><th>Membership</th><th>Label at capture</th><th>Current label</th></tr></thead><tbody>{roster.rows.map(r=><tr key={String(r.wallet_address)} className={r.member?undefined:s.muted}>
     <td>{r.rank==null?'—':String(r.rank)}{r.previous_rank!=null&&r.previous_rank!==r.rank?<span className={s.muted}> (was {String(r.previous_rank)})</span>:null}</td><td><WalletLink address={r.wallet_address}/></td><td>{amount(r.raw_balance,r.decimals)}</td>
     <td>{String(r.event)}{r.exit_reason?` · ${String(r.exit_reason).replaceAll('_',' ')}`:''}{Number(r.below_exit_runs)>0&&r.member?` · ${String(r.below_exit_runs)} run(s) below exit`:''}</td><td>{String(r.label??'Unknown')}{r.label_confidence?` · ${String(r.label_confidence)}`:''}</td><td>{String(r.current_label??'—')}</td></tr>)}</tbody></table>
     <details><summary>Raw top 200 (no exclusions) · {roster.extra.raw?.length??0} owners</summary><table><thead><tr><th>Rank</th><th>Wallet</th><th>BP</th><th>Label</th><th>Excluded</th></tr></thead><tbody>{(roster.extra.raw??[]).map(r=><tr key={String(r.rank)}><td>{String(r.rank)}</td><td><WalletLink address={r.wallet_address}/></td><td>{String(r.bp_balance??'N/A')}</td><td>{String(r.label)}{r.label_confidence?` · ${String(r.label_confidence)}`:''}</td><td>{r.excluded?'Yes':'No'}</td></tr>)}</tbody></table></details></div>}</div>}

   {tab==='alerts'&&<div><div className={s.intelBar}><p>Alerts use finalized, alert-eligible swaps by members of this cohort version after they joined tracking. USD rules need an execution-time valuation. Alerts update in place when more wallets join; they are not delivered outside this page.</p><div className={s.toggles}>{csv('alerts')}</div></div>
    <div className={s.tableWrap}><table><thead><tr><th>Window</th><th>Rule</th><th>Token</th><th>Wallets</th><th>Quantity</th><th>USD</th><th>Lowest tier</th><th>Data</th><th>Transactions</th></tr></thead><tbody>{(data.extra.alerts??[]).map(a=><tr key={String(a.alert_key)}>
     <td>{stamp(a.window_start)}<br/><span className={s.muted}>to {stamp(a.window_end)}</span></td><td>{RULE[String(a.rule)]??String(a.rule)}</td><td><TokenLink row={a}/></td>
     <td>{count(a.wallet_count)}{numeric(a.inferred_wallets)?<span className={s.muted}> · {count(a.inferred_wallets)} inferred</span>:null}</td><td>{amount(a.quantity_raw,a.decimals)}</td><td>{a.valuation_status==='unpriced'?'Unpriced':`${money(a.usd_value)}${a.valuation_status==='partially_priced'?' (partial)':''}`}</td>
     <td>{TIER[String(a.lowest_tier)]}</td><td>{String(a.finality)} · through {stamp(a.data_through)}</td><td><details><summary>{Array.isArray(a.signatures)?a.signatures.length:0} signatures</summary>{(Array.isArray(a.signatures)?a.signatures:[]).map(sig=><div key={String(sig)}><Tx sig={sig}/></div>)}<p>{String(a.detail)}</p></details></td></tr>)}</tbody></table>
    {!(data.extra.alerts??[]).length&&<p className={s.tableEmpty}>No alerts for this cohort version.</p>}</div></div>}

   {tab==='social'&&<div><div className={s.intelBar}><p>Posts on X that match Backpack&apos;s token: the contract address, or $BP alongside Backpack or Solana. Search results that only contain the word &ldquo;backpack&rdquo;, the copycat $BACKPACK cashtag on its own, and posts naming another token&apos;s address are left out. A day the search has not reached reads &ldquo;not searched&rdquo;, never zero; &ldquo;earlier search only&rdquo; means it was searched before $BP was added (from 3 October), so its counts are lower bounds. Holder counts come from each day&apos;s capture; top-200 BP buyers and sellers only from wallets with transaction history{social?.extra.history?.[0]?.through?`, observed through ${stamp(social.extra.history[0].through)}`:''}.</p>
    <div className={s.toggles}>{csv('social',{days:'30'})}</div></div>
    {!social&&<p className={s.muted}>Loading posts on X…</p>}
    {social?.extra.social_status&&<p className={s.tableEmpty}>The X tracker has not been set up in this database.</p>}
    {social&&!social.extra.social_status&&<>
     <div className={s.tableWrap}><table><thead><tr><th>Day (UTC)</th><th>Backpack posts</th><th>Accounts</th><th>Copy-paste</th><th>X search</th><th>BP holders</th><th>$100+ holders</th><th>Change</th><th>Top 200 in / out</th><th>Top-200 BP buyers / sellers</th></tr></thead>
     <tbody>{social.rows.map(r=><tr key={String(r.day)}><td>{String(r.day)}</td>
      <td>{r.posts==null?<span className={s.muted}>not searched</span>:count(r.posts)}{r.x_coverage==='partial'&&r.posts!=null?<span className={s.muted}> so far</span>:null}</td>
      <td>{r.authors==null?'—':count(r.authors)}</td><td>{numeric(r.copy_paste_posts)?<span className={s.badgeWarn}>{count(r.copy_paste_posts)}</span>:r.copy_paste_posts==null?'—':'0'}</td>
      <td title={`${count(r.x_windows_done)} of ${count(r.x_windows)} search windows searched to the end`}>{COVERAGE[String(r.x_coverage)]}</td>
      <td>{r.holders==null?'—':count(r.holders)}</td><td>{r.holders_over_100==null?'—':count(r.holders_over_100)}</td>
      <td>{r.holders_change==null?'—':`${Number(r.holders_change)>0?'+':''}${count(r.holders_change)}`}</td>
      <td>{r.cohort_entered==null?'—':`+${count(r.cohort_entered)} / −${count(r.cohort_left)}`}</td>
      <td>{r.bp_buyers==null?<span className={s.muted}>not observed</span>:`${count(r.bp_buyers)} / ${count(r.bp_sellers)}`}</td></tr>)}</tbody></table></div>
     <h3>Copy-paste campaigns</h3>
     <p className={s.muted}>Near-identical posts from three or more accounts. Who paid for or coordinated them is not known from the posts.</p>
     {(social.extra.campaigns??[]).length?<div className={s.tableWrap}><table><thead><tr><th>First · last</th><th>Accounts</th><th>Posts</th><th>Template</th></tr></thead><tbody>{(social.extra.campaigns??[]).map(g=><tr key={String(g.key)}>
      <td>{stamp(g.first)}<br/><span className={s.muted}>{stamp(g.last)}</span></td><td>{count(g.accounts)}</td><td>{count(g.posts)}</td>
      <td>{String(g.sample)}{g.url?<> <a href={String(g.url)} target="_blank" rel="noreferrer">first post ↗</a></>:null}<br/><span className={s.muted}>{(Array.isArray(g.handles)?g.handles:[]).map(h=>`@${h}`).join(' · ')}</span></td></tr>)}</tbody></table></div>:<p className={s.tableEmpty}>No copy-paste campaign in this window.</p>}
     <h3>Coins the top holders share, on X</h3>
     <p className={s.muted}>Coins in this portfolio read that the X tracker also follows. Posts count only where they match that coin. A ticker match means the tracker follows that coin on another chain, which is not the same asset as the Solana token.</p>
     {(social.extra.shared??[]).length?<div className={s.tableWrap}><table><thead><tr><th>Coin</th><th>Linked by</th><th>Top-200 holders ($100+)</th><th>Posts on X · 7 days</th><th>Accounts</th><th>Searched through</th></tr></thead><tbody>{(social.extra.shared??[]).map(c=><tr key={String(c.mint)}>
      <td><a href={`/market/crypto?coin=${encodeURIComponent(String(c.coin))}`}>{String(c.coin_label??c.coin)}</a></td><td>{String(c.linked_by)}</td>
      <td>{count(c.holders)} <span className={s.muted}>({count(c.meaningful_holders)})</span></td><td>{count(c.posts_7d)}{numeric(c.unsearched_windows_7d)?<span className={s.muted}> · {count(c.unsearched_windows_7d)} windows not yet searched</span>:null}</td>
      <td>{count(c.accounts_7d)}</td><td>{stamp(c.searched_through)}</td></tr>)}</tbody></table></div>:<p className={s.tableEmpty}>None of the coins in this read are followed by the X tracker.</p>}
    </>}</div>}

   {detail&&<div className={s.intelDetail} role="region" aria-label={`${detail.kind} detail`}><div className={s.sectionHead}><h3>{detail.kind==='token'?`Token ${detail.key==='native'?'SOL':short(detail.key)}`:`Wallet ${short(detail.key)}`}</h3><div className={s.toggles}>{csv(detail.kind,detail.kind==='token'?{mint:detail.key}:{wallet:detail.key})}<button onClick={()=>{detailRequest.current++;setDetail(null);}}>Close</button></div></div>
    {detail.error&&<p role="alert">{detail.error}</p>}{!detail.data&&!detail.error&&<p className={s.muted}>Loading…</p>}
    {detail.data&&detail.kind==='token'&&<><p><code>{detail.key}</code> · {String(detail.data.extra.asset?.[0]?.name??'No metadata')} · {String(detail.data.extra.asset?.[0]?.asset_class??'unclassified')}{(detail.data.extra.asset?.[0]?.extension_flags as string[]|undefined)?.length?` · Token-2022: ${(detail.data.extra.asset?.[0]?.extension_flags as string[]).join(', ')}`:''}</p>
     <div className={s.twoColumns}><div><h4>Tracked holders ({detail.data.rows.length})</h4><table><thead><tr><th>Rank</th><th>Wallet</th><th>Amount</th><th>Value</th></tr></thead><tbody>{detail.data.rows.map(r=><tr key={String(r.wallet_address)}><td>{String(r.rank??'—')}</td><td><WalletLink address={r.wallet_address}/></td><td>{String(r.amount??'N/A')}{r.balance_visibility==='partial'?' (public part)':''}</td><td>{money(r.value_usd)}</td></tr>)}</tbody></table></div>
     <div><h4>Purchases and sales · 30 days</h4><table><thead><tr><th>Time</th><th>Wallet</th><th>Side</th><th>Amount</th><th>Tier</th><th>Tx</th></tr></thead><tbody>{(detail.data.extra.activity??[]).filter(r=>r.kind==='swap'||r.kind==='transfer_in'||r.kind==='transfer_out').map(r=>{const buy=r.output_mint===detail.key;return <tr key={`${r.signature}-${r.wallet_address}-${r.event_index}`}><td>{stamp(r.block_time)}</td><td><WalletLink address={r.wallet_address}/></td><td>{r.kind==='swap'?(buy?'Bought':'Sold'):String(r.kind).replace('_',' ')}</td><td>{buy?amount(r.output_raw,r.output_decimals):amount(r.input_raw,r.input_decimals)}</td><td>{TIER[String(r.tier)]}</td><td><Tx sig={r.signature}/></td></tr>;})}</tbody></table>
      <h4>Alerts</h4>{(detail.data.extra.alerts??[]).length?(detail.data.extra.alerts??[]).map(a=><p key={String(a.alert_key)}>{RULE[String(a.rule)]} · {stamp(a.window_start)} · {count(a.wallet_count)} wallets</p>):<p className={s.muted}>None.</p>}</div></div></>}
    {detail.data&&detail.kind==='wallet'&&(()=>{const readRow=detail.data.extra.read?.[0],history=detail.data.extra.history?.[0];const rows=detail.data.rows.filter(r=>holdingFilter==='all'||(holdingFilter==='meaningful'&&(numeric(r.value_usd)??0)>=(meta.meaningfulUsd))||(holdingFilter==='unpriced'&&r.value_usd==null)||(holdingFilter==='spam'&&(r.spam_class==='suspected_spam'||r.dust===true))||(holdingFilter==='core'&&(r.is_sol||r.is_stable)));
     return <><p>Read {String(readRow?.status??'not in this run')} · SPL {String(readRow?.spl_status??'—')} · Token-2022 {String(readRow?.token2022_status??'—')} · SOL {String(readRow?.sol_status??'—')} · priced portfolio {money(readRow?.priced_value_usd)} ({count(readRow?.unpriced_holdings)} unpriced) · history {String(history?.backfill_status??'not started')} since {stamp(history?.earliest_retrieved)}{Array.isArray(history?.gaps)&&history.gaps.length?` · ${history.gaps.length} recorded gap(s)`:''}</p>
      <div className={s.toggles}>{([['all','All holdings'],['meaningful',`≥ ${money(meta.meaningfulUsd)}`],['unpriced','Unpriced'],['spam','Spam / dust'],['core','SOL & stablecoins']] as const).map(([k,label])=><label key={k}><input type="radio" name="holding-filter" checked={holdingFilter===k} onChange={()=>setHoldingFilter(k)}/>{label}</label>)}</div>
      <div className={s.tableWrap}><table><thead><tr><th>Token</th><th>Amount</th><th>Value</th><th>Pricing</th><th>Notes</th></tr></thead><tbody>{rows.map(r=><tr key={String(r.mint)}><td><TokenLink row={r}/></td><td>{String(r.amount??'N/A')}</td><td>{money(r.value_usd)}</td><td>{String(r.pricing_status)}</td><td>{String(r.classification_reason||'')}{r.frozen?' · frozen':''}</td></tr>)}</tbody></table></div></>;})()}
   </div>}
  </>}
 </section>;
}
