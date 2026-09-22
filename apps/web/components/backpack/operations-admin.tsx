'use client';
import {useEffect,useState} from 'react';
import type {Row} from '@/lib/backpack';
import s from './monitor.module.css';
type Operations={status:string;checks?:Row[];runs?:Row[];assets?:Row;reconciliation?:Row[];storage?:Row[];alerts?:Row[]};
const credentials:Record<string,string>={database:'DATABASE_URL',helius_finalized_rpc:'HELIUS_API_KEY',helius_das:'HELIUS_API_KEY',jupiter_price:'JUPITER_API_KEY',alpaca_sip:'ALPACA_API_KEY + ALPACA_SECRET_KEY (SIP access)',secondary_solana_rpc:'SOLANA_VALIDATION_RPC_URL or public fallback'};
const show=(v:unknown)=>v===null||v===undefined?'N/A':String(v);
export function OperationsAdmin(){
 const [data,setData]=useState<Operations|null>(null),[error,setError]=useState('');
 async function load(){setError('');try{const r=await fetch('/api/admin/backpack?operations=1',{cache:'no-store'});if(!r.ok)throw Error();setData(await r.json());}catch{setError('Stored operational checks could not be loaded.');}}
 useEffect(()=>{void load();},[]);
 return <section className={`${s.panel} ${s.operationsPanel}`}><h2>Production readiness</h2><p>Last worker-reported checks, not live credential probes. No paid provider calls occur here.</p><button onClick={()=>void load()}>Reload stored status</button>
 {error&&<p role="alert">{error}</p>}{!data&&!error&&<p>Loading stored evidence…</p>}
 {data&&<><p>Schema: {data.status==='ready'?'Available':data.status.replaceAll('_',' ')} · Approved securities: {show(data.assets?.approved)} · Latest capture: {show(data.assets?.latest_capture)}</p>
 <div className={s.tableWrap}><table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th><th>Checked at</th></tr></thead><tbody>{data.checks?.map(c=><tr key={show(c.check_name)}><td>{show(c.check_name).replaceAll('_',' ')}<br/><small>{credentials[show(c.check_name)]??'Registry / public source'}</small></td><td>{Date.now()-Date.parse(show(c.checked_at))>48*3600000?'Stale':show(c.status)}</td><td>{show(c.detail)}</td><td>{show(c.checked_at)}</td></tr>)}</tbody></table></div>
 {!data.checks?.length&&<p>No readiness checks recorded.</p>}
 <h3>Recent captures</h3>{data.runs?.map((r,i)=><p key={i}>{show(r.snapshot_date)} · {show(r.status)} · {show(r.assets_succeeded)} succeeded / {show(r.assets_failed)} failed / {show(r.assets_skipped)} preserved · completed {show(r.completed_at)}</p>)}
 {!data.runs?.length&&<p>No capture runs recorded.</p>}
 <h3>First-capture reconciliation</h3><p>Execution success is not research sign-off. Review the workflow audit artifact and primary sources before treating observations as authoritative.</p>
 <div className={s.tableWrap}><table><thead><tr><th>Asset / date</th><th>Holder enumeration</th><th>Supply / holder sum</th><th>AUM / reproduced AUM</th><th>Equity reference timestamp</th></tr></thead><tbody>{data.reconciliation?.map(r=><tr key={show(r.token_symbol)}><td>{show(r.token_symbol)} · {show(r.date)}<br/>{show(r.quality_status)} coverage</td><td>{r.aggregates_validated?'Reconciled within tolerance':'Unavailable / incomplete'}</td><td>{show(r.token_supply)} / {show(r.balance_tokens)}</td><td>{show(r.reference_aum_usd)} / {show(r.reproduced_aum)}</td><td>{show(r.underlying_price_timestamp)}</td></tr>)}</tbody></table></div>
 {!data.reconciliation?.length&&<p>Awaiting the first captured observations.</p>}
 <h3>Storage and alerts</h3><p>Allocated table/index bytes, not billing. Row counts are estimates. Missing cost data is not zero.</p>
 <details><summary>Inspect table allocation ({data.storage?.length??0} tables)</summary><div className={s.tableWrap}><table><thead><tr><th>Table</th><th>Allocated bytes</th><th>Index bytes</th><th>Measured</th></tr></thead><tbody>{data.storage?.map(r=><tr key={show(r.relation_name)}><td>{show(r.relation_name)}</td><td>{show(r.total_bytes)}</td><td>{show(r.index_bytes)}</td><td>{show(r.measured_at)}</td></tr>)}</tbody></table></div></details>
 {data.alerts?.map((r,i)=><p key={i}>{show(r.date)} · {show(r.metric)}: {show(r.detail)}</p>)}{!data.alerts?.length&&<p>No operational alerts recorded. This does not establish billing coverage.</p>}</>}
 </section>;
}
