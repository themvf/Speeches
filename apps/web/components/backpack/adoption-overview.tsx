'use client';
import {useState} from 'react';
import {LineChart,Line,ResponsiveContainer,XAxis,YAxis,Tooltip,CartesianGrid} from 'recharts';
import {numeric,point,dayOffset,type MonitorData,type Row} from '@/lib/backpack';
import s from './monitor.module.css';

const count=(v:unknown)=>numeric(v)===null?'N/A':new Intl.NumberFormat('en-US').format(Number(v));
const pct=(v:unknown)=>numeric(v)===null?'N/A':`${Number(v).toFixed(2)}%`;

export function GrowthOverview({data}:{data:MonitorData}){
 const [period,setPeriod]=useState(7),day=data.asOf??'';
 const current=point(data.adoption??[],day),row=data.adoptionGrowth?.find(r=>Number(r.period_days)===period);
 const settings=(row?.settings??{}) as Row;
 const stale=!!day&&Date.now()-Date.parse(`${day}T00:00:00Z`)>48*3600000;
 const observed=numeric(row?.observed_days??(current?1:0))??0,required=period+1;
 const byDay=new Map((data.adoption??[]).map(r=>[String(r.date),r]));
 const history=day?Array.from({length:period+1},(_,i)=>{const date=dayOffset(day,i-period),r=byDay.get(date);return {date,holders:numeric(r?.holders),whole:numeric(r?.whole_token_holders),multi:numeric(r?.multi_asset_holders)};}):[];
 const assets=Object.values((current?.assets??{}) as Row) as Row[];
 const issued=assets.filter(a=>(numeric(a.supply)??0)>0).length;
 const ownerCount=numeric(current?.holders),wholeCount=numeric(current?.whole_token_holders);
 const wholeShare=ownerCount!==null&&ownerCount>0&&wholeCount!==null?100*wholeCount/ownerCount:null;
 const cohort=((current?.cohorts??{}) as Row)[String(period)] as Row|undefined;
 const retained=row?.retained_holders??cohort?.retained_holders;
 const entered=row?.entered_holders??cohort?.entered_holders;
 const departed=row?.departed_holders??cohort?.departed_holders;
 const retention=row?.retention_pct??cohort?.retention_pct;
 const state=stale?'Stale evidence':String(row?.state??'Insufficient evidence');
 return <section className={s.adoptionCore} aria-label="Backpack securities adoption analytics">
  <div className={s.adoptionHero}>
   <div>
    <div className={s.eyebrow}>OBSERVABLE BACKPACK SECURITIES ADOPTION</div>
    <h2 className={s.growthHeadline}>{state}</h2>
    <p className={s.heroReason}>{stale?'The latest capture is over 48 hours old.':String(row?.reason??'A complete baseline exists, but growth needs consecutive daily observations.')}</p>
   </div>
   <div className={s.windowPicker}><span>Compare</span><div className={s.ranges} aria-label="Growth assessment window">{[7,30,90].map(p=><button key={p} aria-pressed={p===period} onClick={()=>setPeriod(p)}>{p}D</button>)}</div></div>
  </div>

  <div className={s.scopeNote}><b>What this measures:</b> deduplicated on-chain wallets holding the registered Backpack securities. Wallets are not people, exchange customers, deposits or company revenue. Stock prices and BP do not affect this headline.</div>

  <div className={s.coreMetrics}>
   <article className={s.primaryMetric}><span>Observable owners</span><strong>{count(current?.holders)}</strong><p>Nonzero, deduplicated wallets after verified system exclusions.</p><small>{pct(row?.holder_growth_pct)} over {period} days</small></article>
   <article><span>Whole-token sensitivity</span><strong>{count(current?.whole_token_holders)}</strong><p>{pct(wholeShare)} of observable owners hold at least one whole security token.</p><small>Dust check—not a person or dollar threshold.</small></article>
   <article><span>Multi-security owners</span><strong>{count(current?.multi_asset_holders)}</strong><p>Wallets holding positive balances in at least two registered securities.</p><small>Previous endpoint: {count(row?.previous_multi_asset_holders)}</small></article>
   <article><span>Positive-supply securities</span><strong>{issued} <em>/ {assets.length||'N/A'}</em></strong><p>Registered securities with positive captured supply. This is not the product count.</p><small>Median supply change: {pct(row?.median_supply_growth_pct)}</small></article>
  </div>

  <div className={s.progressPanel}>
   <div><b>{count(observed)} of {required} observations</b><span> needed for the {period}-day direction</span></div>
   <div className={s.progressTrack} aria-label={`${observed} of ${required} observations`}><i style={{width:`${Math.min(100,100*observed/required)}%`}}/></div>
   <small>Momentum needs {period*2+1} observations. Missing days, changed assets, or changed system exclusions pause comparability.</small>
  </div>

  <div className={s.cohortSection}>
   <div className={s.sectionHead}><div><h3>Wallet movement across the {period}-day window</h3><p>Endpoint overlap across the whole registered-security ecosystem—not sums of per-asset changes.</p></div></div>
   <div className={s.cohortGrid}>
    <article><span>Retained</span><strong>{count(retained)}</strong><small>Present at both endpoints</small></article>
    <article><span>Entered</span><strong>{count(entered)}</strong><small>Present now, absent at the earlier endpoint</small></article>
    <article><span>Departed</span><strong>{count(departed)}</strong><small>Present earlier, absent now</small></article>
    <article><span>Endpoint retention</span><strong>{pct(retention)}</strong><small>Retained ÷ earlier endpoint owners</small></article>
   </div>
   <p className={s.cohortCaveat}>Entered and departed describe observed wallet endpoints. They do not prove a person joined or left Backpack, and returning wallets are not identified as new people.</p>
  </div>

  <div className={s.panel}><div className={s.panelTitle}><h3>Ownership over time</h3><span>Complete daily observations only</span></div>{history.some(r=>r.holders!==null)?<div className={s.chart}><ResponsiveContainer width="100%" height="100%"><LineChart data={history}><CartesianGrid vertical={false}/><XAxis dataKey="date" tickFormatter={d=>String(d).slice(5)}/><YAxis/><Tooltip/><Line name="Nonzero owners" dataKey="holders" stroke="#146a5d" strokeWidth={2.5} connectNulls={false} dot/><Line name="At least one whole token" dataKey="whole" stroke="#466d8a" strokeWidth={2} connectNulls={false} dot/><Line name="Multiple securities" dataKey="multi" stroke="#a06c22" strokeWidth={2} connectNulls={false} dot/></LineChart></ResponsiveContainer></div>:<p>No complete ownership baseline yet.</p>}<p className={s.chartLegend}>Nonzero owners · at least one whole token · multiple securities. A single baseline cannot establish growth.</p></div>

  <details className={s.methodology}><summary>Classification rules and evidence safeguards</summary>
   <p>Growing slowly requires 30-day-equivalent ownership growth above {pct(settings.holder_threshold??1)} and median supply growth above {pct(settings.supply_threshold??0.1)}, confirmed across at least {pct(settings.breadth_threshold??60)} of issued securities. Growing rapidly also requires ownership growth of at least {pct(settings.rapid_threshold??10)}.</p>
   <p>Declining requires broad ownership and supply contraction. Status quo keeps both rates inside their noise bands. Conflicting signals remain Mixed. Growth confined to tiny balances is flagged when whole-token ownership is flat or falling.</p>
   <p>Sources: finalized Solana supply, reconciled Helius ownership, approved registry identities, and evidence-backed wallet labels. Methodology: {String(row?.methodology??'adoption-v1')}.</p>
  </details>
 </section>;
}
