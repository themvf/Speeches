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
 const enough=row&&row.state!=='Insufficient evidence';
 const byDay=new Map((data.adoption??[]).map(r=>[String(r.date),r]));
 const history=day?Array.from({length:period+1},(_,i)=>{const date=dayOffset(day,i-period),r=byDay.get(date);return {date,holders:numeric(r?.holders),whole:numeric(r?.whole_token_holders),multi:numeric(r?.multi_asset_holders)};}):[];
 const bp=point(data.bp,day),oldBp=point(data.bp,dayOffset(day,-period));
 return <section className={s.growthOverview} aria-label="Backpack adoption growth">
 <div className={s.sectionHead}><div><div className={s.eyebrow}>IS BACKPACK SECURITIES ADOPTION GROWING?</div><h2 className={s.growthHeadline}>{stale?'Stale evidence':String(row?.state??'Insufficient evidence')}</h2></div><div className={s.ranges} aria-label="Growth assessment window">{[7,30,90].map(p=><button key={p} aria-pressed={p===period} onClick={()=>setPeriod(p)}>{p} days</button>)}</div></div>
 <p>{stale?'The latest capture is over 48 hours old. This is historical evidence.':String(row?.reason??'Awaiting complete stored ownership evidence. Missing observations are not evidence of decline.')}</p>
 <p className={s.muted}>{period}-day window · {day||'No capture yet'} · Tracked securities only. Stock prices, BP price and trading volume do not determine this headline. Exchange customers, deposits and revenue are outside this coverage.</p>
 {!enough&&<p><b>{count(row?.observed_days??(current?1:0))} of {period+1} daily observations</b> available for direction; momentum needs {period*2+1}. Securities, system exclusions and complete holder coverage must remain comparable. Today's baseline is shown below.</p>}
 <div className={s.growthGrid}>
 <article><h3>Ownership</h3><strong>{count(current?.holders)} wallets</strong><p>{pct(row?.holder_growth_pct)} over {period} days · previous {count(row?.previous_holders)}.</p><small>Deduplicated nonzero owners; verified system wallets excluded. Wallets are not people.</small></article>
 <article><h3>Token issuance</h3><strong>{pct(row?.median_supply_growth_pct)}</strong><p>Median per-issued-security supply change over {period} days. No stock prices or addition of unlike tokens.</p></article>
 <article><h3>Growth breadth</h3><strong>{pct(row?.growing_holder_breadth_pct)}</strong><p>Issued securities gaining holders. Expanding supply: {pct(row?.expanding_supply_pct)}. Contracting supply: {pct(row?.contracting_supply_pct)}.</p></article>
 <article><h3>Multi-asset ownership</h3><strong>{count(current?.multi_asset_holders)} wallets</strong><p>Holding at least two securities with positive balances. Previous: {count(row?.previous_multi_asset_holders)}.</p></article>
 <article><h3>Momentum</h3><strong>{String(row?.momentum??'Insufficient evidence')}</strong><p>Speed and acceleration are separate: rapid growth can be slowing. Compared with the preceding {period}-day window.</p></article>
 <article><h3>Dust sensitivity</h3><strong>{count(current?.whole_token_holders)} wallets</strong><p>Holding at least one whole token of any security; growth {pct(row?.whole_token_growth_pct)}. A sensitivity check, not economic value.</p></article>
 </div>
 <div className={s.panel}><h3>Ownership over time</h3>{history.some(r=>r.holders!==null)?<div className={s.chart}><ResponsiveContainer width="100%" height="100%"><LineChart data={history}><CartesianGrid vertical={false}/><XAxis dataKey="date" tickFormatter={d=>String(d).slice(5)}/><YAxis/><Tooltip/><Line name="Nonzero owners" dataKey="holders" stroke="#21776c" connectNulls={false} dot/><Line name="At least one whole token" dataKey="whole" stroke="#43657e" connectNulls={false} dot/><Line name="Multiple securities" dataKey="multi" stroke="#996d24" connectNulls={false} dot/></LineChart></ResponsiveContainer></div>:<p>No complete ownership baseline yet.</p>}<p>Nonzero owners · at least one whole token · multiple securities. A single baseline cannot establish growth.</p></div>
 <details><summary>Growth rules and evidence</summary>
 <p>Rates use a 30-day equivalent (observed percentage change × 30 / window days), not a forecast. Holder noise band: ±{pct(settings.holder_threshold??1)}; median supply noise band: ±{pct(settings.supply_threshold??0.1)}.</p>
 <p>Growing slowly requires both rates above those bands, with at least {pct(settings.breadth_threshold??60)} of issued securities gaining holders and expanding supply. Growing rapidly also requires holder growth of at least {pct(settings.rapid_threshold??10)}. These are configurable research settings, not empirically validated universal cutoffs.</p>
 <p>Declining requires both rates below the negative bands and broad contraction. Status quo means both aggregate rates are within the bands. Conflicting signals are Mixed. Growth limited to tiny balances is flagged Mixed when the one-token comparison is available and not growing.</p>
 <p>Momentum requires holder and supply rates to move together by more than {String(settings.momentum_threshold??0.25)} percentage points versus the previous window. Consistently unissued securities are excluded from breadth; first issuance needs a positive baseline. Gaps, changed assets or changed system exclusions suppress comparisons.</p>
 <p>Sources: finalized Solana supply, supply-reconciled Helius holder enumerations, approved registry and captured wallet labels. Unlabelled custody, wallet splitting and dust can distort adoption. Nonzero ownership is separate from the existing $100 meaningful-holder metric. Methodology: {String(row?.methodology??'adoption-v1')}.</p>
 </details>
 <p><b>BP ownership remains separate.</b> Meaningful BP holders: {count(oldBp?.holders_over_100)} → {count(bp?.holders_over_100)}; top-20 economic concentration: {pct(oldBp?.economic_top_20_holder_pct)} → {pct(bp?.economic_top_20_holder_pct)}. BP dollar thresholds can change with price.</p>
 </section>;
}
