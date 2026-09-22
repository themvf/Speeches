'use client';
import {useState} from 'react';
import {numeric,point,dayOffset,type MonitorData} from '@/lib/backpack';
import s from './monitor.module.css';
const money=(v:unknown)=>numeric(v)===null?'N/A':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',notation:'compact',maximumFractionDigits:2}).format(Number(v));
const pct=(v:unknown)=>numeric(v)===null?'N/A':`${Number(v).toFixed(2)}%`;
const count=(v:unknown)=>numeric(v)===null?'N/A':new Intl.NumberFormat('en-US').format(Number(v));
export function GrowthOverview({data}:{data:MonitorData}){
 const [period,setPeriod]=useState(30);
 const row=data.growth?.find(r=>Number(r.period_days)===period),day=data.asOf??'';
 const stale=!!day&&Date.now()-Date.parse(`${day}T00:00:00Z`)>48*3600000;
 const state=stale?'Stale evidence':String(row?.state??'Insufficient evidence');
 const enough=row&&row.state!=='Insufficient evidence';
 const bp=point(data.bp,day),oldBp=day?point(data.bp,dayOffset(day,-period)):undefined;
 const bpH=numeric(bp?.holders_over_100),oldH=numeric(oldBp?.holders_over_100),bpC=numeric(bp?.economic_top_20_holder_pct),oldC=numeric(oldBp?.economic_top_20_holder_pct);
 const bpState=stale?'Stale evidence':bpH===null||oldH===null||bpC===null||oldC===null?'Insufficient evidence':bpH>oldH&&bpC<oldC?'Broadening':bpH<oldH&&bpC>oldC?'Concentrating':'Mixed / little change';
 return <section className={s.growthOverview} aria-label="Backpack growth overview">
 <div className={s.sectionHead}><div><div className={s.eyebrow}>IS THE BACKPACK NETWORK GROWING?</div><h2 className={s.growthHeadline}>{state}</h2></div><div className={s.ranges} aria-label="Growth assessment window">{[7,30,90].map(p=><button key={p} aria-pressed={p===period} onClick={()=>setPeriod(p)}>{p} days</button>)}</div></div>
 <p>{stale?'The most recent capture is over 48 hours old. Historical findings below are not a current assessment.':String(row?.reason??'No trustworthy adoption window has been captured yet. Missing history is not evidence of decline.')}</p>
 <p className={s.muted}>Tracked securities · {period}-day assessment · {day||'No capture yet'} · BP price and trading volume do not determine this headline.</p>
 {enough?<ul className={s.growthReasons}>
 <li>Net on-chain issuance was <b>{money(row.net_issuance_usd)}</b>, or {pct(row.issuance_aum_pct)} of starting AUM. Reference AUM changed {pct(row.aum_growth_pct)}.</li>
 <li>Meaningful holders moved from <b>{count(row.previous_meaningful_holders)}</b> to <b>{count(row.meaningful_holders)}</b> ({pct(row.holder_growth_pct)}).</li>
 <li>Momentum: <b>{String(row.momentum)}</b>. Prior issuance: {money(row.previous_net_issuance_usd)} ({pct(row.previous_issuance_aum_pct)} of starting AUM); prior holder growth: {pct(row.previous_holder_growth_pct)}.</li>
 </ul>:<p>Direction needs {period+1} consecutive daily observations; momentum needs {period*2+1}. Asset coverage, holder completeness and price timestamps must pass checks. DEX coverage can remain incomplete.</p>}
 <div className={s.growthGrid}>
 <article><h3>Capital</h3><strong>{money(row?.net_issuance_usd)}</strong><p>Net issuance over {period} days. Supply change valued at daily reference prices; not verified customer deposits.</p></article>
 <article><h3>Adoption</h3><strong>{pct(row?.holder_growth_pct)}</strong><p>Meaningful-holder growth. Multi-asset adoption: {pct(row?.previous_multi_asset_adoption_pct)} → {pct(row?.multi_asset_adoption_pct)}.</p></article>
 <article><h3>Momentum</h3><strong>{String(row?.momentum??'Insufficient evidence')}</strong><p>Current issuance rate {pct(row?.issuance_aum_pct)} vs {pct(row?.previous_issuance_aum_pct)} previously; holder growth {pct(row?.holder_growth_pct)} vs {pct(row?.previous_holder_growth_pct)}.</p></article>
 <article><h3>Ecosystem depth</h3><strong>{count(row?.significant_securities)} securities</strong><p>At least {money(row?.significance_threshold_usd)} AUM each; previously {count(row?.previous_significant_securities)}. Top-five AUM share: {pct(row?.previous_top_5_aum_pct)} → {pct(row?.top_5_aum_pct)}.</p><small>Liquidity and DeFi depth remain separate evidence; AUM breadth does not establish either.</small></article>
 <article><h3>BP ownership · separate</h3><strong>{bpState}</strong><p>Meaningful holders: {count(oldBp?.holders_over_100)} → {count(bp?.holders_over_100)}. Top-20 economic concentration: {pct(oldBp?.economic_top_20_holder_pct)} → {pct(bp?.economic_top_20_holder_pct)}.</p><small>USD thresholds can change with BP price. Verified system exclusions can change with labels. This is not a price forecast.</small></article>
 </div>
 <details><summary>Why this assessment? Rules, sources and AUM decomposition</summary>
 <p>Direction uses issuance above ±{pct(row?.issuance_threshold_pct??0.1)} of starting AUM and holder growth above ±{pct(row?.holder_threshold_pct??1)}. Both positive: Growing; both negative: Declining; disagreement or small changes: Mixed / flat.</p>
 <p>Growing, but slowing requires both issuance rate and holder growth to weaken by more than {String(row?.slowdown_threshold_pp??0.25)} percentage points versus the prior equal-length window. Lack of a prior window does not imply steady momentum.</p>
 <p>AUM change attributable to supply at starting prices: {money(row?.supply_effect_usd)}. Price contribution at ending supply: {money(row?.price_effect_usd)}. These add to endpoint AUM change and differ from daily-priced net issuance.</p>
 <p>{String(row?.methodology??'growth-v1: uses persisted daily snapshots only. No LLM estimates, trading-volume proxy, or BP price input.')}</p>
 <p>Sources: finalized Solana supply, reconciled Helius owner enumeration, Alpaca reference prices, approved asset registry and captured wallet labels. Wallets are not identified individuals. Missing days, changing security coverage, incomplete holders or stale/future equity references suppress the network assessment. Open each metric’s source details below for the underlying timestamps and limitations.</p>
 </details>
 </section>;
}
