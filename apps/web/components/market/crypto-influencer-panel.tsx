"use client";
import { useState } from 'react';
import { formatCount, growthLabel, rankAccounts, type Leaderboard, type Tracking } from '@/lib/crypto-social';

const boards: {id:Leaderboard;label:string;description:string}[] = [
 {id:'attention',label:'Attention',description:'Distinct accounts mentioning, replying to, quoting, or reposting each account in the last 28 days. Attention may include criticism.'},
 {id:'growth',label:'Follower growth',description:'Net followers gained across observed dates seven calendar days apart. Missing observations stay blank.'},
 {id:'emerging',label:'Emerging',description:'Positive follower growth and more distinct interacting accounts this week. Provisional: search coverage can differ between weeks.'},
 {id:'posting',label:'Posting',description:'Observed coin-related posts in the last 28 days. This is a sample, not total account activity.'},
 {id:'connections',label:'Connections',description:'Distinct incoming and outgoing interaction partners across the full saved coin graph. This measures breadth, not verified influence across communities.'},
 {id:'bios',label:'Bio matches',description:'Coin terms found in the latest retrieved bio. These are self-descriptions, not verified affiliations or holdings.'},
];
export function CryptoInfluencerPanel({data}:{data:Tracking}) {
 const [board,setBoard]=useState<Leaderboard>('attention');
 const [selected,setSelected]=useState<string|null>(null);
 const bioIds=new Set(data.bioMatches.filter(m=>m.field==='bio').map(m=>m.account_id));
 const ranked=rankAccounts(data.accounts,board,bioIds);
 const account=data.accounts.find(a=>a.id===selected);
 const history=data.history.filter(p=>p.account_id===selected);
 return <div className="space-y-3">
  <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold">People & growth</h3>
   <span className="text-xs text-[color:var(--ink-faint)]">{data.accounts.filter(a=>a.tracked).length} tracked · {data.accounts.length} candidates</span></div>
  <div className="flex flex-wrap gap-2" role="group" aria-label="Account rankings">{boards.map(b=><button key={b.id} aria-pressed={board===b.id} onClick={()=>setBoard(b.id)} className={`rounded-lg px-3 py-2 text-xs ${board===b.id?'bg-cyan-900 text-cyan-100':'bg-slate-800 text-slate-200'}`}>{b.label}</button>)}</div>
  <p className="text-xs text-[color:var(--ink-faint)]">{boards.find(b=>b.id===board)?.description}</p>
  {!ranked.length?<p className="py-3 text-sm">{board==='growth'||board==='emerging'?'No qualifying growth observations yet. Seven-day growth needs a baseline; acceleration needs 14 days.':'No matching accounts observed yet.'}</p>:
  <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr>{['Account','Followers','7d gain','7d %',board==='posting'?'Posts':board==='connections'?'Partners':'Participants', 'Bio'].map(h=><th key={h} className="p-2 whitespace-nowrap">{h}</th>)}</tr></thead>
   <tbody>{ranked.slice(0,40).map(a=><tr key={a.id} className="border-t border-[color:var(--line)]">
    <td className="p-2"><button onClick={()=>setSelected(a.id)} className="text-cyan-300 text-left">@{a.handle}</button><span className="block text-[10px] text-[color:var(--ink-faint)]">{a.tracked?'Tracking':'Candidate'} · {a.category}</span></td>
    <td>{formatCount(a.followers)}</td><td>{growthLabel(a.growth_7d)}</td><td>{growthLabel(a.growth_percent_7d,true)}</td>
    <td>{board==='posting'?a.posts:board==='connections'?a.connections:a.participants}</td><td>{bioIds.has(a.id)?'Match':'—'}</td>
   </tr>)}</tbody></table></div>}
  {ranked.length>40&&<p className="text-xs">Showing 40 of {ranked.length} candidates for this ranking.</p>}
  {account&&<div className="rounded-lg border border-[color:var(--line)] p-3 space-y-3">
   <div className="flex justify-between"><a href={`https://x.com/i/user/${account.id}`} target="_blank" rel="noreferrer" className="text-cyan-300">@{account.handle} ↗</a><button className="text-xs" onClick={()=>setSelected(null)}>Close</button></div>
   <p className="text-sm whitespace-pre-wrap break-words">{account.bio??'Bio unavailable in latest observation.'}</p>
   <p className="text-xs text-[color:var(--ink-faint)]">Discovered: {account.reason}. Profile observed: {account.profile_observed_at?new Date(account.profile_observed_at).toLocaleString():'Awaiting profile collection'}.</p>
   <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
    {[['Coin posts',account.posts],['Active days',account.active_days],['Distinct amplifiers',account.amplifiers],['Repeat participants',account.repeat_participants],['Median engagement',account.median_engagement],['24–30h median',account.median_engagement_24_30h],['Age-matched posts',account.age_matched_samples],['Growth acceleration/day',account.growth_acceleration]].map(([label,value])=><div key={String(label)}><span className="block text-[color:var(--ink-faint)]">{label}</span>{formatCount(value)}</div>)}
   </div>
   <p className="text-xs text-[color:var(--ink-faint)]">{account.engagement_samples??0} engagement samples; {account.active_days<5||(account.engagement_samples??0)<10?'provisional, limited evidence.':'observed sample.'} General engagement counts have different post ages. Top five participants: {formatCount(account.top5_attention_percent)}% of interaction days; reciprocal partners: {account.reciprocal_participants}.</p>
   {account.evidence&&<a className="text-xs underline" href={account.evidence} target="_blank" rel="noreferrer">Open an interaction source post</a>}
   <details><summary className="cursor-pointer text-xs">Follower observations ({history.length})</summary><div className="max-h-48 overflow-y-auto"><table className="w-full text-left text-xs"><thead><tr><th>Observed (UTC)</th><th>Followers</th></tr></thead><tbody>{history.map(p=><tr key={p.observed_at}><td>{p.observed_at.slice(0,16).replace('T',' ')}</td><td>{p.available?formatCount(p.followers):'Unavailable'}</td></tr>)}</tbody></table></div></details>
  </div>}
  <details><summary className="cursor-pointer text-xs">Bio changes ({data.bioChanges.length})</summary>
   <p className="my-2 text-xs text-[color:var(--ink-faint)]">Dates show when a change was first observed, not when the bio was edited.</p>
   {data.bioChanges.map(b=><div key={`${b.account_id}-${b.observed_at}`} className="my-3 text-xs"><strong>@{b.handle} · {b.observed_at.slice(0,10)}</strong><p className="break-words">Before: {b.previous_bio||'(empty)'}</p><p className="break-words">After: {b.bio||'(empty)'}</p></div>)}
  </details>
  <details><summary className="cursor-pointer text-xs">Account posting samples</summary><p className="my-2 text-xs text-[color:var(--ink-faint)]">One-page timeline samples; replies included. Capped samples cannot establish total posts per day.</p>
   {data.coverage.map(c=><p key={`${c.account_id}-${c.end_at}`} className="text-xs">@{c.handle} · {c.start_at.slice(0,10)} · {c.in_window_posts} observed posts · {c.status.replaceAll('_',' ')}</p>)}
  </details>
  <details><summary className="cursor-pointer text-xs">Collection details</summary>
   <p className="my-2 text-xs">{data.campaign?`Pilot ends ${data.campaign.end_at.slice(0,10)}. Daily collection, up to 40 unique profiles. One extra activity page per day, rotating searches and samples.`:'Tracking starts with the first collection run.'}</p>
   <p className="text-xs">Bio scanning is included with profile collection. Paid keyword user search awaits a verified provider page-size limit.</p>
   <ul className="my-2 text-xs">{data.ledger.map(l=><li key={l.allocation}>{l.allocation.replaceAll('_',' ')}: {formatCount(l.reserved_credits)} reserved · {formatCount(l.estimated_credits)} estimated credits</li>)}</ul>
  </details>
 </div>;
}
