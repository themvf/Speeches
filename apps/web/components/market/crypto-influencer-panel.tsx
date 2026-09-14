"use client";
import { useState } from 'react';
import { formatCount, growthLabel, rankAccounts, type Leaderboard, type Tracking } from '@/lib/crypto-social';
import styles from './crypto-research.module.css';

const boards: {id:Leaderboard;label:string;description:string}[] = [
 {id:'attention',label:'Most discussed',description:'Ranked by distinct people interacting with each account in saved coin posts over 28 days. Attention can include criticism.'},
 {id:'growth',label:'Fastest growing',description:'Ranked by net followers gained over seven calendar days. Percentage growth adds context for smaller accounts.'},
 {id:'emerging',label:'Emerging voices',description:'Follower growth plus more people interacting this week. Early signal only: collection coverage can differ between weeks.'},
 {id:'posting',label:'Most active',description:'Ranked by observed coin-related posts over 28 days. Sample counts are not total posting volume.'},
 {id:'connections',label:'Most connected',description:'Ranked by distinct incoming and outgoing interaction partners in the saved coin graph.'},
 {id:'bios',label:'Coin in bio',description:'Accounts whose latest retrieved bio mentions the coin. A match does not verify affiliation or holdings.'},
];
export function CryptoInfluencerPanel({data}:{data:Tracking}) {
 const [board,setBoard]=useState<Leaderboard>('attention');
 const [selected,setSelected]=useState<string|null>(null);
 const [query,setQuery]=useState('');
 const [limit,setLimit]=useState(10);
 const bioIds=new Set(data.bioMatches.filter(m=>m.field==='bio').map(m=>m.account_id));
 const ranked=rankAccounts(data.accounts,board,bioIds).filter(a=>`${a.handle} ${a.name}`.toLowerCase().includes(query.toLowerCase()));
 const account=data.accounts.find(a=>a.id===selected);
 const history=data.history.filter(p=>p.account_id===selected);
 const leader=rankAccounts(data.accounts,'attention').find(a=>a.participants>0);
 const grower=rankAccounts(data.accounts,'growth')[0];
 const metricLabel=board==='posting'?'Coin posts':board==='connections'?'Partners':board==='emerging'?'New participants¹':'People interacting';
 return <div>
  <div className={styles.cards}>
   <div className={styles.card}><div className={styles.label}>Most discussed · 28 days</div><div className={styles.value}>{leader?<button className={styles.link} onClick={()=>setSelected(leader.id)}>@{leader.handle}</button>:'Building the picture'}</div><p className={styles.muted}>{leader?`${formatCount(leader.participants)} distinct people interacted in saved posts.`:'No incoming interactions observed yet.'}</p></div>
   <div className={styles.card}><div className={styles.label}>Fastest growing · 7 days</div><div className={`${styles.value} ${styles.positive}`}>{grower?<button onClick={()=>{setBoard('growth');setSelected(grower.id);}}>{growthLabel(grower.growth_7d)} followers</button>:'Awaiting baseline'}</div><p className={styles.muted}>{grower?`@${grower.handle} · ${growthLabel(grower.growth_percent_7d,true)}`:'Growth appears once seven-day observations are available.'}</p></div>
   <div className={styles.card}><div className={styles.label}>Coin in bio</div><div className={styles.value}>{bioIds.size} accounts</div><p className={styles.muted}>{data.accounts.filter(a=>a.tracked).length} accounts tracked daily · {data.accounts.length} candidates found.</p></div>
  </div>
  <div className={styles.toolbar}><h3>People to watch</h3><div className="flex flex-wrap gap-2"><select aria-label="Rank people by" value={board} onChange={e=>{setBoard(e.target.value as Leaderboard);setLimit(10);}}>{boards.map(b=><option key={b.id} value={b.id}>{b.label}</option>)}</select><input type="search" aria-label="Find an account" placeholder="Find an account…" value={query} onChange={e=>{setQuery(e.target.value);setLimit(10);}}/></div></div>
  <p className={`${styles.muted} mb-3`}>{boards.find(b=>b.id===board)?.description}</p>
  {!ranked.length?<div className={styles.empty}>{query?'No accounts match your search.':board==='growth'||board==='emerging'?'No qualifying growth yet. We need a seven-day baseline before ranking rising accounts.':'No matching accounts observed yet.'}</div>:
  <div className={styles.table}><table><thead><tr><th className="text-left">Account</th><th>Followers</th><th>7-day growth</th>{board!=='growth'&&<th>{metricLabel}</th>}</tr></thead>
   <tbody>{ranked.slice(0,limit).map(a=><tr key={a.id}>
    <td><button onClick={()=>setSelected(a.id)} className={styles.link}>@{a.handle}</button><div className={styles.muted}>{a.tracked?'Tracking daily':'Discovered'} {bioIds.has(a.id)&&<span className={styles.badge}>Coin in bio</span>}</div></td>
    <td>{formatCount(a.followers)}</td><td><span className={a.growth_7d==null?'':Number(a.growth_7d)>0?styles.positive:Number(a.growth_7d)<0?styles.negative:''}>{growthLabel(a.growth_7d)}</span>{a.growth_percent_7d!=null&&<div className={styles.muted}>{growthLabel(a.growth_percent_7d,true)}</div>}</td>
    {board!=='growth'&&<td>{board==='posting'?a.posts:board==='connections'?a.connections:board==='emerging'?a.participants_7d-a.participants_previous_7d:a.participants}</td>}
   </tr>)}</tbody></table></div>}
  {board==='emerging'&&<p className={styles.muted}>¹ Change in distinct interacting accounts versus the prior week; not necessarily first-time participants.</p>}
  {ranked.length>limit&&<button onClick={()=>setLimit(limit+10)} className={`${styles.link} mt-3 text-xs`}>Show 10 more · {ranked.length} results</button>}
  {account&&<div className={styles.detail}>
   <div className="flex justify-between gap-3"><a href={`https://x.com/i/user/${account.id}`} target="_blank" rel="noreferrer" className={styles.link}>@{account.handle} ↗</a><button className="text-xs" onClick={()=>setSelected(null)}>Close details</button></div>
   <p className="my-3 text-sm whitespace-pre-wrap break-words">{account.bio??'Bio unavailable in latest observation.'}</p>
   <p className={styles.insight}>{account.participants>0?`${account.participants} distinct people interacted with this account; ${account.repeat_participants} returned on multiple days. ${account.amplifiers} quoted or reposted it.`:'No incoming interactions captured yet. This account remains a discovery candidate.'}</p>
   <div className="grid grid-cols-2 gap-4 my-4 text-sm">{[['Coin posts',account.posts],['Active days',account.active_days],['Typical engagement',account.median_engagement],['Growth acceleration / day',account.growth_acceleration]].map(([label,value])=><div key={String(label)}><span className={`block ${styles.muted}`}>{label}</span>{formatCount(value)}</div>)}</div>
   <p className={styles.muted}>Typical engagement is the median count of likes, replies, quotes and reposts. Post ages differ. {account.active_days<5||(account.engagement_samples??0)<10?'Limited evidence; treat this as an early signal.':''}</p>
   {account.evidence&&<a className={`${styles.link} block text-xs mt-3`} href={account.evidence} target="_blank" rel="noreferrer">See an interaction source ↗</a>}
   <details><summary>Follower history · {history.length} observations</summary><div className="max-h-48 overflow-auto"><table><thead><tr><th className="text-left">Observed (UTC)</th><th>Followers</th></tr></thead><tbody>{history.map(p=><tr key={p.observed_at}><td>{p.observed_at.slice(0,16).replace('T',' ')}</td><td>{p.available?formatCount(p.followers):'Unavailable'}</td></tr>)}</tbody></table></div></details>
   <details><summary>Evidence & measurement details</summary><p className={styles.muted}>Discovered: {account.reason}. Role: {account.category}. Profile observed: {account.profile_observed_at?new Date(account.profile_observed_at).toLocaleString():'Pending'}. Growth baseline: {account.baseline_at?.slice(0,10)??'Pending'}.</p><p className={styles.muted}>{account.engagement_samples??0} engagement samples. 24–30h median: {formatCount(account.median_engagement_24_30h)} across {account.age_matched_samples??0} posts. Top five participants: {account.top5_attention_percent==null?'Unknown':`${formatCount(account.top5_attention_percent)}%`} of interaction days. Reciprocal partners: {account.reciprocal_participants}. Acceleration requires 14-day history.</p></details>
  </div>}
  <details><summary>Bio changes <span className={styles.badge}>{data.bioChanges.length}</span></summary><p className={styles.muted}>Dates show when we first observed each change.</p>{!data.bioChanges.length&&<p className={styles.muted}>No bio changes observed yet.</p>}{data.bioChanges.map(b=><div key={`${b.account_id}-${b.observed_at}`} className="my-4 text-sm"><strong>@{b.handle}</strong><span className={`ml-2 ${styles.muted}`}>{b.observed_at.slice(0,10)}</span><p className={`${styles.muted} break-words`}>Before: {b.previous_bio||'(empty)'}</p><p className="break-words">After: {b.bio||'(empty)'}</p></div>)}</details>
  <details><summary>Account posting samples</summary><p className={styles.muted}>One-page samples include replies; they do not establish total posts per day.</p>{!data.coverage.length?<p className={styles.muted}>No account timeline samples collected yet.</p>:<div className={styles.table}><table><thead><tr><th className="text-left">Account</th><th>UTC day</th><th>Posts found</th><th>Coverage</th></tr></thead><tbody>{data.coverage.map(c=><tr key={`${c.account_id}-${c.end_at}`}><td>@{c.handle}</td><td>{c.start_at.slice(0,10)}</td><td>{c.in_window_posts}</td><td>{c.status==='capped'?'Limited sample':'Search boundary reached'}</td></tr>)}</tbody></table></div>}</details>
 </div>;
}
