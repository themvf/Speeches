"use client";
import {useEffect,useState} from 'react';
import {pct,share} from '@/lib/crypto-impact';
import {whyLeader,type Leader} from '@/lib/crypto-leaders';
import {PageHeader} from './crypto-shell';
import {paths} from '@/lib/crypto-workspace';
import styles from './crypto-research.module.css';
type AccountData={status:string;account?:{id:string;handle:string;name:string;followers:number|null};latest_profile?:{bio:string|null;followers:number|null;observed_at:string}|null;profile_history?:{followers:number|null;available:boolean;observed_at:string}[];activity?:{coin:string;posts:number;originals:number;first_at:string;last_at:string}[];posts?:{id:string;text:string;url:string;posted_at:string;kind:string;coins:string[]|null;return_24h:number|null}[];summary?:Leader|null};
const compact=(n:number|null|undefined)=>n==null?'unknown':Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n);
export function CryptoAccountView({accountId,onCoin,onMark}:{accountId:string;onCoin:(coin:string)=>void;onMark:(coin:string,id:string)=>void}){
 const [data,setData]=useState<AccountData|null>(null),[error,setError]=useState('');
 useEffect(()=>{const c=new AbortController();setData(null);setError('');fetch(`/api/market/crypto/account?id=${accountId}`,{signal:c.signal}).then(async r=>{const b=await r.json();if(!r.ok||!b.ok)throw Error();setData(b.data);}).catch(()=>{if(!c.signal.aborted)setError('This account could not be loaded.');});return()=>c.abort();},[accountId]);
 const s=data?.summary??null;const history=(data?.profile_history??[]).filter(h=>h.available&&h.followers!=null).slice().reverse();
 const growth=history.length>=2?Number(history[history.length-1].followers)-Number(history[0].followers):null;
 const firstCoin=data?.activity?.[0]?.coin;
 return <div style={{display:'flex',flexDirection:'column',gap:16}}>
  <PageHeader eyebrow="People" title={data?.account?.handle?'@'+data.account.handle:'Account'} crumbs={[{href:paths.people,label:'People'},{label:data?.account?.handle?'@'+data.account.handle:'Account'}]}>Profile, roles, price-linked history and posts for one account across every tracked coin.</PageHeader>
  {error?<p role="alert" className={styles.empty}>{error}</p>:!data?<p role="status" className={styles.muted}>Loading account evidence…</p>:data.status!=='ready'?<p className={styles.empty}>No saved evidence for this account.</p>:
  <div className={styles.wsGrid5}>
   <div style={{display:'flex',flexDirection:'column',gap:16}}>
    <section className={styles.wsSection}><h3>Profile</h3>
     <p style={{margin:0,fontSize:13,lineHeight:1.6,whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{data.latest_profile?.bio??'No bio saved for this account yet.'}</p>
     <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
      <div className={styles.wsStat}><span>Followers</span><strong>{compact(data.latest_profile?.followers??data.account?.followers)}</strong><small>{growth==null?'no comparable snapshots yet':`${growth>0?'+':''}${growth.toLocaleString()} across ${history.length} daily snapshots`}</small></div>
      <div className={styles.wsStat}><span>Early on</span><strong className={styles.wsPink}>{s?.early_coins?`${s.early_coins} coin${s.early_coins>1?'s':''}`:'—'}</strong><small>{s?.roles.filter(r=>r.early).map(r=>r.coin).join(', ')||'no early role yet'}</small></div>
      <div className={styles.wsStat}><span>Episodes</span><strong>{s?.episodes??0}</strong><small>across {s?.impact.length??0} coins</small></div>
      <div className={styles.wsStat}><span>+24h vs drift</span><strong className={s?.median_excess_24h==null?'':s.median_excess_24h>0?styles.positive:styles.negative}>{pct(s?.median_excess_24h)}</strong><small>{s?.impact.length?`up ${share(s.impact.reduce((a,i)=>a+(i.share_up_24h??0)*i.episodes,0)/Math.max(1,s.episodes))} of episodes`:'no complete window yet'}</small></div>
     </div>
     {s&&<p className={styles.insight}>{whyLeader(s)}. Roles are text heuristics; price figures are the pinned pool&#39;s move after posts, not attribution.</p>}
     <div style={{display:'flex',gap:8,flexWrap:'wrap'}}><a className={styles.control} href={`https://x.com/i/user/${accountId}`} target="_blank" rel="noreferrer">Open X profile ↗</a>{firstCoin&&<button className={styles.control} onClick={()=>onMark(firstCoin,accountId)}>Mark on {firstCoin} chart</button>}</div>
    </section>
    <section className={styles.wsSection}><h3>Follower history · daily snapshots</h3>{history.length<2?<p className={styles.muted}>Growth requires snapshots on comparable dates; nothing is backdated.</p>:<Spark points={history.map(h=>Number(h.followers))} first={history[0].observed_at.slice(0,10)} last={history[history.length-1].observed_at.slice(0,10)}/>}</section>
   </div>
   <div style={{display:'flex',flexDirection:'column',gap:16}}>
    <section className={styles.wsSection}><h3>Across tracked coins</h3>{!data.activity?.length?<p className={styles.muted}>No saved non-repost posts on any tracked coin.</p>:<div className={styles.table}><table><thead><tr><th>Coin</th><th>Posts</th><th>First · last</th><th>Role</th><th>Episodes</th><th>+24h vs drift</th><th></th></tr></thead><tbody>{data.activity.map(a=>{const role=s?.roles.find(r=>r.coin===a.coin),imp=s?.impact.find(i=>i.coin===a.coin);return <tr key={a.coin}><td><strong>{a.coin}</strong></td><td>{a.posts}<small className={styles.cellNote}>{a.originals} original</small></td><td>{String(a.first_at).slice(0,10)}<small className={styles.cellNote}>{String(a.last_at).slice(0,10)}</small></td><td><small>{role?role.role+(role.early?' · early':''):'—'}</small></td><td>{imp?.episodes??0}</td><td className={imp?.median_excess_24h==null?'':Number(imp.median_excess_24h)>0?styles.positive:styles.negative} style={{fontWeight:600}}>{pct(imp?.median_excess_24h)}<small className={styles.cellNote}>{imp?`up ${share(imp.share_up_24h)}`:'no window'}</small></td><td><button className={styles.link} onClick={()=>onMark(a.coin,accountId)}>Timeline →</button></td></tr>;})}</tbody></table></div>}{s?.watcher&&<small className={styles.muted}>Watcher ranking: {s.watcher}.</small>}</section>
    <section className={styles.wsSection}><h3>Latest posts · with what the pool did after</h3>{!data.posts?.length?<p className={styles.muted}>No saved posts.</p>:<div style={{display:'flex',flexDirection:'column',gap:10}}>{data.posts.map(p=><article key={p.id} className={styles.wsPostCard}><div style={{display:'flex',justifyContent:'space-between',gap:12,flexWrap:'wrap'}}><small className={styles.muted}>{String(p.posted_at).slice(0,16).replace('T',' ')} UTC · {(p.coins??[]).map(c=><button key={c} className={styles.link} style={{minHeight:0,marginRight:6}} onClick={()=>onCoin(c)}>{c}</button>)} · {p.kind}</small>{p.return_24h!=null&&<span className={Number(p.return_24h)>0?styles.positive:styles.negative} style={{fontSize:12,fontWeight:600}}>pool {pct(p.return_24h)} after 24h</span>}</div><p style={{margin:0,fontSize:13,lineHeight:1.55,whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{p.text}</p><a href={p.url} target="_blank" rel="noreferrer" style={{fontSize:12}}>Open source ↗</a></article>)}</div>}</section>
   </div>
  </div>}
 </div>;
}
function Spark({points,first,last}:{points:number[];first:string;last:string}){
 const min=Math.min(...points),max=Math.max(...points);const y=(v:number)=>max===min?45:80-(v-min)/(max-min)*70;
 const d=points.map((v,i)=>`${i?'L':'M'}${10+i*(420/Math.max(1,points.length-1))},${y(v)}`).join(' ');
 return <svg viewBox="0 0 440 100" style={{width:'100%',height:'auto'}} role="img" aria-label={`Follower count from ${points[0].toLocaleString()} to ${points[points.length-1].toLocaleString()}`}><path d={d} fill="none" stroke="#7dd3fc" strokeWidth="2.5"/><text x="10" y="96" fill="var(--ink-faint)" fontSize="10">{first} · {points[0].toLocaleString()}</text><text x="430" y="96" textAnchor="end" fill="var(--ink-faint)" fontSize="10">{last} · {points[points.length-1].toLocaleString()}</text></svg>;
}
