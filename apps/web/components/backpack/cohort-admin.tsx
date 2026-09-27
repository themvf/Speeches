'use client';
import {useEffect,useState} from 'react';
import type {Row} from '@/lib/backpack';
import s from './monitor.module.css';
export function CohortAdmin(){
 const [versions,setVersions]=useState<Row[]>([]),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
 async function load(){try{const r=await fetch('/api/admin/backpack?cohorts=1',{cache:'no-store'});const data=await r.json();if(r.ok)setVersions(data.versions??[]);else setMessage(data.error);}catch{setMessage('Could not load cohort versions.');}}
 useEffect(()=>{void load();},[]);
 const original=versions.find(v=>v.kind==='original');
 async function approve(e:React.FormEvent<HTMLFormElement>){
  e.preventDefault();const form=e.currentTarget,body=Object.fromEntries(new FormData(form).entries());setBusy(true);
  try{const r=await fetch('/api/admin/backpack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'approve_original_cohort',version_id:Number(body.version_id),notes:body.notes,approved:body.approved==='on'})});const data=await r.json();setMessage(data.message??data.error);if(r.ok)await load();}catch{setMessage('Could not approve the cohort.');}finally{setBusy(false);}
 }
 return <section><h2>BP holder cohorts</h2><p>Current versions are created daily from a complete, supply-reconciled BP capture. The original cohort is one reviewed current version, frozen for longitudinal analysis; its wallets stay tracked after they sell BP. It can be approved once and never replaced.</p>
 {original?<p role="status"><b>Original cohort approved</b> from version {String(original.derived_from_version)} ({String(original.source_date)}, {String(original.size)} wallets) by {String(original.approved_by)}. {String(original.approval_notes??'')}</p>
 :versions.length?<form onSubmit={approve}><label>Current version to freeze<select name="version_id">{versions.filter(v=>v.kind==='current').map(v=><option key={String(v.version_id)} value={String(v.version_id)}>v{String(v.version_id)} · {String(v.source_date)} · {String(v.size)} wallets{v.bootstrap?' · bootstrap':''}</option>)}</select></label>
 <label>Review notes: exclusions checked, known system wallets labelled, anomalies<textarea required name="notes" maxLength={2000}/></label>
 <label className={s.check}><input type="checkbox" required name="approved"/>I reviewed this version&apos;s ranking, labels and exclusions. Unlabelled large wallets are not assumed to be exchanges.</label>
 <button disabled={busy}>{busy?'Approving…':'Approve as the original cohort'}</button></form>
 :<p>No cohort version yet. The daily capture creates one after a complete BP holder enumeration.</p>}
 <p role="status">{message}</p>
 <div className={s.tableWrap}><table><thead><tr><th>Version</th><th>Kind</th><th>Capture</th><th>Wallets</th><th>Entered / left / queued</th><th>Entrant cap bound</th><th>Status</th></tr></thead><tbody>{versions.map(v=><tr key={String(v.version_id)}><td>v{String(v.version_id)}</td><td>{String(v.kind)}</td><td>{String(v.source_date)}</td><td>{String(v.size)}</td><td>{String(v.entered)} / {String(v.left_count)} / {String(v.queued)}</td><td>{v.entrant_cap_bound?'Yes':'No'}</td><td>{String(v.status)}</td></tr>)}</tbody></table></div></section>;
}
