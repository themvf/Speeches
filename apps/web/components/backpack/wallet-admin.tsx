'use client';
import {useEffect,useState} from 'react';
import {walletLabels,confidences} from '@/lib/backpack-admin';
import type {Row} from '@/lib/backpack';
import s from './monitor.module.css';
export function WalletAdmin(){
 const [labels,setLabels]=useState<Row[]>([]),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
 async function load(){try{const r=await fetch('/api/admin/backpack?labels=1',{cache:'no-store'});const data=await r.json();if(r.ok)setLabels(data.labels);else setMessage(data.error);}catch{setMessage('Could not load labels.');}}
 useEffect(()=>{void load();},[]);
 async function save(e:React.FormEvent<HTMLFormElement>){
  e.preventDefault();const form=e.currentTarget,body=Object.fromEntries(new FormData(form).entries());setBusy(true);
  try{const r=await fetch('/api/admin/backpack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,action:'label_wallet',approved:body.approved==='on'})});const data=await r.json();setMessage(data.message??data.error);if(r.ok){form.reset();await load();}}catch{setMessage('Could not save wallet evidence.');}finally{setBusy(false);}
 }
 return <section><h2>Wallet attribution</h2><p>Record reviewed evidence only. Confirmed/high-confidence system labels exclude wallets from future economic concentration and whale cohorts. Market makers remain included. Use Unknown to revoke attribution; every revision is retained.</p><form onSubmit={save}><div className={s.formGrid}>
 <label>Wallet address<input required name="wallet_address" minLength={32} maxLength={44}/></label>
 <label>Entity<input required name="entity" maxLength={2000}/></label>
 <label>Label<select name="label" defaultValue="Unknown">{walletLabels.map(v=><option key={v}>{v}</option>)}</select></label>
 <label>Confidence<select name="confidence" defaultValue="low">{confidences.map(v=><option key={v}>{v}</option>)}</select></label>
 <label>Evidence URL<input required name="source" type="url" maxLength={2000}/></label></div>
 <label>Evidence and attribution limitations<textarea required name="notes" maxLength={2000}/></label>
 <label className={s.check}><input type="checkbox" required name="approved"/>I reviewed evidence for this exact wallet; transaction patterns alone do not establish ownership.</label>
 <button disabled={busy}>{busy?'Saving…':'Save wallet evidence'}</button><p role="status">{message}</p></form>
 <div className={s.tableWrap}><table><thead><tr><th>Wallet</th><th>Label / entity</th><th>Confidence</th><th>Evidence</th><th>Reviewed</th></tr></thead><tbody>{labels.map(r=><tr key={String(r.wallet_address)}><td><code>{String(r.wallet_address)}</code></td><td>{String(r.label)} / {String(r.entity)}</td><td>{String(r.confidence)}</td><td><a href={String(r.source).startsWith('https://')?String(r.source):undefined} target="_blank" rel="noreferrer">Source</a><p>{String(r.notes)}</p></td><td>{String(r.verified_at)}</td></tr>)}</tbody></table></div></section>;
}
