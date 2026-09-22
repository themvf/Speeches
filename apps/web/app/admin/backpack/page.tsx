'use client';
import Link from 'next/link';
import {useState} from 'react';
import s from '@/components/backpack/monitor.module.css';
export default function RegistryAdmin(){
 const [message,setMessage]=useState(''),[busy,setBusy]=useState(false);
 async function save(e:React.FormEvent<HTMLFormElement>){
  e.preventDefault();setBusy(true);setMessage('Saving…');
  const form=e.currentTarget,body=Object.fromEntries(new FormData(form).entries());
  try{const r=await fetch('/api/admin/backpack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,action:'add_asset',approved:body.approved==='on'})});const v=await r.json();setMessage(r.ok?'Approved registry entry saved. Use Refresh Now on the monitor to capture it.':v.error);if(r.ok)form.reset();}catch{setMessage('Registry could not be saved.');}finally{setBusy(false);}
 }
 return <main className={s.form}><Link href="/market/crypto/backpack">← Backpack monitor</Link><h1>Backpack security registry</h1><p>Add a mint only after verifying the exact address and issuer. A matching name, exchange listing or ticker is insufficient. Entries created here carry manual approval; they are not automatically labeled independently verified.</p><form onSubmit={save}><div className={s.formGrid}>{[['token_symbol','Token symbol'],['token_name','Token name'],['solana_mint','Solana mint'],['issuer','Issuer'],['underlying_symbol','Underlying ticker'],['underlying_exchange','Primary exchange'],['underlying_name','Underlying name'],['official_source','Official announcement / evidence URL']].map(([key,label])=><label key={key}>{label}<input name={key} required maxLength={2000} type={key==='official_source'?'url':'text'}/></label>)}<label>Security type<select name="asset_type"><option value="common_stock">Common stock</option><option value="etf">ETF</option><option value="other_security">Other security</option></select></label><label>Launch date<input name="launch_date" type="date"/></label></div><label>Approval evidence and one-token / one-share verification<textarea name="approval_notes" required maxLength={2000}/></label><label className={s.check}><input name="approved" type="checkbox" required/>I verified this exact mint, the issuer, and that one token represents one underlying share. I approve its inclusion.</label><button disabled={busy}>{busy?'Saving…':'Add approved security'}</button><p role="status">{message}</p></form></main>;
}
