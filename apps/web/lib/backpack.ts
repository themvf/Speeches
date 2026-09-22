/** Shared research methodology. Database numerics arrive as decimal strings. */
export const BP_MINT = 'BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy';
export type Row = Record<string, unknown>;
export type Quality = {metric:string;status:string;source:string;calculation:string;limitation:string;observed_at:string|null};
export type MonitorData = {
 status:string; asOf:string|null; assets:Row[]; history:Row[]; ecosystem:Row[];
 bp:Row[]; quality:Quality[]; quotes:Row[]; holders:Row[]; dex:Row[]; runs:Row[]; usage:Row[];whales?:Row[]; analytics?:Row[];
};
export function numeric(v:unknown):number|null {
 if(v===null||v===undefined||v===''||typeof v==='boolean')return null;
 const n=Number(v);return Number.isFinite(n)?n:null;
}
export function change(now:unknown,before:unknown):number|null {
 const a=numeric(now),b=numeric(before);return a!==null&&b!==null&&b>0?(a/b-1)*100:null;
}
export function dayOffset(day:string,n:number):string {
 if(!/^\d{4}-\d{2}-\d{2}$/.test(day.slice(0,10)))return '';
 const d=new Date(`${day.slice(0,10)}T00:00:00Z`);d.setUTCDate(d.getUTCDate()+n);return d.toISOString().slice(0,10);
}
export function point(rows:Row[],day:string):Row|undefined{return rows.find(r=>String(r.date).slice(0,10)===day.slice(0,10));}
export function windowChange(rows:Row[],day:string,field:string,days:number):number|null {
 const a=point(rows,day),b=point(rows,dayOffset(day,-days));
 // Adding/removing covered securities must not silently look like same-universe growth.
 if(!a||!b||(a.assets_expected!==undefined&&a.assets_expected!==b.assets_expected))return null;
 return change(a[field],b[field]);
}
export function rollingIssuance(rows:Row[],day:string,days:number):number|null {
 let total=0;
 for(let i=0;i<days;i++){
  const v=numeric(point(rows,dayOffset(day,-i))?.net_supply_change_usd);
  if(v===null)return null;
  total+=v;
 }
 return total;
}
export function qualityFor(data:MonitorData,field:string,assetId?:string):Quality {
 const rows=data.quality.filter(q=>q.metric===(field==='meaningful_holders'?'holders_over_100':field)&&(!assetId||String((q as unknown as Row).asset_id)===assetId));
 if(rows.length===1)return rows[0];
 return {metric:field,status:rows.some(q=>q.status==='Unavailable')?'Partial':rows.length?'Estimated':'Unavailable',
  source:rows.length?[...new Set(rows.map(q=>q.source))].join('; '):'No observation',
  calculation:rows.length?[...new Set(rows.map(q=>q.calculation))].join('; '):'Not yet captured',
  limitation:rows.length?[...new Set(rows.map(q=>q.limitation))].join('; '):'Unknown values are not zero.',observed_at:data.asOf};
}
export function mechanicalChanges(rows:Row[],day:string):string[] {
 const a=point(rows,day),b=point(rows,dayOffset(day,-1));
 if(!a||!b)return ['Two consecutive daily observations are needed to identify changes.'];
 const facts:string[]=[];
 for(const [field,label] of [['reference_aum_usd','Reference AUM'],['meaningful_holders','Meaningful wallets']] as const){
  const delta=windowChange(rows,day,field,1);
  if(delta!==null&&delta!==0)facts.push(`${label} ${delta>0?'increased':'decreased'} ${Math.abs(delta).toFixed(2)}% since yesterday.`);
 }
 const issuance=rollingIssuance(rows,day,30),before=rollingIssuance(rows,dayOffset(day,-1),30);
 if(issuance!==null&&before!==null&&issuance!==before)facts.push(`30-day net on-chain issuance is $${issuance.toLocaleString('en-US',{maximumFractionDigits:0})}, versus $${before.toLocaleString('en-US',{maximumFractionDigits:0})} yesterday.`);
 return facts.length?facts:['No change in comparable, available adoption metrics since yesterday.'];
}
export function sortRows(rows:Row[],field:string,descending:boolean):Row[]{
 return [...rows].sort((a,b)=>{
  const av=a[field],bv=b[field];
  if(av==null)return bv==null?0:1;if(bv==null)return -1;
  const an=numeric(av),bn=numeric(bv);
  const cmp=an!==null&&bn!==null?an-bn:String(av).localeCompare(String(bv));
  return descending?-cmp:cmp;
 });
}
