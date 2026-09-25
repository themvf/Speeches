// Time-of-day profile for the tracked coins, recomputed weekly by crypto-hour-profile.yml.
// Association only: this says when trading happens, never what a price will do next.
export type HourRow={utc_hour:number;et_hour:number;n:number;mean:number|null;median:number|null;t:number|null};
export type Pooled={coins:number;units:number;max_abs_t?:number;table:HourRow[];
 p_global?:{day:number;coin:number;market:number};split_half?:{r:number;p:number;coins:number}|null;test?:string};
export type HourProfile={window_days:number;since:string;until:string;et_offset_hours:number;
 pooled:Partial<Record<'returns'|'volume_share'|'volatility',Pooled>>;
 volume_excluded:{coin:string;kind:string;reason:string}[];verdict:string[]};

export const EVEN_SHARE=1/24;

/** What a metric may be called, given BOTH tests. Neither alone is the answer.
 *  The global test asks whether this window's profile is flat; the split-half asks whether the
 *  same shape comes back. A shape that recurs is the finding even when a dozen coins cannot push
 *  the strictest null under 0.05 — and a null that clears without recurring is not yet a finding. */
export type Standing='recurring'|'this window only'|'recurs, unproven'|'none'|'not tested';
export function standing(p:Pooled|undefined):Standing{
 if(!p||!p.p_global)return 'not tested';
 const clears=p.p_global.coin<0.05,holds=!!p.split_half&&p.split_half.p<0.05;
 return clears&&holds?'recurring':clears?'this window only':holds?'recurs, unproven':'none';
}

/** Hours ordered for reading as a day in the viewer's terms, Eastern midnight first. */
export function byEasternHour(table:HourRow[]):HourRow[]{
 return [...table].sort((a,b)=>a.et_hour-b.et_hour);
}

/** An hour's activity as a multiple of an even share. Null when the hour has no observations. */
export function relativeActivity(row:HourRow):number|null{
 return row.mean==null?null:row.mean/EVEN_SHARE;
}

/** The quietest and busiest stretches, as contiguous runs away from an even share.
 *  Returns Eastern hours, because that is how the reader thinks about their own day. */
export function extremes(table:HourRow[],quiet=0.8,busy=1.2):{quiet:number[];busy:number[]}{
 const rows=byEasternHour(table);
 const pick=(test:(x:number)=>boolean)=>rows.filter(r=>{const v=relativeActivity(r);return v!=null&&test(v);}).map(r=>r.et_hour);
 return {quiet:pick(v=>v<quiet),busy:pick(v=>v>busy)};
}

/** Contiguous runs of hours, wrapping midnight, rendered as "2:00–7:00" style labels. */
export function runs(hours:number[]):string[]{
 if(!hours.length)return [];
 const set=new Set(hours),out:string[]=[];
 for(const h of [...set].sort((a,b)=>a-b)){
  if(set.has((h+23)%24))continue;              // not the start of a run
  let end=h,guard=0;
  while(set.has((end+1)%24)&&guard++<24)end=(end+1)%24;
  out.push(`${h}:00–${(end+1)%24}:00`);
 }
 return out.length?out:[...set].sort((a,b)=>a-b).map(h=>`${h}:00–${(h+1)%24}:00`);
}
