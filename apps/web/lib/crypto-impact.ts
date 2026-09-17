// Ranks accounts by what the pinned pool's price did after their posts. Association only: no causal claim.
export const IMPACT_VERSION='price-events-v1';
export type ImpactRow={account_id:string;handle:string;followers:number|null;coins:string[];episodes:number;posts:number;
 median_1h:number|null;median_6h:number|null;median_24h:number|null;median_excess_24h:number|null;
 share_up_24h:number|null;share_beat_24h:number|null;median_volume_ratio:number|null;first_at:string;last_at:string;
 best_post_id:string|null;best_post_url:string|null;best_return_24h:number|null;worst_return_24h:number|null};
export type ImpactBaseline={coin:string;median_24h:number|null;share_up_24h:number|null;hours:number;source_id:string|null;first_hour:string|null;last_hour:string|null};
export const MIN_EPISODES=3;
export function rankImpact(rows:ImpactRow[],minEpisodes=MIN_EPISODES,coin='ALL'){
 return rows.filter(r=>r.episodes>=minEpisodes&&(coin==='ALL'||r.coins.includes(coin)))
  .sort((a,b)=>(b.median_excess_24h??-Infinity)-(a.median_excess_24h??-Infinity)||b.episodes-a.episodes||a.account_id.localeCompare(b.account_id));
}
export function confidence(r:ImpactRow){
 if(r.episodes>=10&&r.coins.length>=2)return 'Repeated across coins';
 if(r.episodes>=10)return 'Repeated evidence';
 if(r.episodes>=MIN_EPISODES)return 'Limited sample';
 return 'Single observations';
}
export function pct(value:number|null|undefined,digits=1){
 if(value==null||!Number.isFinite(Number(value)))return '—';
 const n=Number(value)*100;return `${n>0?'+':''}${n.toLocaleString(undefined,{maximumFractionDigits:digits})}%`;
}
export function share(value:number|null|undefined){
 return value==null?'—':`${Math.round(Number(value)*100)}%`;
}
