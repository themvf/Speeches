export type SocialAccount = {
 id:string;handle:string;name:string;category:string;tracked:boolean;reason:string;
 followers:number|null;following:number|null;available:boolean|null;bio:string|null;
 profile_observed_at:string|null;baseline_at:string|null;older_baseline_at:string|null;
 growth_7d:number|null;growth_percent_7d:number|null;growth_acceleration:number|null;
 posts:number;active_days:number;originals:number|null;replies:number|null;quotes:number|null;reposts:number|null;
 median_engagement:number|null;engagement_samples:number|null;median_engagement_24_30h:number|null;age_matched_samples:number|null;
 participants:number;amplifiers:number;participants_7d:number;participants_previous_7d:number;
 connections:number;repeat_participants:number;top5_attention_percent:number|null;reciprocal_participants:number;
 evidence:string|null;daily_observations:number;
};
export type Leaderboard = 'attention'|'growth'|'emerging'|'posting'|'connections'|'bios';
export type ProfilePoint = {account_id:string;observed_at:string;followers:number|null;available:boolean};
export type Tracking = {
 campaign:{started_at:string;end_at:string}|null;
 accounts:SocialAccount[];history:ProfilePoint[];
 bioMatches:{account_id:string;field:string;term:string}[];
 bioChanges:{account_id:string;handle:string;bio:string;previous_bio:string;observed_at:string}[];
 coverage:{account_id:string;handle:string;start_at:string;end_at:string;in_window_posts:number;status:string}[];
 ledger:{allocation:string;reserved_credits:number;estimated_credits:number|null;requests:number}[];
 keywordSearchStatus:string;lookbackDays:number;
};
export function rankAccounts(accounts:SocialAccount[],board:Leaderboard,bioIds:Set<string>=new Set()) {
 const rows=accounts.filter(a=>board==='growth'?a.growth_7d!=null&&Number(a.growth_7d)>0:
  board==='emerging'?a.growth_7d!=null&&Number(a.growth_7d)>0&&a.participants_7d>a.participants_previous_7d:
  board==='bios'?bioIds.has(a.id):true);
 return rows.sort((a,b)=>{
  const score=(x:SocialAccount)=>board==='growth'?Number(x.growth_7d):board==='emerging'?x.participants_7d-x.participants_previous_7d:
   board==='posting'?x.posts:board==='connections'?x.connections:x.participants;
  return score(b)-score(a)||b.amplifiers-a.amplifiers||a.id.localeCompare(b.id);
 });
}
export function formatCount(value:number|string|null|undefined) {
 return value==null?'—':Number(value).toLocaleString(undefined,{maximumFractionDigits:1});
}
export function growthLabel(value:number|null|undefined,percent=false) {
 return value==null?'—':`${Number(value)>0?'+':''}${formatCount(value)}${percent?'%':''}`;
}
