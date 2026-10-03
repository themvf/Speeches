/** Backpack on X next to the BP holder data. Pure: the store loads rows, these functions shape them.
 *  A post counts only if it matches the BACKPACK registry entry (contract, $BACKPACK, or $BP with Backpack/Solana
 *  context); saved search results that merely contain the word "backpack" do not. */
import {matchesCoin} from './crypto-coins.ts';

export type SocialPost={id:string;text:string;url?:string|null;posted_at:string|Date;author_id:string;handle?:string|null;followers?:number|null};
export type CopyPasteGroup={key:string;accounts:number;posts:number;first:string;last:string;sample:string;url:string|null;handles:string[]};
export type DayInput={
 days:number;now:Date;posts:SocialPost[];
 windows:{day:string;windows:number;done:number;searched:number}[];
 holders:{day:string;unique_holders:unknown;holders_over_100:unknown;holders_complete:unknown}[];
 cohorts:{day:string;version_id:unknown;entered:unknown;left_count:unknown}[];
 trades:{day:string;bp_buyers:unknown;bp_sellers:unknown}[];
 tradesThrough:string|Date|null;
};

const iso=(v:string|Date)=>new Date(v).toISOString();
const dayOf=(v:string|Date)=>iso(v).slice(0,10);

export function genuinePosts(posts:SocialPost[],coin='BACKPACK'){
 const seen=new Set<string>();
 return posts.filter(p=>!seen.has(p.id)&&seen.add(p.id)&&matchesCoin(p.text,coin,'words'));
}

/** Text with links, mentions, numbers and punctuation removed; the first twelve words identify a template. */
export function templateKey(text:string){
 const words=text.toLowerCase().replace(/https?:\/\/\S+/g,' ').replace(/@\w+/g,' ').replace(/[1-9a-hj-np-z]{32,44}/gi,' ')
  .replace(/[^a-z$#\s]/g,' ').split(/\s+/).filter(Boolean);
 return words.length>=6?words.slice(0,12).join(' '):'';
}

/** Near-identical posts from several different accounts: a copy-paste campaign, whoever runs it. */
export function copyPasteGroups(posts:SocialPost[],minAccounts=3):CopyPasteGroup[]{
 const groups=new Map<string,SocialPost[]>();
 for(const p of posts){const key=templateKey(p.text);if(key)groups.set(key,[...(groups.get(key)??[]),p]);}
 return [...groups.entries()].map(([key,items])=>{
  const sorted=[...items].sort((a,b)=>iso(a.posted_at).localeCompare(iso(b.posted_at)));
  const handles=[...new Set(sorted.map(p=>String(p.handle??p.author_id)))];
  return {key,accounts:new Set(sorted.map(p=>p.author_id)).size,posts:sorted.length,first:iso(sorted[0].posted_at),
   last:iso(sorted[sorted.length-1].posted_at),sample:sorted[0].text.slice(0,240),url:sorted[0].url??null,handles:handles.slice(0,12)};
 }).filter(g=>g.accounts>=minAccounts).sort((a,b)=>b.accounts-a.accounts||a.first.localeCompare(b.first));
}

const num=(v:unknown)=>v===null||v===undefined||v===''?null:Number.isFinite(Number(v))?Number(v):null;

/** One row per UTC day, newest first. Missing observation stays null; zero means observed and none. */
export function socialDays(input:DayInput){
 const genuine=genuinePosts(input.posts);
 const campaign=new Set(copyPasteGroups(genuine).flatMap(g=>genuine.filter(p=>templateKey(p.text)===g.key).map(p=>p.id)));
 const byDay=new Map<string,SocialPost[]>();
 for(const p of genuine){const d=dayOf(p.posted_at);byDay.set(d,[...(byDay.get(d)??[]),p]);}
 const windows=new Map(input.windows.map(w=>[w.day,w])),holders=new Map(input.holders.map(h=>[h.day,h]));
 const trades=new Map(input.trades.map(t=>[t.day,t])),through=input.tradesThrough?dayOf(input.tradesThrough):null;
 const rows=[];
 for(let i=0;i<input.days;i++){
  const day=dayOf(new Date(input.now.getTime()-i*86_400_000)),w=windows.get(day),posts=byDay.get(day)??[];
  const coverage=!w||w.windows===0?'not_searched':w.done>=w.windows?'complete':(w.searched>0||w.done>0)?'partial':'not_searched';
  const h=holders.get(day),prior=holders.get(dayOf(new Date(new Date(day).getTime()-86_400_000)));
  const complete=(x?:{holders_complete:unknown})=>x?.holders_complete===true||x?.holders_complete==='true';
  const unique=complete(h)?num(h?.unique_holders):null,before=complete(prior)?num(prior?.unique_holders):null;
  const versions=input.cohorts.filter(c=>c.day===day),t=trades.get(day),observed=through!==null&&day<=through;
  rows.push({day,
   posts:posts.length||coverage!=='not_searched'?posts.length:null,
   authors:posts.length||coverage!=='not_searched'?new Set(posts.map(p=>p.author_id)).size:null,
   copy_paste_posts:posts.length||coverage!=='not_searched'?posts.filter(p=>campaign.has(p.id)).length:null,
   x_coverage:coverage,x_windows:w?.windows??0,x_windows_done:w?.done??0,
   holders:unique,holders_over_100:complete(h)?num(h?.holders_over_100):null,
   holders_change:unique!==null&&before!==null?unique-before:null,
   cohort_version:versions.length?versions.map(c=>String(c.version_id)).join(','):null,
   cohort_entered:versions.length?versions.reduce((a,c)=>a+(num(c.entered)??0),0):null,
   cohort_left:versions.length?versions.reduce((a,c)=>a+(num(c.left_count)??0),0):null,
   bp_buyers:observed?num(t?.bp_buyers)??0:null,bp_sellers:observed?num(t?.bp_sellers)??0:null});
 }
 return rows;
}

/** Recent posts per tracked coin, counted only where they match that coin. */
export function coinActivity(posts:(SocialPost&{coin:string})[]){
 const out=new Map<string,{posts:number;accounts:Set<string>}>();
 const seen=new Set<string>();
 for(const p of posts){
  const k=`${p.coin}:${p.id}`;if(seen.has(k)||!matchesCoin(p.text,p.coin,'words'))continue;seen.add(k);
  const e=out.get(p.coin)??{posts:0,accounts:new Set<string>()};e.posts++;e.accounts.add(p.author_id);out.set(p.coin,e);
 }
 return new Map([...out].map(([coin,e])=>[coin,{posts:e.posts,accounts:e.accounts.size}]));
}
