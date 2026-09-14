import type {RunPost} from './crypto-run';
export type Role={label:string;reason:string;confidence:string};
export type Sentiment={post_id:string;coin:string;label:string;confidence:number;explanation:string;model:string;version:string;observed_at:string};
export const sentimentLabels=['bullish','bearish','neutral','mixed','unclear'];
export function accountRoles(posts:RunPost[]):Map<string,Role[]>{
 const unique=[...new Map(posts.map(p=>[p.id,p])).values()].sort((a,b)=>a.posted_at.localeCompare(b.posted_at));
 const grouped=new Map<string,RunPost[]>();for(const p of unique)grouped.set(p.author_id,[...(grouped.get(p.author_id)??[]),p]);
 const first=unique[0]?.posted_at.slice(0,10),last=unique.at(-1)?.posted_at.slice(0,10);
 const mature=first&&last&&Date.parse(last)-Date.parse(first)>=7*86400000&&grouped.size>=10;
 const roles=new Map<string,Role[]>();
 for(const [id,items] of grouped){const r:Role[]=[],latest=[...items].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
 if((latest.followers??0)>=100000)r.push({label:'Large audience',confidence:'Observed',reason:`${latest.followers!.toLocaleString()} followers observed ${latest.followers_observed_at?.slice(0,10)??'at an unknown date'}. Potential reach, not proven influence.`});
 if(mature&&Date.parse(items[0].posted_at.slice(0,10))-Date.parse(first!)<3*86400000)r.push({label:'Early observed voice',confidence:'Provisional',reason:`First saved post ${items[0].posted_at.slice(0,10)}, within the first three days of this coin’s saved corpus. Not proof of early adoption or a pre-breakout call.`});
 const amplifies=items.filter(p=>p.kind==='quote'||p.kind==='repost').length;
 if(amplifies>=3&&amplifies/items.length>=.5)r.push({label:'Amplifier',confidence:'Observed pattern',reason:`${amplifies} of ${items.length} saved posts are quotes or reposts; amplification may include criticism.`});
 const incoming=new Set(unique.filter(p=>p.author_id!==id&&p.edges.some(e=>e.target_id===id)).map(p=>p.author_id));
 if(incoming.size>=5)r.push({label:'Conversation participant',confidence:'Observed pattern',reason:`Referenced by ${incoming.size} distinct accounts. References do not establish endorsement or that this account started the conversation.`});
 const promo=items.filter(p=>/join my telegram|join (my|our) (group|channel)|next (100|1000)x|don't miss|do not miss|buy now/i.test(p.text)).length;
 if(promo>=3)r.push({label:'Repeated promotional language',confidence:'Provisional',reason:`${promo} saved posts contain explicit promotional phrases. Rule-based indicator; inspect context. This does not establish paid promotion.`});
 roles.set(id,r);
 }return roles;
}
export function sentimentDays(posts:RunPost[],annotations:Sentiment[],coin:string,balanced:boolean){
 const map=new Map(annotations.filter(a=>a.coin===coin).map(a=>[a.post_id,a]));
 const groups=new Map<string,Map<string,Sentiment[]>>();
 for(const p of [...new Map(posts.map(p=>[p.id,p])).values()]){const day=p.posted_at.slice(0,10);if(!groups.has(day))groups.set(day,new Map());const authors=groups.get(day)!;const key=balanced?p.author_id:p.id;const list=authors.get(key)??[];const a=map.get(p.id);if(a)list.push(a);authors.set(key,list);}
 return [...groups].sort(([a],[b])=>a.localeCompare(b)).map(([day,groups])=>{const counts:Record<string,number>={bullish:0,bearish:0,neutral:0,mixed:0,unclear:0,unclassified:0};for(const list of groups.values()){if(!list.length){counts.unclassified++;continue;}const labels=new Set(list.map(a=>a.label));counts[labels.size===1?list[0].label:'mixed']++;}const classified=groups.size-counts.unclassified;return {day,counts,total:groups.size,classified,bullish:classified?counts.bullish/classified*100:null,bearish:classified?counts.bearish/classified*100:null};});
}
