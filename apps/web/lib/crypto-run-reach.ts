import type {RunPost} from './crypto-run.ts';

export function largeAccounts(posts:RunPost[],moveDay:string){
 const groups=new Map<string,RunPost[]>();
 for(const p of posts){const group=groups.get(p.author_id)??[];if(!group.some(x=>x.id===p.id))group.push(p);groups.set(p.author_id,group);}
 return [...groups.entries()].map(([id,group])=>{
  group.sort((a,b)=>a.posted_at.localeCompare(b.posted_at)||a.id.localeCompare(b.id));
  const profile=[...group].sort((a,b)=>(b.followers_observed_at??'').localeCompare(a.followers_observed_at??''))[0];
  const likes=group.flatMap(p=>p.likes==null?[]:[p.likes]).sort((a,b)=>a-b);
  const middle=Math.floor(likes.length/2);
  const medianLikes=likes.length?(likes.length%2?likes[middle]:(likes[middle-1]+likes[middle])/2):null;
  const measured=group.filter(p=>p.quotes!=null&&p.reposts!=null);
  return {id,handle:profile.handle,followers:profile.followers??null,observed:profile.followers_observed_at??null,
   first:group[0],posts:group.length,before:moveDay?group.filter(p=>p.posted_at.slice(0,10)<moveDay).length:null,
   medianLikes,likeSamples:likes.length,amplification:measured.length?measured.reduce((n,p)=>n+p.quotes!+p.reposts!,0):null,
   amplificationSamples:measured.length};
 }).sort((a,b)=>(b.followers??-1)-(a.followers??-1)||a.id.localeCompare(b.id));
}
