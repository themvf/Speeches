import type {RunPost} from './crypto-run';
export type PostFilters={query:string;exclude:string;author:string;kind:string;from:string;to:string;followers:number;likes:number;reposts:number;sort:string;onePerAuthor:boolean};
export const defaultPostFilters:PostFilters={query:'',exclude:'',author:'',kind:'all',from:'',to:'',followers:0,likes:0,reposts:0,sort:'newest',onePerAuthor:false};
// Quoted phrases stay together; other terms are ANDed. Exclusions are ORed.
export function searchTerms(value:string){return (value.toLowerCase().match(/"[^"]+"|\S+/g)??[]).map(s=>s.replace(/^"|"$/g,'')).filter(Boolean);}
export function filterSavedPosts<T extends RunPost>(posts:T[],f:PostFilters):T[]{
 const include=searchTerms(f.query),exclude=searchTerms(f.exclude),author=f.author.trim().replace(/^@/,'').toLowerCase();
 const seen=new Set<string>();
 let result=posts.filter(p=>{if(seen.has(p.id))return false;seen.add(p.id);const text=p.text.toLowerCase(),day=p.posted_at.slice(0,10);return include.every(t=>text.includes(t))&&!exclude.some(t=>text.includes(t))&&p.handle.toLowerCase().includes(author)&&(f.kind==='all'||p.kind===f.kind)&&(!f.from||day>=f.from)&&(!f.to||day<=f.to)&&(!f.followers||(p.followers??-1)>=f.followers)&&(!f.likes||(p.likes??-1)>=f.likes)&&(!f.reposts||(p.reposts??-1)>=f.reposts);});
 result.sort((a,b)=>{const metric=f.sort==='followers'?'followers':f.sort==='likes'?'likes':f.sort==='reposts'?'reposts':null;return (metric?(b[metric]??-1)-(a[metric]??-1):f.sort==='oldest'?a.posted_at.localeCompare(b.posted_at):b.posted_at.localeCompare(a.posted_at))||a.id.localeCompare(b.id);});
 if(f.onePerAuthor){const authors=new Set<string>();result=result.filter(p=>{if(authors.has(p.author_id))return false;authors.add(p.author_id);return true;});}return result;
}
