import fs from 'node:fs';
import {rankVoices,VOICE_VERSION} from '../apps/web/lib/crypto-voices.ts';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result=rankVoices(input.posts,input.coin,input.cutoff);
const eligible=result.voices.filter(v=>!['Project / platform','Reporting feeds'].includes(v.role));
const select=(role:string)=>[...eligible].filter(v=>v.scores[role]>0).sort((a,b)=>b.scores[role]-a.scores[role]||a.id.localeCompare(b.id));
const reach=[...eligible].sort((a,b)=>(b.followers??0)-(a.followers??0)||a.id.localeCompare(b.id));
const posting=[...eligible].sort((a,b)=>b.posts-a.posts||a.id.localeCompare(b.id));
const chosen=new Map();
for(const [role,rows] of [['Audience',reach],['Early discoverers',select('Early discoverers')],['Original analysis',select('Original analysis')],['Amplifiers',select('Amplifiers')]] as const){let n=0;for(const v of rows)if(!chosen.has(v.id)){chosen.set(v.id,{id:v.id,handle:v.handle,role});if(++n===5)break;}}
for(const v of select('Amplifiers'))if(chosen.size<20&&!chosen.has(v.id))chosen.set(v.id,{id:v.id,handle:v.handle,role:'Amplifiers'});
console.log(JSON.stringify({model:VOICE_VERSION,selection:[...chosen.values()],baseline:{reach:reach.slice(0,20).map(v=>v.id),posting:posting.slice(0,20).map(v=>v.id)},coverage:{loaded:result.loaded,eligible:result.eligible,anchor:result.anchor},eligibleIds:result.voices.map(v=>v.id)}));
