// Ranks saved rows with the same code the routes use. Input: {kind:'voices'|'watchers', coin, posts}. No network.
import fs from 'node:fs';
import {rankVoices,VOICE_VERSION} from '../apps/web/lib/crypto-voices.ts';
import {rankWatchers,WATCHER_VERSION} from '../apps/web/lib/crypto-watchers.ts';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
if(input.kind==='voices'){const r=rankVoices(input.posts,input.coin);console.log(JSON.stringify({version:VOICE_VERSION,payload:{...r,total:input.posts.length,unfinished:input.unfinished??0}}));}
else{const ranked=rankWatchers(input.posts,input.coin);console.log(JSON.stringify({version:WATCHER_VERSION,payload:{accounts:ranked.slice(0,10),loaded:input.posts.length,total:input.posts.length,candidates:ranked.length,coin:input.coin}}));}
