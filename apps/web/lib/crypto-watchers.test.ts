import test from 'node:test';
import assert from 'node:assert/strict';
import {rankWatchers,watcherSignals,type WatcherPost} from './crypto-watchers.ts';
const p:WatcherPost={id:'1',author_id:'a',handle:'alice',text:'$ZCAT whale wallet bought $500K',url:'https://x.com/i/status/1',posted_at:'2026-09-14T12:00:00Z',kind:'original',edges:[],coins:['ZCAT'],followers:100};
test('requires numeric reporting rather than whale hype or irrelevant coin text',()=>{assert.equal(watcherSignals('whales incoming').supported,false);assert.equal(rankWatchers([{...p,text:'whale wallet bought $500K of unrelated coin'}]).length,0);assert.equal(rankWatchers([p]).length,1);assert.equal(watcherSignals('24h volume: $2.5M').supported,true);assert.equal(watcherSignals('聪明钱包 买入$55.9K').supported,true);});
test('deduplicates posts and excludes repost-only accounts',()=>{assert.equal(rankWatchers([p,p])[0].matched,1);assert.equal(rankWatchers([{...p,kind:'repost'}]).length,0);});
test('repeated reports outrank a single big audience and coin scope is respected',()=>{const rows=rankWatchers([p,{...p,id:'2',posted_at:'2026-09-15T12:00:00Z'},{...p,id:'3',author_id:'b',followers:10000000}]);assert.equal(rows[0].id,'a');assert.equal(rankWatchers([p],'STANDARD').length,0);assert.equal(rows[0].confidence,'Provisional');});
