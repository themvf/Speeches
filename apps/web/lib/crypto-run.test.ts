import test from 'node:test';
import assert from 'node:assert/strict';
import {filterEvidence,peopleIn,networkEdges,layoutNetwork,scopePosts,largestDailyGain,daysBetween,type RunPost} from './crypto-run.ts';
import {verifiedPools,candles,ZCAT_ADDRESS} from './server/crypto-run-market.ts';
const post:RunPost={id:'1',author_id:'a',handle:'alice',text:'example',url:'https://x.com/alice/status/1',posted_at:'2026-08-01T00:00:00Z',kind:'quote',edges:[{target_id:'b',target:'bob',kind:'quote'},{target_id:'b',target:'bob',kind:'mention'},{target_id:'a',target:'alice',kind:'mention'}]};
test('target-only accounts remain inspectable; participants and source posts are deduplicated',()=>{const bob=peopleIn([post]).find(p=>p.id==='b')!;assert.equal(bob.posts.length,0);assert.equal(bob.incoming.length,1);assert.equal(bob.participants,1);assert.equal(networkEdges([post]).length,2);assert.equal(networkEdges([post],'quote').length,1);});
test('UTC selection includes both boundary days and excludes others',()=>{const posts=[post,{...post,id:'2',posted_at:'2026-08-02T23:59:59Z'},{...post,id:'3',posted_at:'2026-08-03T00:00:00Z'}];assert.deepEqual(scopePosts(posts,'2026-08-01','2026-08-02').map(p=>p.id),['1','2']);assert.deepEqual(daysBetween('2026-07-31','2026-08-01'),['2026-07-31','2026-08-01']);});
test('daily gain never bridges missing days or uses zero-price baselines',()=>{const p=(day:string,close:number)=>({day,close,volume:1});assert.equal(largestDailyGain([p('2026-08-01',1),p('2026-08-03',100)]),null);assert.equal(largestDailyGain([p('2026-08-01',0),p('2026-08-02',100)]),null);assert.deepEqual(largestDailyGain([p('2026-08-03',2),p('2026-08-02',1)]),{day:'2026-08-03',percent:100});});
test('network layout is bounded, deterministic and excludes links to omitted nodes',()=>{const edges=networkEdges(Array.from({length:50},(_,i)=>({...post,id:String(i),author_id:String(i)})));const a=layoutNetwork(edges);assert.deepEqual(a,layoutNetwork(edges));assert.equal(a.nodes.length,32);const ids=new Set(a.nodes.map(n=>n.id));assert.ok(a.nodes.every(n=>Number.isFinite(n.x)&&n.x>=90&&n.x<=770&&n.y>=40&&n.y<=425));assert.ok(a.links.every(e=>ids.has(e.source)&&ids.has(e.target)));});
test('pools require the exact contract and preserve quote-token orientation',()=>{const attributes={address:'BTccxxTFi7a9xJTE1exKn38Jgie35s6gNeRxd8DM61Rc',name:'ZEC / ZCAT',pool_created_at:'2026-08-30T23:29:55Z',reserve_in_usd:'1500'};assert.equal(verifiedPools([{attributes,relationships:{quote_token:{data:{id:'solana_unrelated'}}}}]).length,0);assert.equal(verifiedPools([{attributes,relationships:{quote_token:{data:{id:`solana_${ZCAT_ADDRESS}`}}}}])[0].side,'quote');});
test('candles reject malformed values, deduplicate days and leave gaps unfilled',()=>{const row=(day:string)=>[Date.parse(day)/1000,1,2,1,2,100];const result=candles([row('2026-08-01'),row('2026-08-03'),row('2026-08-01'),[1e100,1,2,1,2,100],[1,1,1,1,-2,10],null]);assert.deepEqual(result.map(p=>p.day),['2026-08-01','2026-08-03']);});

test('evidence filter separates contract, bounded coin words and unrelated search results',()=>{const sample=[{...post,text:'$ZCAT rising'},{...post,id:'2',text:ZCAT_ADDRESS},{...post,id:'3',text:'unrelated'},{...post,id:'4',text:'zcatfish'}];assert.equal(filterEvidence(sample,'ZCAT','words').length,2);assert.equal(filterEvidence(sample,'ZCAT','contract').length,1);assert.equal(filterEvidence(sample,'ZCAT','all').length,4);assert.equal(filterEvidence([{...post,text:'Zcash and $ZEC'}],'ZEC','words').length,1);});
test('incomplete daily candles never produce a largest-gain insight',()=>{assert.equal(largestDailyGain([{day:'2026-09-13',close:1,volume:1,complete:true},{day:'2026-09-14',close:50,volume:1,complete:false}]),null);});
test('PONS evidence accepts checksum addresses and contextual mentions, not unrelated names',()=>{const sample=[{...post,text:'0x39dBED3a2bd333467115dE45665cC57F813C4571'},{...post,id:'2',text:'$PONS news'},{...post,id:'3',text:'Pons dictionary'},{...post,id:'4',text:'PONS on Robinhood'}];assert.equal(filterEvidence(sample,'PONS','contract').length,1);assert.equal(filterEvidence(sample,'PONS','words').length,3);assert.equal(filterEvidence(sample,'PONS','all').length,4);});

test('large accounts rank current audiences and keep missing metrics distinct from zero',async()=>{
 const {largeAccounts}=await import('./crypto-run-reach.ts');
 const sample=[{...post,followers:100000,followers_observed_at:'2026-09-14',likes:0,quotes:0,reposts:0},
 {...post,id:'2',posted_at:'2026-08-02T00:00:00Z',followers:100000,followers_observed_at:'2026-09-14',likes:10,quotes:2,reposts:3},
 {...post,id:'3',author_id:'b',handle:'bob',followers:1000000,followers_observed_at:'2026-09-14'},
 {...post,id:'4',author_id:'c',handle:'carol',followers:null}];
 const rows=largeAccounts([...sample,sample[0]],'2026-08-02');
 assert.deepEqual(rows.map(r=>r.id),['b','a','c']);
 const a=rows[1];assert.equal(a.posts,2);assert.equal(a.before,1);assert.equal(a.medianLikes,5);assert.equal(a.amplification,5);
 assert.equal(rows[0].medianLikes,null);assert.equal(rows[0].amplification,null);assert.equal(rows[2].followers,null);
 assert.equal(largeAccounts(sample,'')[0].before,null);
});

test('DPONS evidence is distinct from PONS and accepts contract casing',()=>{const texts=['0x0E6D1EBB33F3B8F2D09BACF3B1A1D5C581110C33','$DPONS news','Diamond Pons news','$PONS news','dponsfake'];const sample=texts.map((text,i)=>({...post,id:String(i),text}));assert.deepEqual(filterEvidence(sample,'DPONS','words').map(p=>p.id),['0','1','2']);assert.deepEqual(filterEvidence(sample,'DPONS','contract').map(p=>p.id),['0']);});

test('STANDARD excludes ordinary words while matching project and checksum contract',()=>{const texts=['standard procedure','gold standard','The Standard Reserve','$STANDARD','0x88ad8DdF1E3898412146a534538d418c6F8A9062','standard_rsv'];const sample=texts.map((text,i)=>({...post,id:String(i),text}));assert.deepEqual(filterEvidence(sample,'STANDARD','words').map(p=>p.id),['2','3','4','5']);assert.deepEqual(filterEvidence(sample,'STANDARD','contract').map(p=>p.id),['4']);});
