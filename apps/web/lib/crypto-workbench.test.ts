import test from 'node:test';
import assert from 'node:assert/strict';
import {parseQuery,matchesQuery,describeScope,readState,writeState,legacyState,EMPTY} from './crypto-workbench.ts';
import {summarize,type CoinRole,type CoinImpact} from './crypto-leaders.ts';
import {COINS} from './crypto-coins.ts';
const role=(coin:string,day:number|null,contract=false):CoinRole=>({coin,role:day!=null?'Early discoverers':'Amplifiers',early:day!=null,analysis:false,amplifier:day==null,posts:3,first:'2026-08-01',day,contract});
const imp=(coin:string,episodes:number,up:number,x:number):CoinImpact=>({coin,episodes,median_24h:x,median_excess_24h:x,share_up_24h:up/episodes});
const a=summarize('1','Tyler_Did_It',null,[role('ZCAT',5),role('PONS',3)],[imp('ZCAT',9,7,.2)],null);
const b=summarize('2','whale_alert',null,[role('ZCAT',null)],[imp('ZCAT',2,1,.1)],'Whale watcher');
test('parses coins, handles, thresholds and contract addresses',()=>{
 const q=parseQuery('zcat @Tyler early day<4 hit>0.7 posts>5 watcher');
 assert.equal(q.coin,'ZCAT');assert.equal(q.handle,'tyler');assert.equal(q.early,true);assert.equal(q.day,4);assert.equal(q.hit,.7);assert.equal(q.posts,5);assert.equal(q.watcher,true);
 const withAddress=COINS.find(c=>c.address)!;const c=parseQuery(withAddress.address!);
 assert.equal(c.coin,withAddress.symbol);assert.equal(c.contract,withAddress.address);
 assert.equal(parseQuery('tyler').handle,'tyler');assert.deepEqual(parseQuery('??').unknown,['??']);
});
test('matches leaders against the query and the pinned coin',()=>{
 assert.equal(matchesQuery(a,parseQuery(''),null),true);
 assert.equal(matchesQuery(a,parseQuery(''),'PONS'),true);assert.equal(matchesQuery(a,parseQuery(''),'ZEC'),false);
 assert.equal(matchesQuery(a,parseQuery('day<4'),null),true);assert.equal(matchesQuery(a,parseQuery('day<3'),null),false);
 assert.equal(matchesQuery(a,parseQuery('hit>0.7'),null),true);assert.equal(matchesQuery(b,parseQuery('hit>0.1'),null),false,'thin sample never passes a hit threshold');
 assert.equal(matchesQuery(b,parseQuery('watcher'),null),true);assert.equal(matchesQuery(a,parseQuery('watcher'),null),false);
 assert.equal(describeScope(parseQuery('@tyler hit>0.7'),'ZCAT'),'ZCAT · @tyler · hit>0.7');
});
test('state round-trips through the query string',()=>{
 const s={...EMPTY,coin:'ZCAT',account:'9',tab:'posts' as const,q:'hit>0.5',day:'2026-09-01'};
 assert.equal(writeState(s),'/market/crypto?coin=ZCAT&account=9&tab=posts&q=hit%3E0.5&day=2026-09-01');
 assert.deepEqual(readState(new URLSearchParams('coin=zcat&account=9&tab=posts&q=hit%3E0.5&day=2026-09-01&to=nope')),{...s,to:null});
 assert.equal(writeState(EMPTY),'/market/crypto');assert.equal(readState(new URLSearchParams('tab=bogus&coin=BAD')).tab,'people');
});
test('old paths and ?view= links land on the same screen',()=>{
 assert.deepEqual(legacyState('/market/crypto/coins/ZEC',new URLSearchParams('day=2026-08-20')),{...EMPTY,coin:'ZEC',tab:'timeline',day:'2026-08-20'});
 assert.deepEqual(legacyState('/market/crypto/accounts/5',new URLSearchParams('')),{...EMPTY,account:'5'});
 assert.equal(legacyState('/market/crypto/people',new URLSearchParams(''))!.tab,'people');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('view=impact&coin=PONS'))!.coin,'PONS');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('view=network&coin=PONS'))!.tab,'connections');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('')),null);
});
