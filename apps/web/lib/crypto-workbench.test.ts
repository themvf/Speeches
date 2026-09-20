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
test('ring token filters by computed ring',()=>{
 const rings=new Map([['1',{ring:1 as const,tags:[1 as const],posts:9,evidence:'x'}]]);
 assert.equal(parseQuery('ring:2').ring,2);assert.equal(matchesQuery(a,parseQuery('ring:1'),null,rings),true);assert.equal(matchesQuery(b,parseQuery('ring:1'),null,rings),false);
 assert.equal(describeScope(parseQuery('ring:2'),'ZCAT'),'ZCAT · ring 2 defender');
});
test('day1 and coins>N tokens',()=>{
 const d1=summarize('3','launch_crew',null,[role('KNOTS',1),role('STONK',1),role('ZCAT',6)],[],null);
 assert.equal(parseQuery('day1 coins>2').day1,true);assert.equal(parseQuery('d1 coins>2').coins,2);
 assert.equal(matchesQuery(d1,parseQuery('day1'),null),true);assert.equal(matchesQuery(a,parseQuery('day1'),null),false,'day 5 is early, not day 1');
 assert.equal(matchesQuery(d1,parseQuery('coins>2'),null),true);assert.equal(matchesQuery(a,parseQuery('coins>2'),null),false);
 assert.equal(describeScope(parseQuery('day1 coins>2'),null),'all coins · day-1 · coins>2');
 const circle=new Map([['1',{handle:'Tyler_Did_It',inside:true,count:2,from:new Map([['spaceman',2]])}]]);
 assert.equal(parseQuery('circle').circle,true);assert.equal(matchesQuery(a,parseQuery('circle'),null,undefined,circle),true);assert.equal(matchesQuery(b,parseQuery('circle'),null,undefined,circle),false);
});
test('state round-trips through the query string',()=>{
 const s={...EMPTY,coin:'ZCAT',account:'9',tab:'posts' as const,q:'hit>0.5',day:'2026-09-01'};
 assert.equal(writeState(s),'/market/crypto?coin=ZCAT&account=9&tab=posts&q=hit%3E0.5&day=2026-09-01');
 assert.deepEqual(readState(new URLSearchParams('coin=zcat&account=9&tab=posts&q=hit%3E0.5&day=2026-09-01&to=nope')),{...s,to:null});
 assert.equal(writeState(EMPTY),'/market/crypto');assert.equal(readState(new URLSearchParams('tab=bogus&coin=BAD')).tab,'people');
 assert.equal(readState(new URLSearchParams('tab=brief&window=30d')).window,'30d');
 assert.equal(writeState({...EMPTY,coin:'BACKPACK',tab:'brief',window:'7d'}),'/market/crypto?coin=BACKPACK&tab=brief&window=7d');
});
test('old paths and ?view= links land on the same screen',()=>{
 assert.deepEqual(legacyState('/market/crypto/coins/ZEC',new URLSearchParams('day=2026-08-20')),{...EMPTY,coin:'ZEC',tab:'timeline',day:'2026-08-20'});
 assert.deepEqual(legacyState('/market/crypto/accounts/5',new URLSearchParams('')),{...EMPTY,account:'5'});
 assert.equal(legacyState('/market/crypto/people',new URLSearchParams(''))!.tab,'people');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('view=impact&coin=PONS'))!.coin,'PONS');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('view=network&coin=PONS'))!.tab,'connections');
 assert.equal(legacyState('/market/crypto',new URLSearchParams('')),null);
});
