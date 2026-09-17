import test from 'node:test';
import assert from 'node:assert/strict';
import {summarize,rankLeaders,whyLeader,afterPosting,postingStyles,dayAfterAnchor,type CoinRole,type CoinImpact} from './crypto-leaders.ts';
const role=(coin:string,early=false,extra:Partial<CoinRole>={}):CoinRole=>({coin,role:early?'Early discoverers':'Amplifiers',early,analysis:false,amplifier:!early,posts:3,first:'2026-08-01',day:early?2:null,contract:early,...extra});
const imp=(coin:string,episodes:number,x:number|null,up=.5):CoinImpact=>({coin,episodes,median_24h:x,median_excess_24h:x,share_up_24h:up});
test('cross-coin summary weights per-coin medians by episodes and counts early coins and ups',()=>{
 const l=summarize('1','a',null,[role('ZCAT',true),role('PONS')],[imp('ZCAT',1,.5,1),imp('PONS',3,-.1,1/3)],null);
 assert.deepEqual(l.coins,['PONS','ZCAT']);assert.equal(l.early_coins,1);assert.equal(l.episodes,4);assert.equal(l.median_excess_24h,-.1);
 assert.equal(l.up,2);assert.equal(l.hit_rate,.5);
 assert.match(whyLeader(l),/early on ZCAT day 2/);assert.match(whyLeader(l),/up 2 of 4 times · typically -10% vs the coin's own drift/);
});
test('ranking: early breadth, then a real sample beats none, then hit rate, then move size',()=>{
 const a=summarize('a','a',null,[role('ZCAT',true),role('PONS',true)],[],null);
 const b=summarize('b','b',null,[role('ZCAT',true)],[imp('ZCAT',5,.9,.6)],'Whale watcher');
 const c=summarize('c','c',null,[],[imp('ZEC',9,.2,.8)],null),d=summarize('d','d',null,[],[imp('ZEC',9,.5,.6)],null),e=summarize('e','e',null,[],[imp('ZEC',2,.99,1)],null);
 assert.deepEqual(rankLeaders([e,d,c,b,a]).map(l=>l.account_id),['a','b','c','d','e']);
});
test('after-posting text needs three linked posts',()=>{
 assert.equal(afterPosting(summarize('e','e',null,[],[imp('ZEC',2,.99,1)],null)),null);
 assert.equal(afterPosting(summarize('c','c',null,[],[imp('ZEC',9,.2,.8)],null)),"up 7 of 9 times · typically +20% vs the coin's own drift a day later");
 assert.equal(whyLeader(summarize('e','e',null,[],[imp('ZEC',2,.99,1)],null)),'2 price-linked posts · too few to say');
});
test('posting styles collapse roles into plain phrases',()=>{
 const l=summarize('1','a',null,[role('ZCAT',true),role('PONS',false,{analysis:true}),role('ZEC',false,{role:'Reporting feeds'})],[],'Whale watcher');
 assert.deepEqual(postingStyles(l),['first','analysis','feed','watcher']);
 assert.deepEqual(postingStyles(summarize('2','b',null,[role('PONS')],[],null)),['amplifier']);
 assert.deepEqual(postingStyles(summarize('3','c',null,[role('PONS',true,{contract:false})],[],null)),['early']);
});
test('day after anchor counts the anchor day as day 1',()=>{
 assert.equal(dayAfterAnchor('2026-08-01T10:00:00Z','2026-08-01T08:00:00Z'),1);
 assert.equal(dayAfterAnchor('2026-08-03T01:00:00Z','2026-08-01T08:00:00Z'),2);
 assert.equal(dayAfterAnchor('2026-08-03T01:00:00Z',null),null);
});
