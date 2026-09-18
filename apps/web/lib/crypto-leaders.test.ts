import test from 'node:test';
import assert from 'node:assert/strict';
import {dayOneCoins,interestTrend,summarize,rankLeaders,whyLeader,afterPosting,postingStyles,dayAfterAnchor,type CoinRole,type CoinImpact} from './crypto-leaders.ts';
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

test('day-one coins and interest trend',()=>{
 const l=summarize('9','crew',null,[{coin:'KNOTS',role:'Early discoverers',early:true,analysis:false,amplifier:false,posts:4,first:'2026-09-01',day:1,contract:true},{coin:'ZCAT',role:'Early discoverers',early:true,analysis:false,amplifier:false,posts:2,first:'2026-08-01',day:6,contract:false}],[],null);
 assert.deepEqual(dayOneCoins(l),['KNOTS']);
 const now=Date.parse('2026-09-18T12:00:00Z');
 const fading=interestTrend([{week:'2026-08-24',posts:9},{week:'2026-08-31',posts:5},{week:'2026-09-07',posts:1},{week:'2026-09-14',posts:1}],'2026-09-15T00:00:00Z',now);
 assert.equal(fading.label,'fading');assert.deepEqual(fading.bars,[9,5,1,1]);assert.equal(fading.lastDays,3);
 assert.equal(interestTrend([{week:'2026-08-24',posts:2},{week:'2026-08-31',posts:1},{week:'2026-09-07',posts:4},{week:'2026-09-14',posts:3}],'2026-09-17T00:00:00Z',now).label,'rising');
 assert.equal(interestTrend([{week:'2026-08-24',posts:9}],'2026-08-25T00:00:00Z',now).label,'gone');
 assert.deepEqual(interestTrend([{week:'2026-08-24',posts:3}],'2026-08-25T00:00:00Z',now).bars,[3,0,0,0],'silent weeks show as gaps');
 assert.equal(interestTrend([{week:'2026-09-14',posts:3}],'2026-09-17T00:00:00Z',now).label,'new');
 assert.equal(interestTrend([],null,now).label,'new');
});
