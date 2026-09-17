import test from 'node:test';
import assert from 'node:assert/strict';
import {summarize,rankLeaders,whyLeader,type CoinRole,type CoinImpact} from './crypto-leaders.ts';
const role=(coin:string,early=false):CoinRole=>({coin,role:early?'Early discoverers':'Amplifiers',early,analysis:false,amplifier:!early,posts:3,first:'2026-08-01'});
const imp=(coin:string,episodes:number,x:number|null):CoinImpact=>({coin,episodes,median_24h:x,median_excess_24h:x,share_up_24h:.5});
test('cross-coin summary weights per-coin medians by episodes and counts early coins',()=>{
 const l=summarize('1','a',null,[role('ZCAT',true),role('PONS')],[imp('ZCAT',1,.5),imp('PONS',3,-.1)],null);
 assert.deepEqual(l.coins,['PONS','ZCAT']);assert.equal(l.early_coins,1);assert.equal(l.episodes,4);assert.equal(l.median_excess_24h,-.1);
 assert.match(whyLeader(l),/early on 1 coin \(ZCAT\)/);
});
test('ranking prefers early breadth, then coin count, then excess return',()=>{
 const a=summarize('a','a',null,[role('ZCAT',true),role('PONS',true)],[],null),b=summarize('b','b',null,[role('ZCAT',true)],[imp('ZCAT',5,.9)],'Whale watcher'),c=summarize('c','c',null,[],[imp('ZEC',9,.2)],null);
 assert.deepEqual(rankLeaders([c,b,a]).map(l=>l.account_id),['a','b','c']);assert.equal(whyLeader(c),'9 price-linked episodes');
});
