import test from 'node:test';
import assert from 'node:assert/strict';
import {rankImpact,confidence,pct,share,type ImpactRow} from './crypto-impact.ts';
const row=(o:Partial<ImpactRow>):ImpactRow=>({account_id:'1',handle:'a',followers:null,coins:['ZCAT'],episodes:3,posts:5,median_1h:0,median_6h:0,median_24h:.1,median_excess_24h:.05,share_up_24h:.6,share_beat_24h:.5,median_volume_ratio:1,first_at:'2026-09-01',last_at:'2026-09-10',best_post_id:null,best_post_url:null,best_return_24h:null,worst_return_24h:null,...o});
test('ranks by median excess return, requires a minimum sample, and respects the coin filter',()=>{
 const rows=[row({account_id:'1',median_excess_24h:.05}),row({account_id:'2',median_excess_24h:.2,episodes:2}),row({account_id:'3',median_excess_24h:.2,coins:['PONS']}),row({account_id:'4',median_excess_24h:null})];
 assert.deepEqual(rankImpact(rows).map(r=>r.account_id),['3','1','4']);
 assert.deepEqual(rankImpact(rows,1).map(r=>r.account_id),['3','2','1','4']);
 assert.deepEqual(rankImpact(rows,1,'PONS').map(r=>r.account_id),['3']);
});
test('confidence labels and formatting never invent precision',()=>{
 assert.equal(confidence(row({episodes:12,coins:['ZCAT','PONS']})),'Repeated across coins');
 assert.equal(confidence(row({episodes:12})),'Repeated evidence');
 assert.equal(confidence(row({episodes:1})),'Single observations');
 assert.equal(pct(.1234),'+12.3%');assert.equal(pct(-.05),'-5%');assert.equal(pct(null),'—');assert.equal(pct('0.02' as unknown as number),'+2%');
 assert.equal(share(.666),'67%');assert.equal(share(null),'—');
});
