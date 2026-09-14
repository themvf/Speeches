import test from 'node:test';
import assert from 'node:assert/strict';
import {accountRoles,sentimentDays} from './crypto-intelligence.ts';
const post=(id:string,author_id=id,day='2026-09-01')=>({id,author_id,handle:author_id,text:'coin',url:'',posted_at:day+'T12:00:00Z',kind:'original',edges:[]});
const sentiment=(post_id:string,label:string,coin='PONS')=>({post_id,coin,label,confidence:.8,explanation:'example',model:'test',version:'v1',observed_at:'2026-09-14'});
test('sentiment stays coin-specific, unknown is not neutral, duplicate posts count once',()=>{
 const posts=[post('a'),post('b'),post('c')];const days=sentimentDays([...posts,posts[0]],[sentiment('a','bullish'),sentiment('b','bearish','ZCAT')],'PONS',false);
 assert.equal(days[0].total,3);assert.equal(days[0].classified,1);assert.equal(days[0].counts.unclassified,2);assert.equal(days[0].bullish,100);assert.equal(days[0].counts.neutral,0);
});
test('author balancing combines conflicting stances and keeps completely unclassified authors',()=>{
 const posts=[post('a','same'),post('b','same'),post('c','other')];const day=sentimentDays(posts,[sentiment('a','bullish'),sentiment('b','bearish')],'PONS',true)[0];
 assert.equal(day.total,2);assert.equal(day.counts.mixed,1);assert.equal(day.counts.unclassified,1);
});
test('roles distinguish observed audience from holdings and require sufficient corpus for early labels',()=>{
 const p={...post('a'),followers:200000};const r=accountRoles([p]).get('a')!;
 assert.deepEqual(r.map(x=>x.label),['Large audience']);
 const corpus=[p,...Array.from({length:10},(_,i)=>post(String(i),String(i),'2026-09-10'))];
 assert.ok(accountRoles(corpus).get('a')!.some(r=>r.label==='Early observed voice'));
 assert.ok(![...accountRoles(corpus).values()].flat().some(r=>/whale/i.test(r.label)));
});
