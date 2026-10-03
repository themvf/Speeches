import test from 'node:test';
import assert from 'node:assert/strict';
import {standing,runs,extremes,relativeActivity,byEasternHour,EVEN_SHARE,type Pooled,type HourRow} from './crypto-hour-profile.ts';

const row=(et_hour:number,mult:number):HourRow=>({utc_hour:(et_hour+4)%24,et_hour,n:14,mean:mult*EVEN_SHARE,median:null,t:null});
const pooled=(coin:number,splitP:number|null):Pooled=>({coins:12,units:100,table:[],
 p_global:{day:0.001,coin,market:0.01},split_half:splitP==null?null:{r:0.47,p:splitP,coins:10}});

test('standing needs both tests, because they answer different questions',()=>{
 assert.equal(standing(pooled(0.01,0.02)),'recurring');        // flat rejected AND the shape returns
 assert.equal(standing(pooled(0.01,0.40)),'this window only'); // significant here, gone next half
 assert.equal(standing(pooled(0.07,0.04)),'recurs, unproven'); // the real activity case: 12 coins
 assert.equal(standing(pooled(0.40,0.40)),'none');
 assert.equal(standing(pooled(0.01,null)),'this window only'); // no holdout is not a pass
 assert.equal(standing(undefined),'not tested');
 assert.equal(standing({coins:0,units:0,table:[],test:'too short to test'}),'not tested');
});

test('runs collapse contiguous hours and wrap past midnight',()=>{
 assert.deepEqual(runs([2,3,4,5,6,7]),['2:00–8:00']);
 assert.deepEqual(runs([13,14]),['13:00–15:00']);
 assert.deepEqual(runs([]),[]);
 assert.deepEqual(runs([9]),['9:00–10:00']);
 // A run crossing midnight is one stretch, not two.
 assert.deepEqual(runs([22,23,0,1]),['22:00–2:00']);
 // Two separate stretches stay separate, reported from the earliest start.
 assert.deepEqual(runs([2,3,13,14]),['2:00–4:00','13:00–15:00']);
 // Every hour flagged: there is no start, so it must still terminate and say something.
 const all=runs(Array.from({length:24},(_,i)=>i));
 assert.ok(all.length>0&&all.length<=24);
});

test('activity is read against an even share, and a missing hour stays missing',()=>{
 assert.equal(relativeActivity(row(3,0.65)),0.65);
 assert.equal(relativeActivity({...row(3,1),mean:null}),null);
 const table=[row(3,0.65),row(5,0.69),row(13,1.35),row(10,1.0),{...row(1,1),mean:null}];
 const got=extremes(table);
 assert.deepEqual(got.quiet,[3,5]);
 assert.deepEqual(got.busy,[13]);          // 1.0 is neither; a null hour is never either
});

test('hours are ordered as the reader lives them, Eastern midnight first',()=>{
 const table=[row(13,1),row(0,1),row(23,1)];
 assert.deepEqual(byEasternHour(table).map(r=>r.et_hour),[0,13,23]);
 assert.deepEqual(table.map(r=>r.et_hour),[13,0,23]);   // and the input is not mutated
});
