import test from 'node:test';
import assert from 'node:assert/strict';
import {computeRings,ringToken} from './crypto-rings.ts';
import type {RunPost} from './crypto-run.ts';
let n=0;
const post=(author:string,text:string,at:string,kind='original',edges:{target_id:string;target:string;kind:string}[]=[]):RunPost=>({id:'p'+(n++),author_id:author,handle:author,text,url:'',posted_at:at,kind,edges});
const reply=(author:string,to:string,text:string,at:string)=>post(author,`@${to} ${text}`,at,'reply',[{target_id:to,target:to,kind:'reply'}]);
const D='2026-09-0';
test('hub: a reply star plus a relayed line',()=>{
 const line='buying zcat at 700k is like buying zcash at 700k';
 const ps=[post('imagyn',line,D+'2T18:31:00Z'),post('space',line+' 🤫',D+'2T18:59:00Z','quote',[{target_id:'imagyn',target:'imagyn',kind:'quote'}]),post('eyl',line,D+'2T19:15:00Z','quote'),post('regrets',line,D+'2T20:35:00Z','quote'),
  reply('space','imagyn','we move silently',D+'2T18:58:00Z'),reply('regrets','imagyn','sleeping comfy',D+'2T15:15:00Z'),reply('eyl','imagyn','Z on zcat',D+'1T07:41:00Z'),reply('mirage','imagyn','zcat',D+'1T21:11:00Z'),reply('mirage','imagyn','the cat is in the bag',D+'1T22:00:00Z'),reply('mirage','imagyn','accumulated 2M more',D+'2T17:44:00Z'),reply('mirage','regrets','they will post',D+'2T01:40:00Z'),reply('mirage','regrets','six figures',D+'2T01:41:00Z'),reply('regrets','mirage','lol',D+'2T01:42:00Z'),reply('space','regrets','keep it lowkey',D+'2T22:04:00Z'),reply('eyl','space','so good',D+'2T17:07:00Z'),reply('regrets','eyl','gm',D+'2T13:13:00Z'),
  post('lurker','nice cat',D+'1T10:00:00Z'),reply('lurker','imagyn','what is zcat',D+'1T11:00:00Z')];
 const r=computeRings(ps,'ZCAT',['ZEC']);
 for(const h of ['imagyn','space','eyl','regrets','mirage'])assert.equal(r.get(h)?.ring,1,h);
 assert.notEqual(r.get('lurker')?.ring,1,'one reply is not a hub tie');
});
test('defender, witness, campaign and early caller',()=>{
 const ps=[reply('mirage','valueandtime','what makes it a vamp of redacted, be serious',D+'1T00:25:00Z'),reply('mirage','inversebrah','not vamping anything',D+'1T01:28:00Z'),
  post('slot','the fact I will have $100 of $zec in a few hours just from holding $zcat',D+'1T01:21:00Z'),
  ...[1,2,3].map(i=>post('space',`shhh the cat stays anonymous ${i} @blknoiz06`,D+`1T0${i}:00:00Z`,'reply',[{target_id:'blknoiz06',target:'blknoiz06',kind:'mention'}])),
  post('yeti','HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR the anon cat looking good',D+'1T00:01:00Z'),post('late','HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR still good',D+'3T23:00:00Z')];
 const r=computeRings(ps,'ZCAT',['ZEC']);
 assert.equal(r.get('mirage')?.ring,2);assert.equal(r.get('slot')?.ring,3);assert.equal(r.get('space')?.ring,4);assert.equal(r.get('yeti')?.ring,5);assert.equal(r.get('late')?.ring,null);
});
test('feeds, piggybackers and critics are separated and take precedence',()=>{
 const ps=[...[1,2,3,4,5].map(i=>post('whalewatchalert',`New Pair: a whale just bought $${i}K of $ZCAT`,D+`1T0${i}:00:00Z`)),
  post('elio','$ZCAT at 1.9M, $GZ under 30K, buy $GZ',D+'1T01:00:00Z'),post('elio','$GZ is the next $ZCAT',D+'1T02:00:00Z'),
  post('cov','$zcat is a bundled vamp coin, KOL bundles farming you',D+'1T02:00:00Z'),post('cov','stop buying, rugged',D+'1T03:00:00Z'),
  reply('cov','mitnick','nice vamp coin bundled to shit by kols',D+'1T04:00:00Z')];
 const r=computeRings(ps,'ZCAT',['ZEC']);
 assert.equal(r.get('whalewatchalert')?.ring,7);assert.equal(r.get('elio')?.ring,6);assert.equal(r.get('cov')?.ring,8,'critic outranks defender even when the reply says vamp');
});
test('ring token parses',()=>{assert.equal(ringToken('ring:2'),2);assert.equal(ringToken('ring9'),null);assert.equal(ringToken('rings'),null);});
