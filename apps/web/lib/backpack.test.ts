import test from 'node:test';
import assert from 'node:assert/strict';
import {change,numeric,rollingIssuance,windowChange,dayOffset,sortRows,mechanicalChanges} from './backpack.ts';
test('unknown values are never zero; zero itself remains valid',()=>{
 assert.equal(numeric(null),null);assert.equal(numeric(''),null);assert.equal(numeric('0'),0);assert.equal(change(10,0),null);
});
test('first launch has no dates and must render without throwing',()=>{
 assert.equal(rollingIssuance([],'',30),null);assert.equal(windowChange([],'','reference_aum_usd',30),null);
 assert.ok(mechanicalChanges([],'').length);
});
test('rolling issuance requires every daily observation',()=>{
 const rows=Array.from({length:30},(_,i)=>({date:dayOffset('2026-09-22',-i),net_supply_change_usd:i===0?-100:100}));
 assert.equal(rollingIssuance(rows,'2026-09-22',30),2800);
 assert.equal(rollingIssuance(rows.slice(1),'2026-09-22',30),null);
});
test('changes use exact dates and consistent coverage, never index offsets',()=>{
 assert.equal(windowChange([{date:'2026-09-22',reference_aum_usd:200},{date:'2026-08-23',reference_aum_usd:100}],'2026-09-22','reference_aum_usd',30),100);
 assert.equal(windowChange([{date:'2026-09-22',reference_aum_usd:200},{date:'2026-08-22',reference_aum_usd:100}],'2026-09-22','reference_aum_usd',30),null);
 assert.equal(windowChange([{date:'2026-09-22',assets_expected:2,x:200},{date:'2026-08-23',assets_expected:1,x:100}],'2026-09-22','x',30),null);
});
test('sorting numerics and keeping missing last in either direction',()=>{
 for(const desc of [true,false])assert.equal(sortRows([{x:10},{x:null},{x:2}],'x',desc).at(-1)?.x,null);
 assert.deepEqual(sortRows([{x:'10'},{x:'2'}],'x',false).map(r=>r.x),['2','10']);
});

test('wallet attribution requires a real public key, evidence, and explicit review',async()=>{
 const {solanaAddress,labelError}=await import('./backpack-admin.ts');
 const body={wallet_address:'BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy',label:'Treasury',entity:'Example',confidence:'high',source:'https://example.test/evidence',notes:'Reviewed address',approved:true};
 assert.equal(labelError(body),null);
 assert.equal(solanaAddress('1'.repeat(33)),false);
 for(const update of [{approved:false},{source:'javascript:alert(1)'},{confidence:'guessed'},{notes:''},{wallet_address:'BP'}])assert.ok(labelError({...body,...update}));
});

test('admin origin validation preserves public Host across internal URL normalization',async()=>{
 const {sameOrigin}=await import('./backpack-admin.ts');
 const request=(origin:string,host='127.0.0.1:3107')=>new Request('http://localhost:3107/api/admin/backpack',{headers:{origin,host}});
 assert.equal(sameOrigin(request('http://127.0.0.1:3107')),true);
 for(const origin of ['https://other.test','null','http://127.0.0.1:3108','https://127.0.0.1:3107','http://127.0.0.1:3107/path'])assert.equal(sameOrigin(request(origin)),false);
});

import {validRevalidationToken,BACKPACK_CACHE_CONTROL} from './backpack-cache.ts';
test('cache invalidation requires the exact configured bearer credential',()=>{
 assert.equal(validRevalidationToken(null,undefined),false);
 assert.equal(validRevalidationToken('Bearer undefined',undefined),false);
 assert.equal(validRevalidationToken('Bearer wrong','test-secret'),false);
 assert.equal(validRevalidationToken('Bearer test-secret','test-secret'),true);
 assert.equal(BACKPACK_CACHE_CONTROL,'public, s-maxage=3600, stale-while-revalidate=86400');
});
