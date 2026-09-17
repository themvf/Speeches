import test from 'node:test';
import assert from 'node:assert/strict';
import {legacyPath,paths,readCoinQuery,writeCoinQuery} from './crypto-workspace.ts';
test('legacy query links map onto real paths',()=>{
 assert.equal(legacyPath(new URLSearchParams('view=impact&coin=PONS'),'ZCAT'),paths.people);
 assert.equal(legacyPath(new URLSearchParams('view=run&coin=ZEC&from=2026-08-01&day=2026-08-20'),'ZCAT'),'/market/crypto/coins/ZEC?from=2026-08-01&day=2026-08-20');
 assert.equal(legacyPath(new URLSearchParams('account=5'),'ZCAT'),'/market/crypto/accounts/5');
 assert.equal(legacyPath(new URLSearchParams('view=watchers'),'ZCAT'),paths.signals);
 assert.equal(legacyPath(new URLSearchParams('coin=BAD'),'ZCAT'),null);
 assert.equal(legacyPath(new URLSearchParams(''),'ZCAT'),null);
});
test('coin query round-trips and drops invalid dates',()=>{
 const q=readCoinQuery(new URLSearchParams('from=2026-08-01&to=nope&highlight=9'));
 assert.deepEqual(q,{from:'2026-08-01',to:null,day:null,highlight:'9'});
 assert.equal(writeCoinQuery(q),'?from=2026-08-01&highlight=9');assert.equal(writeCoinQuery({from:null,to:null,day:null,highlight:null}),'');
});
