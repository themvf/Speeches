import test from 'node:test';
import assert from 'node:assert/strict';
import {readState,writeState} from './crypto-workspace.ts';
test('legacy view names map onto the five destinations and round-trip',()=>{
 const s=readState(new URLSearchParams('view=impact&coin=PONS&account=12'),'ZCAT');
 assert.equal(s.view,'people');assert.equal(s.coin,'PONS');assert.equal(s.account,'12');
 assert.equal(readState(new URLSearchParams('view=network&coin=ZEC'),'ZCAT').tab,'connections');
 assert.equal(readState(new URLSearchParams('account=5'),'ZCAT').view,'account');
 assert.equal(readState(new URLSearchParams('coin=BAD&from=nope'),'ZCAT').coin,'ZCAT');
 const url=writeState({view:'coins',coin:'ZEC',tab:'posts',account:null,from:'2026-08-01',to:'2026-09-01',day:'2026-08-20',highlight:'9'},{from:'2026-07-25',to:'2026-09-01'});
 assert.equal(url,'?view=coins&coin=ZEC&tab=posts&from=2026-08-01&day=2026-08-20&highlight=9');
 assert.equal(writeState({view:'signals',coin:'ZCAT',tab:'timeline',account:null,from:null,to:null,day:null,highlight:null},{from:'',to:''}),'?view=signals');
});
