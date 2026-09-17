import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {COINS,COIN_SYMBOLS,coinConfig,isCoin,matchesCoin} from './crypto-coins.ts';
const fixtures=JSON.parse(fs.readFileSync(path.join(import.meta.dirname,'../../../tests/fixtures/crypto-coin-matches.json'),'utf8')) as [string,string,'contract'|'words',boolean][];
test('registry integrity: unique symbols, valid patterns, lowercase EVM addresses, ISO archive starts',()=>{
 assert.equal(new Set(COIN_SYMBOLS).size,COINS.length);
 for(const c of COINS){for(const w of [...c.words,...c.exclude,c.discoveryName,...c.contextWords.flat()])assert.doesNotThrow(()=>new RegExp(w,'i'));
  if(c.address?.startsWith('0x'))assert.equal(c.address,c.address.toLowerCase());
  assert.match(c.archiveStart,/^\d{4}-\d{2}-\d{2}$/);assert.ok(c.official.every(h=>h===h.toLowerCase()));assert.ok(c.searchQuery.length>0);}
 assert.ok(isCoin('ZCAT')&&!isCoin('BTC')&&!isCoin(null));assert.throws(()=>coinConfig('BTC'));
});
test('shared match fixtures agree with the Python port',()=>{for(const [text,coin,mode,expected] of fixtures)assert.equal(matchesCoin(text,coin,mode),expected,`${coin} ${mode}: ${text}`);});
