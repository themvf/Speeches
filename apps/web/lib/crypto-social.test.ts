import test from 'node:test';
import assert from 'node:assert/strict';
import { rankAccounts, growthLabel, type SocialAccount } from './crypto-social.ts';
const account=(id:string,extras:Partial<SocialAccount>={})=>({id,participants:0,amplifiers:0,posts:0,connections:0,
 growth_7d:null,participants_7d:0,participants_previous_7d:0,...extras}) as SocialAccount;
test('targets with no posts rank on attention; input is preserved',()=>{
 const rows=[account('1',{posts:100}),account('2',{participants:5})];
 assert.equal(rankAccounts(rows,'attention')[0].id,'2');assert.equal(rows[0].id,'1');
});
test('missing growth is not zero and emerging needs two independent observations',()=>{
 const rows=[account('1'),account('2',{growth_7d:10}),account('3',{growth_7d:5,participants_7d:3,participants_previous_7d:1})];
 assert.deepEqual(rankAccounts(rows,'growth').map(a=>a.id),['2','3']);
 assert.deepEqual(rankAccounts(rows,'emerging').map(a=>a.id),['3']);
 assert.equal(growthLabel(null),'—');assert.equal(growthLabel(0),'0');
});
test('bio board uses confirmed bio matches only',()=>{
 assert.deepEqual(rankAccounts([account('1'),account('2')],'bios',new Set(['2'])).map(a=>a.id),['2']);
});
