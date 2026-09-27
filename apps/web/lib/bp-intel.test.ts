import test from 'node:test';
import assert from 'node:assert/strict';
import {CSV_COLUMNS,parseIntelQuery,rawToDecimal,share,toCsv,withAmounts} from './bp-intel.ts';

const wallet='BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy';
test('query parsing validates every parameter and requires scope for detail sections',()=>{
 assert.deepEqual(parseIntelQuery(new URLSearchParams('section=overlap')).query,
  {section:'overlap',cohort:null,run:null,wallet:null,mint:null,days:7,format:'json',ranking:'filtered'});
 const ok=parseIntelQuery(new URLSearchParams(`section=token&mint=native&cohort=12&days=30&format=csv&run=0f8c7a52-1d2e-4f3a-9b1c-2d3e4f5a6b7c`)).query;
 assert.equal(ok?.cohort,12);assert.equal(ok?.mint,'native');assert.equal(ok?.format,'csv');
 for(const bad of ['section=nope','section=roster&cohort=0','section=roster&cohort=1e3','section=wallet','section=token','section=activity&days=31',
  'section=activity&days=0','section=alerts&format=xlsx','section=overlap&run=abc',`section=wallet&wallet=${wallet}x`,'section=roster&ranking=top'])
  assert.ok(parseIntelQuery(new URLSearchParams(bad)).error,bad);
 assert.equal(parseIntelQuery(new URLSearchParams(`section=wallet&wallet=${wallet}`)).query?.wallet,wallet);
});

test('raw amounts format exactly beyond float precision and unknown stays unknown',()=>{
 assert.equal(rawToDecimal('1000000000000000007',6),'1000000000000.000007');
 assert.equal(rawToDecimal('5',9),'0.000000005');
 assert.equal(rawToDecimal('-2500',3),'-2.5');
 assert.equal(rawToDecimal('0',6),'0');
 assert.equal(rawToDecimal(null,6),null);
 assert.equal(rawToDecimal('1.5',6),null);
 assert.equal(rawToDecimal('10',null),null);
});

test('shares never divide by an unknown or zero cohort',()=>{
 assert.equal(share(3,200),0.015);
 assert.equal(share(3,0),null);assert.equal(share(3,null),null);assert.equal(share(null,200),null);
 assert.equal(share(0,200),0);
});

test('CSV escapes quotes, separators and spreadsheet formulas but keeps signed numbers',()=>{
 const csv=toCsv([{a:'x,"y"',b:'=HYPERLINK("http://evil")',c:'-12.5',d:null,e:{k:1}}],[['a','A'],['b','B'],['c','C'],['d','D'],['e','E']]);
 assert.equal(csv,'A,B,C,D,E\r\n"x,""y""","\'=HYPERLINK(""http://evil"")",-12.5,,"{""k"":1}"\r\n');
 assert.ok(toCsv([],CSV_COLUMNS.overlap).startsWith('Mint,Symbol,Name'));
});

test('display amounts are added without replacing stored raw strings',()=>{
 const [row]=withAmounts('wallet',[{raw_amount:'123456789',decimals:6}]);
 assert.equal(row.amount,'123.456789');assert.equal(row.raw_amount,'123456789');
 assert.equal(withAmounts('roster',[{raw_balance:'5000000000000',decimals:9}])[0].bp_balance,'5000');
});
