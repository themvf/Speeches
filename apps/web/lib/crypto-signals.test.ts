import test from 'node:test';
import assert from 'node:assert/strict';
import {boardRows,evaluateSignals,type BoardCoin,type SignalPost} from './crypto-signals.ts';
const coin=(o:Partial<BoardCoin>):BoardCoin=>({symbol:'ZCAT',name:'Anonymous Cat',price_now:1.1,price_24h_ago:1,volume_24h:2000,volume_prev_24h:900,posts_24h:40,posts_prev_24h:20,authors_24h:25,watched_posting_24h:2,linked_posts:10,hourly_hours:400,...o});
const post=(o:Partial<SignalPost>):SignalPost=>({id:'1',author_id:'a',handle:'alice',text:'$ZCAT',posted_at:'2026-09-17T10:00:00Z',kind:'original',url:'u',coins:['ZCAT'],contract:false,followers:null,...o});
test('board sorts by volume ratio and computes changes without inventing values',()=>{
 const rows=boardRows([coin({symbol:'A',volume_24h:100,volume_prev_24h:100}),coin({symbol:'B',volume_24h:null,volume_prev_24h:null,price_now:null,price_24h_ago:null,posts_prev_24h:0}),coin({symbol:'C'})]);
 assert.deepEqual(rows.map(r=>r.symbol),['C','A','B']);assert.equal(rows[2].price_change,null);assert.equal(rows[2].volume_ratio,null);assert.equal(rows[2].posts_change,null);
});
test('each rule fires only on its own condition and names it',()=>{
 const watched=[{id:'a',handle:'alice',reason:'early on 2 coins'}];const start='2026-09-17T00:00:00Z';
 const s=evaluateSignals({coins:[coin({}),coin({symbol:'PONS',price_now:1.01,price_24h_ago:1,volume_24h:100,volume_prev_24h:100,posts_24h:25})],posts:[post({contract:true}),post({id:'2',text:'whale bought $50K of $ZCAT'}),post({id:'3',author_id:'z',contract:true}),post({id:'4',kind:'repost',contract:true}),post({id:'5',text:'whale bought $60K of $ZCAT',posted_at:'2026-09-17T11:00:00Z'}),post({id:'6',text:'whale bought $70K',coins:[]})],watched,firstContract:[{coin:'KNOTS',post_id:'9',handle:'scan',account_id:'s',posted_at:'2026-09-17T05:00:00Z',url:'u'},{coin:'ZCAT',post_id:'8',handle:'old',account_id:'o',posted_at:'2026-08-01T00:00:00Z',url:'u'}],windowStart:start,topAuthorShare:[{coin:'PONS',share:.7}]});
 assert.deepEqual(s.map(x=>x.rule).sort(),['attention_without_price','first_contract_post','volume_surge','watched_contract_post','watched_trade_report']);
 assert.equal(s.find(x=>x.rule==='first_contract_post')?.coin,'KNOTS');assert.equal(s.find(x=>x.rule==='attention_without_price')?.coin,'PONS');
 assert.ok(s.every(x=>x.title&&x.detail));
});
test('high-severity signals come first and alert bursts collapse to one per six hours',()=>{
 const watched=[{id:'a',handle:'alice',reason:'watcher'}];const start='2026-09-17T00:00:00Z';
 const s=evaluateSignals({coins:[coin({posts_24h:5})],posts:[post({id:'1',text:'whale bought $1K of $ZCAT',posted_at:'2026-09-17T01:00:00Z'}),post({id:'2',text:'whale bought $2K of $ZCAT',posted_at:'2026-09-17T02:00:00Z'}),post({id:'3',text:'whale bought $3K of $ZCAT',posted_at:'2026-09-17T09:00:00Z'}),post({id:'4',contract:true,posted_at:'2026-09-17T00:30:00Z'})],watched,firstContract:[],windowStart:start,topAuthorShare:[]});
 assert.deepEqual(s.map(x=>x.rule),['watched_contract_post','watched_trade_report','watched_trade_report']);
});
