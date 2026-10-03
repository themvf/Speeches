import test from 'node:test';
import assert from 'node:assert/strict';
import {coinActivity,copyPasteGroups,genuinePosts,socialDays,templateKey,type SocialPost} from './bp-social.ts';

const MINT='BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy';
const post=(id:string,text:string,at:string,author=id):SocialPost=>({id,text,posted_at:at,author_id:author,handle:`h${author}`,url:`https://x.com/h/status/${id}`});
const TEMPLATE=(n:number)=>`🚀 Backpack $BP is gaining attention on Solana $SOL. Take a look at this crypto airdrop and explore the token. CA=${MINT} #${n}`;

test('only posts about the token count, once each', () => {
 const posts=[post('1','Backpack $BP looks undervalued','2026-09-11T03:00:00Z'),post('1','Backpack $BP looks undervalued','2026-09-11T03:00:00Z'),
  post('2','F Gear 34L School Backpack at a discount','2026-09-11T04:00:00Z'),post('3','$BP earnings beat, oil majors rally','2026-09-11T05:00:00Z'),
  post('4',`CA ${MINT}`,'2026-09-11T06:00:00Z')];
 assert.deepEqual(genuinePosts(posts).map(p=>p.id),['1','4']);
});

test('copy-paste campaigns need several different accounts posting the same template', () => {
 assert.equal(templateKey(TEMPLATE(1)),templateKey(TEMPLATE(2)),'links, numbers and the contract do not split a template');
 const three=[1,2,3].map(i=>post(String(i),TEMPLATE(i),`2026-09-11T0${i}:00:00Z`));
 const [group]=copyPasteGroups(three);
 assert.equal(group.accounts,3);assert.equal(group.first,'2026-09-11T01:00:00.000Z');assert.equal(group.url,'https://x.com/h/status/1');
 assert.deepEqual(copyPasteGroups([...three.slice(0,2),post('9',TEMPLATE(9),'2026-09-11T09:00:00Z','1')]),[],'two accounts are not a campaign');
 assert.equal(templateKey('gm $BP'),'','short posts never form a template');
});

test('daily rows separate not searched from observed and none', () => {
 const now=new Date('2026-09-12T15:00:00Z');
 const rows=socialDays({days:3,now,
  posts:[...[1,2,3].map(i=>post(String(i),TEMPLATE(i),'2026-09-11T0'+i+':00:00Z')),post('7','Backpack $BP looks undervalued','2026-09-11T08:00:00Z')],
  windows:[{day:'2026-09-12',windows:2,done:2,searched:2},{day:'2026-09-11',windows:4,done:1,searched:1}],
  holders:[{day:'2026-09-12',unique_holders:'30500',holders_over_100:'5100',holders_complete:true},
   {day:'2026-09-11',unique_holders:'30400',holders_over_100:'5000',holders_complete:true},
   {day:'2026-09-10',unique_holders:'30000',holders_over_100:null,holders_complete:false}],
  cohorts:[{day:'2026-09-12',version_id:3,entered:5,left_count:2}],
  trades:[{day:'2026-09-11',bp_buyers:4,bp_sellers:1}],tradeCoverage:[{day:'2026-09-11',wallets:171},{day:'2026-09-10',wallets:149}],members:222});
 const [today,yesterday,before]=rows;
 assert.deepEqual([today.day,yesterday.day,before.day],['2026-09-12','2026-09-11','2026-09-10']);
 assert.equal(today.x_coverage,'complete');assert.equal(today.posts,0,'searched to the end and none found');
 assert.equal(yesterday.x_coverage,'partial');assert.equal(yesterday.posts,4);assert.equal(yesterday.authors,4);assert.equal(yesterday.copy_paste_posts,3);
 assert.equal(before.x_coverage,'not_searched');assert.equal(before.posts,null,'not searched is never zero');
 assert.equal(today.holders_change,100);assert.equal(yesterday.holders_change,null,'no change against an incomplete capture');
 assert.equal(before.holders,null);
 assert.equal(today.cohort_entered,5);assert.equal(yesterday.cohort_entered,null);
 assert.equal(today.bp_buyers,null,'no wallet history covers the day: not observed');
 assert.equal(yesterday.bp_buyers,4);assert.equal(yesterday.trade_wallets,171);assert.equal(yesterday.cohort_members,222);
 assert.equal(before.bp_sellers,0,'covered by some wallets: observed and none among them');assert.equal(before.trade_wallets,149);
});

test('activity on other coins counts only posts that match each coin', () => {
 const out=coinActivity([{...post('1','$STONK sending','2026-09-11T01:00:00Z'),coin:'STONK'},{...post('1','$STONK sending','2026-09-11T01:00:00Z'),coin:'STONK'},
  {...post('2','stonks only go up','2026-09-11T02:00:00Z'),coin:'STONK'}]);
 assert.deepEqual(out.get('STONK'),{posts:1,accounts:1});
});

test('a day reached only by an earlier query is searched, not unsearched', () => {
 const rows=socialDays({days:2,now:new Date('2026-09-12T15:00:00Z'),posts:[],windows:[],earlier:[{day:'2026-09-11'}],holders:[],cohorts:[],trades:[],tradeCoverage:[],members:null});
 assert.deepEqual(rows.map(r=>[r.day,r.x_coverage,r.posts]),[['2026-09-12','not_searched',null],['2026-09-11','earlier_query',0]]);
});
