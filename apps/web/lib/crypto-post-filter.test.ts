import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultPostFilters as defaults,filterSavedPosts,searchTerms} from './crypto-post-filter.ts';
const post=(id:string,fields={})=>({id,author_id:id,handle:'Alice',text:'ZCAT early accumulation',url:'https://x.com/i/status/'+id,posted_at:'2026-09-01T12:00:00Z',kind:'original',edges:[],...fields});
test('phrases AND included terms, OR exclusions, case-insensitive account search',()=>{
 assert.deepEqual(searchTerms('ZCAT "early accumulation"'),['zcat','early accumulation']);
 const posts=[post('1'),post('2',{text:'ZCAT early buying'}),post('3',{text:'ZCAT early accumulation join my telegram'})];
 assert.deepEqual(filterSavedPosts(posts,{...defaults,query:'zcat "early accumulation"',exclude:'"join my telegram" airdrop',author:'@ALICE'}).map(p=>p.id),['1']);
});
test('inclusive UTC dates and thresholds reject unknown metrics without treating them as zero',()=>{
 const posts=[post('1',{likes:0,followers:100000}),post('2',{likes:100,followers:100000}),post('3',{likes:null,followers:null}),post('4',{likes:100,followers:100000,posted_at:'2026-09-02T00:00:00Z'})];
 assert.equal(filterSavedPosts(posts,{...defaults}).length,4);
 assert.deepEqual(filterSavedPosts(posts,{...defaults,from:'2026-09-01',to:'2026-09-01',followers:100000,likes:1}).map(p=>p.id),['2']);
});
test('deduplicates cross-search posts, sorts known metrics first, then picks one post per author',()=>{
 const posts=[post('1',{author_id:'a',likes:5}),post('2',{author_id:'a',likes:100}),post('3',{likes:null}),post('4',{likes:0})];
 const result=filterSavedPosts([...posts,posts[1]],{...defaults,sort:'likes',onePerAuthor:true});
 assert.deepEqual(result.map(p=>p.id),['2','4','3']);
 assert.equal(posts.length,4);
});
test('post kinds, repost thresholds and oldest sorting combine',()=>{
 const posts=[post('1',{kind:'reply',reposts:20}),post('2',{reposts:20}),post('3',{reposts:10,posted_at:'2026-08-01T00:00:00Z'})];
 assert.deepEqual(filterSavedPosts(posts,{...defaults,kind:'original',reposts:10,sort:'oldest'}).map(p=>p.id),['3','2']);
});
