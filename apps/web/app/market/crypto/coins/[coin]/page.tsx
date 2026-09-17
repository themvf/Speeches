"use client";
import {Suspense,use,useCallback} from 'react';
import type {Route} from 'next';
import {useRouter,useSearchParams} from 'next/navigation';
import {isCoin,coinConfig} from '@/lib/crypto-coins';
import {paths,readCoinQuery,writeCoinQuery,type CoinQuery} from '@/lib/crypto-workspace';
import {CryptoCoinView} from '@/components/market/crypto-coin-view';
import {PageHeader} from '@/components/market/crypto-shell';
function CoinPage({coin}:{coin:string}){
 const router=useRouter(),params=useSearchParams();const q=readCoinQuery(params);
 const update=useCallback((patch:Partial<CoinQuery>)=>{const next={...readCoinQuery(new URLSearchParams(window.location.search)),...patch};window.history.replaceState(window.history.state,'',paths.coin(coin)+writeCoinQuery(next));router.replace((paths.coin(coin)+writeCoinQuery(next)) as Route,{scroll:false});},[coin,router]);
 return <><PageHeader eyebrow="Coins" title="Investigate a move" crumbs={[{href:paths.signals,label:'Signals'},{label:coinConfig(coin).label}]}>Tap a day on the chart to see who posted in the 24 hours before it and what the pool did after. Posts and connections for the same coin follow below.</PageHeader>
  <CryptoCoinView coin={coin} query={q} onQuery={update} onAccount={id=>router.push(paths.account(id))} onData={()=>router.push(paths.data)}/></>;
}
export default function Page({params}:{params:Promise<{coin:string}>}){
 const {coin}=use(params);const symbol=decodeURIComponent(coin).toUpperCase();
 if(!isCoin(symbol))return <PageHeader eyebrow="Coins" title="Unknown coin" crumbs={[{href:paths.signals,label:'Signals'},{label:symbol}]}>This coin is not in the registry. Pick a tracked coin from the chips on any coin page, or search for a contract to add it.</PageHeader>;
 return <Suspense fallback={null}><CoinPage coin={symbol}/></Suspense>;
}
