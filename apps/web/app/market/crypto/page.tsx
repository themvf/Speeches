"use client";
import {Suspense,useEffect} from 'react';
import {useRouter,useSearchParams} from 'next/navigation';
import {COINS} from '@/lib/crypto-coins';
import {legacyPath,paths} from '@/lib/crypto-workspace';
import {CryptoSignalsView} from '@/components/market/crypto-signals-view';
import {PageHeader} from '@/components/market/crypto-shell';
function Signals(){
 const router=useRouter(),params=useSearchParams();
 useEffect(()=>{const target=legacyPath(params,COINS[0].symbol);if(target)router.replace(target);},[params,router]);
 return <><PageHeader eyebrow="Signals" title="What moved, and did attention lead it?">Saved X posts against archived pool prices for every tracked coin. Association only, never attribution.</PageHeader>
  <CryptoSignalsView onCoin={c=>router.push(paths.coin(c))} onAccount={id=>router.push(paths.account(id))} onPeople={()=>router.push(paths.people)} onData={()=>router.push(paths.data)}/></>;
}
export default function Page(){return <Suspense fallback={null}><Signals/></Suspense>;}
