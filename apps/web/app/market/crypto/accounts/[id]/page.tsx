"use client";
import {use} from 'react';
import type {Route} from 'next';
import {useRouter} from 'next/navigation';
import {paths,writeCoinQuery} from '@/lib/crypto-workspace';
import {CryptoAccountView} from '@/components/market/crypto-account-view';
export default function Page({params}:{params:Promise<{id:string}>}){
 const {id}=use(params);const router=useRouter();
 return <CryptoAccountView accountId={decodeURIComponent(id)} onCoin={c=>router.push(paths.coin(c))} onMark={(coin,account)=>router.push((paths.coin(coin)+writeCoinQuery({from:null,to:null,day:null,highlight:account})) as Route)}/>;
}
