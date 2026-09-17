"use client";
import {useRouter} from 'next/navigation';
import {paths} from '@/lib/crypto-workspace';
import {CryptoPeopleView} from '@/components/market/crypto-people-view';
import {PageHeader} from '@/components/market/crypto-shell';
export default function Page(){
 const router=useRouter();
 return <><PageHeader eyebrow="People" title="Who has a track record?">One list. Every account with saved evidence, ranked by how early they were and what happened after they posted, across coins.</PageHeader>
  <CryptoPeopleView initialCoin={null} onAccount={id=>router.push(paths.account(id))}/></>;
}
