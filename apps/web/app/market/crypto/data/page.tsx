"use client";
import {useRouter} from 'next/navigation';
import {paths} from '@/lib/crypto-workspace';
import {CryptoDataView} from '@/components/market/crypto-data-view';
import {PageHeader} from '@/components/market/crypto-shell';
export default function Page(){
 const router=useRouter();
 return <><PageHeader eyebrow="Data" title="Can I trust the data today?">Coverage per coin, credit ledgers and the registry. Other pages link here whenever coverage limits a comparison.</PageHeader>
  <CryptoDataView onCoin={c=>router.push(paths.coin(c))}/></>;
}
