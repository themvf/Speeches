import {notFound} from 'next/navigation';
import {BackpackMonitor} from '@/components/backpack/monitor';
export default async function Page({params}:{params:Promise<{assetId:string}>}){
 const {assetId}=await params;
 if(!/^[1-9]\d{0,14}$/.test(assetId))notFound();
 return <BackpackMonitor assetId={assetId}/>;
}
