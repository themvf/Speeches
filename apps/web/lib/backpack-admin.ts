export const walletLabels=['Backpack','Treasury','Custody','Market Maker','DEX','Liquidity Pool','Lending Protocol','Bridge','Known Exchange','Protocol','Unknown'];
export const confidences=['confirmed','high','medium','low'];
export function solanaAddress(value:unknown):value is string{
 if(typeof value!=='string'||!/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(value))return false;
 let n=0n;const alphabet='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
 for(const char of value)n=n*58n+BigInt(alphabet.indexOf(char));
 let bytes=value.match(/^1*/)?.[0].length??0;
 for(;n>0n;n>>=8n)bytes++;
 return bytes===32;
}
export function labelError(body:Record<string,unknown>):string|null{
 if(!solanaAddress(body.wallet_address))return 'Wallet must be a 32-byte Solana address';
 if(!walletLabels.includes(String(body.label))||!confidences.includes(String(body.confidence)))return 'Invalid label or confidence';
 for(const field of ['entity','source','notes'])if(typeof body[field]!=='string'||!String(body[field]).trim()||String(body[field]).length>2000)return 'Entity, evidence URL and evidence notes are required (maximum 2,000 characters each)';
 try{if(new URL(String(body.source)).protocol!=='https:')throw new Error();}catch{return 'Evidence must use HTTPS';}
 if(body.approved!==true)return 'Explicit evidence review is required';
 return null;
}

export function sameOrigin(req:Request):boolean{
 const origin=req.headers.get('origin');if(!origin||origin==='null')return false;
 try{
  const actual=new URL(origin),target=new URL(req.url);
  // Next may normalize its internal hostname. Host preserves the browser authority.
  const host=req.headers.get('host')??target.host;
  return actual.origin===origin&&actual.host===host&&actual.protocol===target.protocol;
 }catch{return false;}
}
