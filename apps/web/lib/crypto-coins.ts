// Tracked-coin registry. The data lives in crypto-coins.json, shared with crypto_coins.py;
// tests/fixtures/crypto-coin-matches.json pins both implementations to the same answers.
import registry from './crypto-coins.json' with { type: 'json' };
export type CoinConfig={symbol:string;name:string;label:string;network:string;networkLabel:string;address:string|null;archiveStart:string;
 words:string[];contextWords:string[][];exclude:string[];discoveryName:string;profileTerms:string[];official:string[];searchQuery:string;identityNote:string;originFrom?:string};
export const COINS:CoinConfig[]=registry.coins as CoinConfig[];
export const COIN_SYMBOLS=COINS.map(c=>c.symbol);
const bySymbol=new Map(COINS.map(c=>[c.symbol,c]));
export function coinConfig(symbol:string):CoinConfig{const c=bySymbol.get(symbol);if(!c)throw new Error('Unknown coin '+symbol);return c;}
export function isCoin(symbol:string|null|undefined):symbol is string{return !!symbol&&bySymbol.has(symbol);}
const compiled=new Map<string,{words:RegExp[];context:[RegExp,RegExp][];exclude:RegExp[]}>();
function patterns(symbol:string){let p=compiled.get(symbol);if(!p){const c=coinConfig(symbol);p={words:c.words.map(w=>new RegExp(w,'i')),context:c.contextWords.map(([a,b])=>[new RegExp(a,'i'),new RegExp(b,'i')] as [RegExp,RegExp]),exclude:c.exclude.map(e=>new RegExp(e,'i'))};compiled.set(symbol,p);}return p;}
export function hasContract(text:string,symbol:string){const a=coinConfig(symbol).address;if(!a)return false;return a.startsWith('0x')?text.toLowerCase().includes(a):text.includes(a);}
// 'contract': exact address only. 'words': address, name/cashtag words, or context pairs, minus exclusions.
export function matchesCoin(text:string,symbol:string,mode:'contract'|'words'){
 if(hasContract(text,symbol))return true;
 if(mode==='contract')return false;
 const p=patterns(symbol);
 if(p.exclude.some(r=>r.test(text)))return false;
 return p.words.some(r=>r.test(text))||p.context.some(([a,b])=>a.test(text)&&b.test(text));
}
