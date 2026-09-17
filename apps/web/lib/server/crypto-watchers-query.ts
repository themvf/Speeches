// Shared by the read-only route, the saved-corpus review job and the Python snapshot builder (via the JSON).
import queries from './crypto-ranking-queries.json' with { type: 'json' };
import {COIN_SYMBOLS} from '../crypto-coins.ts';
const quoted=COIN_SYMBOLS.map(s=>`'${s}'`).join(',');
export const WATCHER_QUERY=queries.watchers.replace('__COINS__',quoted);
export const VOICES_QUERY=(limit:number)=>queries.voices.replace('__COIN__','$1').replace('__LIMIT__',String(limit));
export const VOICES_UNFINISHED_QUERY=queries.voices_unfinished.replace('__COIN__','$1');
