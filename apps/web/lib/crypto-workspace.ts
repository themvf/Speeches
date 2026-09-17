// URL state for the research workspace: one place that reads and writes the query string.
import {isCoin} from './crypto-coins.ts';
export const VIEWS=['signals','people','coins','account','data'] as const;
export type View=typeof VIEWS[number];
export const COIN_TABS=['timeline','posts','sentiment','connections'] as const;
export type CoinTab=typeof COIN_TABS[number];
export type WorkspaceState={view:View;coin:string;tab:CoinTab;account:string|null;from:string|null;to:string|null;day:string|null;highlight:string|null};
const isDay=(v:string|null)=>!!v&&/^\d{4}-\d{2}-\d{2}$/.test(v);
export function readState(params:URLSearchParams,defaultCoin:string):WorkspaceState{
 const rawView=params.get('view');const legacy:Record<string,View>={overview:'signals',reach:'people',voices:'people',impact:'people',run:'coins',network:'coins',sentiment:'coins',watchers:'signals',posts:'coins'};
 const view=(VIEWS as readonly string[]).includes(rawView??'')?rawView as View:legacy[rawView??'']??(params.get('account')?'account':'signals');
 const legacyTab:Record<string,CoinTab>={network:'connections',sentiment:'sentiment',posts:'posts'};
 const rawTab=params.get('tab');const tab=(COIN_TABS as readonly string[]).includes(rawTab??'')?rawTab as CoinTab:legacyTab[rawView??'']??'timeline';
 const coin=isCoin(params.get('coin'))?params.get('coin')!:defaultCoin;
 return {view,coin,tab,account:params.get('account'),from:isDay(params.get('from'))?params.get('from'):null,to:isDay(params.get('to'))?params.get('to'):null,day:isDay(params.get('day'))?params.get('day'):null,highlight:params.get('highlight')};
}
export function writeState(s:WorkspaceState,defaults:{from:string;to:string}){
 const q=new URLSearchParams();q.set('view',s.view);
 if(s.view==='coins'||s.view==='account'||s.highlight)q.set('coin',s.coin);
 if(s.view==='coins'&&s.tab!=='timeline')q.set('tab',s.tab);
 if(s.account)q.set('account',s.account);
 if(s.from&&s.from!==defaults.from)q.set('from',s.from);if(s.to&&s.to!==defaults.to)q.set('to',s.to);
 if(s.day)q.set('day',s.day);if(s.highlight)q.set('highlight',s.highlight);
 return '?'+q.toString();
}
