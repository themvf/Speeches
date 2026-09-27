/** BP holder intelligence: request parsing, exact raw-unit formatting and CSV export. Pure; shared by route and UI. */
export const INTEL_SECTIONS=['overview','roster','overlap','activity','alerts','wallet','token'] as const;
export type IntelSection=typeof INTEL_SECTIONS[number];
export type IntelRow=Record<string,unknown>;
export type IntelQuery={section:IntelSection;cohort:number|null;run:string|null;wallet:string|null;mint:string|null;days:number;format:'json'|'csv';ranking:'filtered'|'raw'};
export type IntelMeta={
 versions:IntelRow[];version:IntelRow|null;runs:IntelRow[];run:IntelRow|null;latestRun:IntelRow|null;
 coverage:IntelRow|null;meaningfulUsd:number;live:{enabled:boolean;detail:string};
};
export type IntelPayload={status:'ready'|'schema_pending'|'not_configured'|'no_cohort'|'not_found';section:IntelSection;meta:IntelMeta|null;rows:IntelRow[];extra:Record<string,IntelRow[]>};
export const LIVE_STATUS={enabled:false,detail:'Live ingestion is not enabled (a Milestone 4 decision). Freshness comes from the hourly portfolio refresh and history polling; incoming transfers into newly created token accounts are found by polling, never by a webhook.'};

const address=/^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
export function parseIntelQuery(params:URLSearchParams):{query?:IntelQuery;error?:string}{
 const section=params.get('section')??'overview';
 if(!(INTEL_SECTIONS as readonly string[]).includes(section))return {error:'Unknown section'};
 const cohort=params.get('cohort'),run=params.get('run'),wallet=params.get('wallet'),mint=params.get('mint'),days=params.get('days')??'7',format=params.get('format')??'json',ranking=params.get('ranking')??'filtered';
 if(cohort!==null&&!/^[1-9]\d{0,14}$/.test(cohort))return {error:'Invalid cohort version'};
 if(run!==null&&!uuid.test(run))return {error:'Invalid run'};
 if(wallet!==null&&!address.test(wallet))return {error:'Invalid wallet address'};
 if(mint!==null&&mint!=='native'&&!address.test(mint))return {error:'Invalid mint'};
 if(!/^\d{1,2}$/.test(days)||Number(days)<1||Number(days)>30)return {error:'days must be 1-30'};
 if(format!=='json'&&format!=='csv')return {error:'format must be json or csv'};
 if(ranking!=='filtered'&&ranking!=='raw')return {error:'ranking must be filtered or raw'};
 if(section==='wallet'&&!wallet)return {error:'wallet is required'};
 if(section==='token'&&!mint)return {error:'mint is required'};
 return {query:{section:section as IntelSection,cohort:cohort?Number(cohort):null,run,wallet,mint,days:Number(days),format,ranking}};
}

/** Exact decimal string from a raw integer amount; never passes through floating point. */
export function rawToDecimal(raw:unknown,decimals:unknown):string|null{
 if(raw===null||raw===undefined||decimals===null||decimals===undefined||decimals==='')return null;
 const text=String(raw),places=Number(decimals);
 if(!/^-?\d+$/.test(text)||!Number.isInteger(places)||places<0||places>30)return null;
 const negative=text.startsWith('-'),value=BigInt(negative?text.slice(1):text),divisor=10n**BigInt(places);
 const fraction=places?(value%divisor).toString().padStart(places,'0').replace(/0+$/,''):'';
 return `${negative?'-':''}${value/divisor}${fraction?`.${fraction}`:''}`;
}

/** Holder share of the cohort; null when the denominator is unknown, never a zero. */
export function share(part:unknown,whole:unknown):number|null{
 const a=Number(part),b=Number(whole);
 return part==null||whole==null||!Number.isFinite(a)||!Number.isFinite(b)||b<=0?null:a/b;
}

const cell=(value:unknown):string=>{
 if(value===null||value===undefined)return '';
 let text=typeof value==='object'?JSON.stringify(value):String(value);
 // Spreadsheet formula guard for text cells; plain signed numbers stay numeric.
 if(/^[=+\-@\t\r]/.test(text)&&!/^-?\d+(\.\d+)?$/.test(text))text=`'${text}`;
 return /[",\r\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text;
};
export function toCsv(rows:IntelRow[],columns:readonly (readonly [string,string])[]):string{
 return [columns.map(([,label])=>cell(label)).join(','),...rows.map(row=>columns.map(([key])=>cell(row[key])).join(','))].join('\r\n')+'\r\n';
}

export const CSV_COLUMNS:Record<Exclude<IntelSection,'overview'>|'raw_ranking',readonly (readonly [string,string])[]>={
 raw_ranking:[['rank','Raw rank'],['wallet_address','Wallet'],['bp_balance','BP balance'],['raw_balance','BP raw balance'],['label','Label at capture'],['label_confidence','Label confidence'],['label_source','Label evidence'],['excluded','Excluded from filtered ranking']],
 roster:[['rank','Filtered rank'],['wallet_address','Wallet'],['member','Member'],['event','Membership event'],['previous_rank','Previous rank'],['bp_balance','BP balance'],['raw_balance','BP raw balance'],['label','Label at capture'],['label_confidence','Label confidence'],['label_source','Label evidence'],['below_exit_runs','Runs below exit rank'],['exit_reason','Exit reason']],
 overlap:[['mint','Mint'],['symbol','Symbol'],['name','Name'],['asset_class','Class'],['holders','Holders (positive balance)'],['meaningful_holders','Meaningful holders'],['cohort_size','Cohort size'],['wallets_read','Wallets read'],['combined_value_usd','Combined value USD (priced)'],['median_weight','Median weight of priced portfolio'],['largest_owner_share','Largest owner share'],['unpriced_holders','Unpriced holders'],['partial_visibility','Partial-visibility holders'],['new_buyers_1h','New-position buyers 1h'],['new_buyers_24h','New-position buyers 24h'],['new_buyers_7d','New-position buyers 7d'],['purchased_raw_7d','Purchased raw 7d'],['sold_raw_7d','Sold raw 7d'],['net_raw_7d','Net raw 7d'],['purchased_usd_7d','Purchased USD 7d (priced)'],['sold_usd_7d','Sold USD 7d (priced)'],['decimals','Decimals'],['price','Price USD'],['price_at','Price time'],['stale_price','Stale price'],['liquidity_usd','Liquidity USD'],['spam_class','Spam'],['spam_reason','Spam reason']],
 activity:[['block_time','Block time'],['signature','Signature'],['wallet_address','Wallet'],['event_index','Event index'],['kind','Kind'],['tier','Tier'],['input_mint','Input mint'],['input_symbol','Input'],['input_raw','Input raw'],['input_decimals','Input decimals'],['output_mint','Output mint'],['output_symbol','Output'],['output_raw','Output raw'],['output_decimals','Output decimals'],['usd_value','USD (execution-time estimate)'],['valuation_source','Valuation source'],['new_position','New position'],['first_observed_purchase','First observed purchase'],['re_entry','Re-entry'],['pre_membership','Pre-membership'],['pre_balance_source','Pre-balance source'],['venue','Venue'],['finality','Finality'],['detail','Detail']],
 alerts:[['window_start','Window start'],['window_end','Window end'],['rule','Rule'],['mint','Mint'],['symbol','Symbol'],['wallet_count','Wallets'],['inferred_wallets','Inferred buyers'],['lowest_tier','Lowest tier'],['quantity_raw','Quantity raw'],['decimals','Decimals'],['usd_value','USD'],['valuation_status','Valuation'],['finality','Finality'],['data_through','Data through'],['signatures','Signatures'],['detail','Detail']],
 wallet:[['mint','Mint'],['symbol','Symbol'],['name','Name'],['holding_class','Class'],['amount','Amount'],['raw_amount','Raw amount'],['decimals','Decimals'],['ui_amount','Display amount'],['price','Price USD'],['pricing_status','Pricing'],['value_usd','Value USD'],['balance_visibility','Visibility'],['frozen','Frozen'],['dust','Dust'],['spam_class','Spam'],['classification_reason','Reason'],['observed_at','Observed'],['slot','Slot']],
 token:[['rank','Cohort rank'],['wallet_address','Wallet'],['amount','Amount'],['raw_amount','Raw amount'],['decimals','Decimals'],['value_usd','Value USD'],['pricing_status','Pricing'],['balance_visibility','Visibility'],['observed_at','Observed']],
};

/** Adds exact decimal amounts for CSV/table display without mutating stored raw strings. */
export function withAmounts(section:IntelSection,rows:IntelRow[]):IntelRow[]{
 if(section==='roster')return rows.map(r=>({...r,bp_balance:rawToDecimal(r.raw_balance,r.decimals)}));
 if(section==='wallet'||section==='token')return rows.map(r=>({...r,amount:rawToDecimal(r.raw_amount,r.decimals)}));
 return rows;
}
