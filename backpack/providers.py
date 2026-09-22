"""Server-only bounded adapters. No credentials/URLs containing keys in exceptions."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
import time
import requests
from .metrics import USDC, number

UTC = timezone.utc


class SourceError(Exception): pass


class Providers:
    def __init__(self, env=None, http=None):
        self.env = os.environ if env is None else env
        self.http = http or requests.Session()
        self.usage = defaultdict(int)
        self.maximum = int(self.env.get('BACKPACK_MAX_REQUESTS', '500'))
        self.deadline = time.monotonic() + 1200

    def request(self, provider, method, url, **kwargs):
        for attempt in range(3):
            if sum(self.usage.values()) >= self.maximum or time.monotonic() >= self.deadline:
                raise SourceError('Run request/time budget exhausted')
            self.usage[provider] += 1
            try:
                response = self.http.request(method, url, timeout=20, **kwargs)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if not response.ok: raise SourceError(f'{provider}: HTTP {response.status_code}')
                return response.json()
            except (requests.RequestException, ValueError):
                if attempt == 2: raise SourceError(f'{provider}: unavailable or malformed response') from None
        raise SourceError(f'{provider}: retries exhausted')

    def rpc(self, method, params, independent=False):
        key = self.env.get('HELIUS_API_KEY')
        url = (self.env.get('SOLANA_VALIDATION_RPC_URL') or 'https://api.mainnet-beta.solana.com') if independent else (
            f'https://mainnet.helius-rpc.com/?api-key={key}' if key else self.env.get('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'))
        name = 'Solana validation RPC' if independent else 'Helius RPC' if key else 'Solana RPC'
        body = self.request(name, 'POST', url, json={'jsonrpc':'2.0','id':'backpack','method':method,'params':params})
        if 'error' in body or 'result' not in body: raise SourceError(f'{name}: {method} failed')
        return body['result']

    def supply(self, mint, independent=False):
        result = self.rpc('getTokenSupply', [mint, {'commitment':'finalized'}], independent)
        value = result['value']
        amount = number(value['amount'])
        if amount is None or amount < 0: raise SourceError('Invalid supply')
        return amount / Decimal(10) ** int(value['decimals']), int(value['decimals']), int(result['context']['slot'])

    def holders(self, mint):
        if not self.env.get('HELIUS_API_KEY'): raise SourceError('HELIUS_API_KEY required for complete holder pagination')
        accounts, slots, seen = {}, [], set()
        cursor = None
        for _ in range(int(self.env.get('BACKPACK_MAX_HOLDER_PAGES', '100'))):
            params = {'mint':mint, 'limit':1000, 'options':{'showZeroBalance':False}}
            if cursor: params['cursor'] = cursor
            result = self.rpc('getTokenAccounts', params)
            rows = result.get('token_accounts')
            if not isinstance(rows, list) or result.get('last_indexed_slot') is None: raise SourceError('Invalid holder page')
            slots.append(int(result['last_indexed_slot']))
            for row in rows: accounts[row['address']] = row
            next_cursor = result.get('cursor')
            if not rows or not next_cursor:
                return list(accounts.values()), min(slots), max(slots)
            if next_cursor in seen: raise SourceError('Holder pagination repeated cursor')
            seen.add(next_cursor)
            cursor = next_cursor
        raise SourceError('Holder page budget exhausted; incomplete enumeration withheld')

    def price(self, mint):
        key = self.env.get('JUPITER_API_KEY')
        if not key: raise SourceError('JUPITER_API_KEY not configured')
        result = self.request('Jupiter', 'GET', 'https://api.jup.ag/price/v3', params={'ids':mint}, headers={'x-api-key':key}).get(mint)
        if not result or number(result.get('usdPrice')) is None: raise SourceError('Jupiter price unavailable')
        stamp = self.rpc('getBlockTime', [result['blockId']])
        if stamp is None: raise SourceError('Jupiter price block timestamp unavailable')
        return number(result['usdPrice']), datetime.fromtimestamp(stamp, UTC)

    def equity(self, symbol):
        # Existing project uses Yahoo, but it is not an official exchange feed. Use an
        # explicitly configured licensed SIP feed for this research surface.
        key, secret = self.env.get('ALPACA_API_KEY'), self.env.get('ALPACA_SECRET_KEY')
        if not key or not secret: raise SourceError('Licensed Alpaca SIP feed not configured')
        result = self.request('Alpaca SIP', 'GET', 'https://data.alpaca.markets/v2/stocks/snapshots',
            params={'symbols':symbol, 'feed':'sip'}, headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret})
        data = result.get(symbol) or {}
        trade = data.get('latestTrade') or {}
        price, stamp = number(trade.get('p')), trade.get('t')
        if price is None or price <= 0 or not stamp: raise SourceError('Equity reference unavailable')
        return price, datetime.fromisoformat(stamp.replace('Z','+00:00')), number((data.get('prevDailyBar') or {}).get('c'))

    def quote(self, mint, decimals, price, notional, direction):
        key = self.env.get('JUPITER_API_KEY')
        if not key: raise SourceError('JUPITER_API_KEY not configured')
        if direction == 'sell' and (price is None or price <= 0): raise SourceError('Sell notional requires a current token price')
        raw = int(Decimal(notional) * 10**6) if direction == 'buy' else int(Decimal(notional) / price * Decimal(10)**decimals)
        result = self.request('Jupiter', 'GET', 'https://api.jup.ag/swap/v1/quote',
            params={'inputMint': USDC if direction == 'buy' else mint, 'outputMint': mint if direction == 'buy' else USDC,
                    'amount':str(raw), 'slippageBps':50, 'swapMode':'ExactIn'}, headers={'x-api-key':key})
        if not result.get('routePlan') or number(result.get('outAmount')) is None or number(result['outAmount']) <= 0:
            raise SourceError('No executable route returned')
        return result

    def history(self, address, since, until, last_signature=None):
        key = self.env.get('HELIUS_API_KEY')
        if not key: raise SourceError('HELIUS_API_KEY not configured')
        rows, before, seen = [], None, set()
        newest = None
        for _ in range(int(self.env.get('BACKPACK_MAX_TX_PAGES_PER_WALLET','3'))):
            params = {'api-key':key, 'limit':100}
            if before: params['before'] = before
            if last_signature: params['until'] = last_signature
            page = self.request('Helius Enhanced', 'GET', f'https://api-mainnet.helius-rpc.com/v0/addresses/{address}/transactions', params=params)
            if not isinstance(page, list): raise SourceError('Invalid transaction page')
            if not page: return rows, newest, True
            if newest is None:
                newest = next((tx['signature'] for tx in page if tx.get('timestamp') is not None and tx['timestamp'] < until), None)
            for tx in page:
                stamp = tx.get('timestamp')
                if stamp is not None and since <= stamp < until: rows.append(tx)
            if min(x.get('timestamp', until) for x in page) < since: return rows, newest, True
            before = page[-1]['signature']
            if before in seen: raise SourceError('Repeated history cursor')
            seen.add(before)
        return rows, newest, False

    def validation_market(self, mint):
        data = self.request('DexScreener validation', 'GET', f'https://api.dexscreener.com/token-pairs/v1/solana/{mint}')
        if not isinstance(data, list): raise SourceError('Invalid independent market response')
        # One highest-liquidity exact-base pair. Never sum multi-venue routed volume.
        pairs = [p for p in data if (p.get('baseToken') or {}).get('address') == mint and number(p.get('priceUsd')) is not None]
        if not pairs: raise SourceError('No independent market price')
        return max(pairs, key=lambda p:number((p.get('liquidity') or {}).get('usd')) or 0)
