"""Optional authenticated cache notification; no response bodies or secrets in logs."""
import os
import requests


def notify(env=None):
    env=os.environ if env is None else env
    secret=env.get('BACKPACK_REVALIDATE_SECRET')
    if not secret:
        return {'cache_revalidation':'Unavailable','detail':'Shared revalidation secret not configured; one-hour cache fallback remains active'}
    # Fixed production destination prevents credential forwarding to arbitrary hosts.
    try:
        response=requests.post('https://speeches-zeta.vercel.app/api/market/crypto/backpack/revalidate',
            headers={'Authorization':'Bearer '+secret},timeout=20,allow_redirects=False)
        return {'cache_revalidation':'Verified' if response.status_code==200 else 'Unavailable',
                'detail':'Data-cache notification accepted; existing CDN responses retain their TTL' if response.status_code==200 else 'Cache notification rejected'}
    except requests.RequestException:
        return {'cache_revalidation':'Unavailable','detail':'Cache notification failed; TTL fallback remains active'}


if __name__=='__main__':
    import json
    print(json.dumps(notify()))
