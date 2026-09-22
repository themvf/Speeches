import {createHash,timingSafeEqual} from 'node:crypto';
export const BACKPACK_CACHE_CONTROL='public, s-maxage=3600, stale-while-revalidate=86400';
export function validRevalidationToken(header:string|null,secret:string|undefined):boolean{
 return Boolean(secret)&&timingSafeEqual(createHash('sha256').update(header??'').digest(),createHash('sha256').update(`Bearer ${secret}`).digest());
}
