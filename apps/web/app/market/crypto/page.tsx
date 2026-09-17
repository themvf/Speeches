import Link from 'next/link';
import {Suspense} from 'react';
import type {Metadata} from 'next';
import {CryptoSocialPanel} from '@/components/market/crypto-social-panel';
import styles from '@/components/market/crypto-research.module.css';
export const metadata:Metadata={title:'Crypto research | Policy Research Hub',description:'Saved X posts, account roles and archived pool prices for tracked crypto coins.'};
export default function CryptoResearchPage(){return <div className={styles.researchPage}><Link className={styles.researchBack} href="/market">← Market</Link><Suspense fallback={<p className={styles.muted}>Loading the research workspace…</p>}><CryptoSocialPanel/></Suspense></div>;}
