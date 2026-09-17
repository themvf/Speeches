import Link from 'next/link';
import type {Metadata} from 'next';
import {CryptoShell} from '@/components/market/crypto-shell';
import styles from '@/components/market/crypto-research.module.css';
export const metadata:Metadata={title:'Crypto research | Policy Research Hub',description:'Saved X posts, account track records and archived pool prices for tracked crypto coins.'};
export default function CryptoLayout({children}:{children:React.ReactNode}){
 return <div className={styles.researchPage}><Link className={styles.researchBack} href="/market">← Market</Link><CryptoShell>{children}</CryptoShell></div>;
}
