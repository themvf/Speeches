import Link from 'next/link';
import type {Metadata} from 'next';
import styles from '@/components/market/crypto-workbench.module.css';
export const metadata:Metadata={title:'Crypto workbench | Policy Research Hub',description:'Saved X posts, account track records and archived pool prices for tracked crypto coins on one screen.'};
export default function CryptoLayout({children}:{children:React.ReactNode}){
 return <div className={styles.page}><Link className={styles.back} href="/market">← Market</Link><Link className={styles.back} style={{marginLeft:20}} href="/market/crypto/backpack">Backpack thesis monitor →</Link>{children}</div>;
}
