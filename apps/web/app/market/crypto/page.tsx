import Link from 'next/link';
import {CryptoSocialPanel} from '@/components/market/crypto-social-panel';
import styles from '@/components/market/crypto-research.module.css';
export const metadata={title:'Crypto research | Policy Research Hub'};
export default function CryptoResearchPage(){return <div className={styles.researchPage}><Link className={styles.researchBack} href="/market">← Market</Link><CryptoSocialPanel/></div>;}
