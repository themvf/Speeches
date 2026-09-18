"use client";
import {Suspense} from 'react';
import {CryptoWorkbench} from '@/components/market/crypto-workbench';
export default function Page(){return <Suspense fallback={null}><CryptoWorkbench/></Suspense>;}
