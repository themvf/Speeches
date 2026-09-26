# Crypto Wallet Watch

The `/market/crypto?tab=traders` page follows configured Solana wallet addresses. The first entry is the wallet shown on Pump.fun under the `lbexplorer` profile. The display name is only a profile label; the page does not claim who controls the wallet.

Profiles are configured in `apps/web/lib/trader-watch.ts`. Add another address there to follow another wallet. The page reloads while open every two minutes and has a manual reload button. It does not send background notifications.

## Data

- Current balances use `getTokenAccountsByOwner` for SPL Token and Token-2022 plus `getBalance` for native SOL. Positive balances are combined by mint. DEX Screener supplies a name and symbol when it has a pair for a mint; the mint address remains visible because market labels are not identity verification.
- Recent activity uses `getSignaturesForAddress` and `getTransaction` for the latest eight returned signatures. Token balance changes are observations, not automatic buy, sell, or profit labels.
- The RPC endpoint is `SOLANA_RPC_URL` when configured, otherwise Solana's public mainnet RPC. A failed balance read produces a partial or unavailable state rather than implying that a token is absent.
- The holdings list is a near-current snapshot for one address. It does not reconstruct closed positions, historical holdings, exchange balances, stake accounts, or other wallets.

To follow the wallet when the page is closed, add a scheduled signature cursor and alert destination. Persist source signatures and observed times, and make retries idempotent. Keep transfers separate from trades until transaction evidence supports a trade label.
