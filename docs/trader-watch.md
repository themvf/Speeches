# Crypto Trader Watch

The `/market/crypto?tab=traders` research page now includes a source-linked Trader Watch tab. Profiles are defined in `apps/web/lib/trader-watch.ts`; the first entry is `lbexplorer`. More traders can be added through the same registry without making another page or endpoint.

## Current data path

- Public posts are read from the existing stored X timeline feed for each configured handle. The watch endpoint does not fetch X or start an ingestion run. Add the handle under Admin → X Accounts and run the existing refresh operation to populate posts. Schedule `/api/intel/x-refresh` separately if an ongoing cadence is desired; it requires the existing cron bearer secret and an X provider or syndication access.
- Solana activity is read from `getSignaturesForAddress` and `getTransaction` through `SOLANA_RPC_URL` when configured, otherwise Solana's public mainnet RPC. The page requests the latest eight returned signatures and owner-attributed SPL token balance changes. Public RPC rate limits can make transaction details unavailable; a dedicated RPC endpoint is appropriate for reliable operation.
- Current holdings use `getTokenAccountsByOwner` for both SPL Token programs and `getBalance` for native SOL. Positive balances are combined by mint and displayed with exact decimal quantities. DEX Screener can supply a readable name and symbol for a mint with a trading pair; those labels are marked as external metadata and the mint address remains visible. A failed program read makes the result partial or unavailable rather than implying that a token is absent.
- The holdings list is a near-current snapshot for this one address. It does not reconstruct closed positions, historical holdings, exchange balances, stake accounts, or other wallets.
- The LB wallet address is a candidate only. Set `attribution: "verified"` and add an `evidenceUrl` only after finding a direct public link from the trader to the address. The display keeps the X record and wallet record separate while unverified.

## Interpretation rules

Posts are attributed claims. A token balance increase or decrease is an observation, not a buy, sale, position size, or realized profit. Transfers, deposits, multi-wallet activity, off-chain trades, and missing RPC metadata can change the interpretation. Failed transactions and missing detail are labeled separately. The endpoint returns an explicit unavailable state for either source and never turns an unavailable source into a zero-activity claim.

## Next increment

For an hourly research watch, persist a cursor per profile and poll the existing X collector plus wallet signatures on a scheduled job. Store source IDs, publication/block times, observation times, and attribution status. Add an event review step before assigning entry, addition, reduction, exit, or transfer labels. An alert should link to the exact post or transaction and state which classification was human-confirmed. Price-at-publication comparisons need a timestamped token-specific market source and venue; they should not reuse the top-coin table's current price.
