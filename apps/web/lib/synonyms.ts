const SYNONYMS: Record<string, string[]> = {
  crypto: ["cryptocurrency", "blockchain", "digital assets", "bitcoin", "ethereum", "defi", "virtual currency", "digital securities", "crypto assets", "tokens"],
  cryptocurrency: ["crypto", "blockchain", "digital assets", "bitcoin", "virtual currency"],
  blockchain: ["crypto", "cryptocurrency", "distributed ledger", "digital assets", "smart contracts", "decentralized"],
  "digital assets": ["crypto", "cryptocurrency", "blockchain", "virtual currency", "tokens", "digital currency"],
  "digital securities": ["crypto", "digital assets", "tokens", "security token", "blockchain"],
  bitcoin: ["btc", "crypto", "cryptocurrency", "digital currency", "virtual currency"],
  ethereum: ["eth", "crypto", "blockchain", "defi", "smart contracts"],
  defi: ["decentralized finance", "blockchain", "digital assets", "crypto", "smart contracts"],
  stablecoin: ["digital currency", "cbdc", "digital dollar", "tether", "usdc", "pegged currency"],
  nft: ["non-fungible token", "digital art", "blockchain", "digital collectibles", "tokenized"],
  "insider trading": ["material nonpublic information", "mnpi", "tipping", "10b-5", "insider information", "misappropriation"],
  mnpi: ["material nonpublic information", "insider trading", "tipping", "insider information"],
  fraud: ["misrepresentation", "false statements", "scheme to defraud", "market manipulation", "dishonest conduct", "deceptive"],
  "market manipulation": ["fraud", "pump and dump", "spoofing", "layering", "wash trading", "manipulative"],
  spoofing: ["layering", "market manipulation", "wash trading", "fraudulent orders"],
  "pump and dump": ["market manipulation", "fraud", "microcap", "promotional scheme"],
  esg: ["environmental social governance", "sustainability", "climate disclosure", "green finance", "responsible investing"],
  "climate disclosure": ["esg", "environmental", "sustainability reporting", "climate risk", "carbon"],
  spac: ["special purpose acquisition", "blank check company", "de-spac", "merger"],
  aml: ["anti-money laundering", "bsa", "bank secrecy act", "suspicious activity", "money laundering"],
  "anti-money laundering": ["aml", "bsa", "bank secrecy act", "suspicious activity report", "sar"],
  "money laundering": ["aml", "anti-money laundering", "bsa", "suspicious activity"],
  cybersecurity: ["data breach", "ransomware", "cyber incident", "information security", "hacking", "cyber attack"],
  "data breach": ["cybersecurity", "ransomware", "cyber incident", "information security"],
  "private fund": ["hedge fund", "private equity", "venture capital", "family office", "limited partnership"],
  "hedge fund": ["private fund", "private equity", "investment fund", "limited partnership"],
  "investment adviser": ["ria", "investment advisor", "registered investment adviser", "advisory"],
  fintech: ["financial technology", "robo-adviser", "digital platform", "algorithmic trading", "automated"],
  "payment for order flow": ["pfof", "retail trading", "order routing", "best execution", "payment"],
  "short selling": ["short sale", "naked short", "fails to deliver", "locate requirement"],
  meme: ["social media", "retail investors", "gamestop", "reddit", "wsb", "wallstreetbets"],
  "retail investor": ["individual investor", "public investor", "gamestop", "meme stock"],
  suitability: ["best interest", "reg bi", "fiduciary", "recommendation", "customer protection"],
  "best interest": ["suitability", "reg bi", "fiduciary", "recommendation"],
  "reg bi": ["regulation best interest", "suitability", "best interest", "broker-dealer"],
  custody: ["safekeeping", "client assets", "segregation", "custodian", "safeguarding"],
  "valuation": ["fair value", "pricing", "net asset value", "nav", "mark to market"],
  disclosure: ["transparency", "reporting", "prospectus", "registration", "material information"],
  whistleblower: ["tipster", "informant", "bounty", "reward", "reporting"],
  sanctions: ["embargo", "ofac", "russia", "iran", "north korea", "blocked persons"],
};

export function expandQuery(q: string): string {
  const lower = q.trim().toLowerCase();
  if (!lower) return lower;
  const terms = new Set<string>([lower]);
  for (const [key, synonymList] of Object.entries(SYNONYMS)) {
    const keyMatch = lower.includes(key);
    const synMatch = synonymList.some((s) => lower.includes(s));
    if (keyMatch || synMatch) {
      terms.add(key);
      for (const s of synonymList) terms.add(s);
    }
  }
  return [...terms].join(" ");
}

export function synonymsFor(q: string): string[] {
  const lower = q.trim().toLowerCase();
  const extras = new Set<string>();
  for (const [key, synonymList] of Object.entries(SYNONYMS)) {
    if (lower.includes(key) || synonymList.some((s) => lower.includes(s))) {
      extras.add(key);
      for (const s of synonymList) extras.add(s);
    }
  }
  extras.delete(lower);
  return [...extras];
}
