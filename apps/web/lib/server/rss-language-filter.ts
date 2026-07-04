type LanguageFilterArticle = {
  title?: string | null;
  description?: string | null;
  author?: string | null;
};

const ENGLISH_ONLY_FEED_PREFIXES = ["prnewswire_", "google_news_"];

const ENGLISH_MARKERS = [
  "the", "and", "of", "to", "for", "with", "from", "by", "on", "in", "as", "at",
  "announces", "launches", "reports", "joins", "appoints", "releases", "expands",
  "foundation", "company", "holdings", "million", "billion", "dollar", "new", "us", "u s", "ai",
];

const NON_ENGLISH_MARKERS: Record<string, string[]> = {
  spanish: [
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", "que", "con", "por", "para",
    "se", "su", "sus", "y", "en", "como", "anuncia", "lanza", "millones", "dolares", "resumen", "une",
  ],
  german: [
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "und", "oder", "mit", "von",
    "fur", "uber", "unter", "darunter", "mehr", "als", "im", "am", "zu", "meldet", "millionen", "zusammenfassung",
  ],
  french: [
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "avec", "pour", "sur", "et", "que", "qui",
    "annonce", "lance", "millions", "resume", "rejoint",
  ],
  portuguese: [
    "o", "a", "os", "as", "um", "uma", "dos", "das", "do", "da", "de", "em", "com", "para", "por", "e",
    "que", "anuncia", "lanca", "milhoes", "resumo", "junta",
  ],
  italian: [
    "il", "lo", "la", "gli", "le", "un", "una", "del", "della", "dei", "delle", "con", "per", "e", "che",
    "annuncia", "lancia", "milioni", "riepilogo", "unisce",
  ],
};

function normalizeForLanguage(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function markerCount(normalizedText: string, markers: string[]): number {
  const haystack = ` ${normalizedText} `;
  return markers.reduce((count, marker) => haystack.includes(` ${marker} `) ? count + 1 : count, 0);
}

function nonAsciiRatio(value: string): number {
  const chars = Array.from(value).filter((char) => !/\s/.test(char));
  if (chars.length === 0) return 0;
  const nonAscii = chars.filter((char) => char.charCodeAt(0) > 127).length;
  return nonAscii / chars.length;
}

export function shouldEnglishOnlyFilterFeed(feedKey: string): boolean {
  const key = String(feedKey || "").trim().toLowerCase();
  return ENGLISH_ONLY_FEED_PREFIXES.some((prefix) => key.startsWith(prefix));
}

export function isEnglishRssArticle(article: LanguageFilterArticle): boolean {
  const rawText = [article.title, article.description, article.author]
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join(" ");
  if (!rawText) return false;

  if (/[\u0400-\u04ff\u0370-\u03ff\u0590-\u05ff\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff]/u.test(rawText)) {
    return false;
  }

  const normalized = normalizeForLanguage(rawText);
  if (!normalized) return false;

  const englishScore = markerCount(normalized, ENGLISH_MARKERS);
  const foreignScores = Object.values(NON_ENGLISH_MARKERS).map((markers) => markerCount(normalized, markers));
  const strongestForeignScore = Math.max(0, ...foreignScores);
  const hasForeignPunctuation = /[¡¿]/.test(rawText);
  const accentRatio = nonAsciiRatio(rawText);

  if (hasForeignPunctuation && strongestForeignScore >= 1) return false;
  if (strongestForeignScore >= 4 && strongestForeignScore > englishScore) return false;
  if (strongestForeignScore >= 3 && englishScore <= 1) return false;
  if (accentRatio > 0.06 && strongestForeignScore >= 2 && strongestForeignScore >= englishScore) return false;

  return true;
}
