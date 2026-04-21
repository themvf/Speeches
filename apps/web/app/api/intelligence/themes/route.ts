import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import {
  scoreThemeArticle,
  scoreThemeArticles,
  THEME_MAPPING,
  THEME_WEIGHTS,
  type RawThemeInput,
  type ThemeArticleInput,
  type ThemeContextInput
} from "@/lib/theme-intelligence";

export const runtime = "nodejs";

type ThemeRequestBody = {
  id?: string;
  title?: string;
  raw_themes?: RawThemeInput;
  context?: ThemeContextInput;
  articles?: ThemeArticleInput[];
};

function isThemeArticleInput(value: unknown): value is ThemeArticleInput {
  if (!value || typeof value !== "object") {
    return false;
  }
  const rawThemes = (value as { raw_themes?: unknown }).raw_themes;
  return typeof rawThemes === "string" || Array.isArray(rawThemes);
}

export async function GET() {
  const requestId = createRequestId();
  return ok(
    {
      mapping: THEME_MAPPING,
      weights: THEME_WEIGHTS,
      severity_thresholds: {
        CRITICAL: "score >= 25",
        HIGH: "score >= 15",
        NORMAL: "score < 15"
      }
    },
    requestId
  );
}

export async function POST(request: Request) {
  const requestId = createRequestId();
  let body: ThemeRequestBody;

  try {
    body = (await request.json()) as ThemeRequestBody;
  } catch {
    return fail("Invalid JSON body.", "INVALID_JSON", 400, requestId);
  }

  if (Array.isArray(body.articles)) {
    const articles = body.articles.filter(isThemeArticleInput);
    if (articles.length === 0) {
      return fail("Provide at least one article with raw_themes.", "NO_ARTICLES", 400, requestId);
    }
    return ok({ signals: scoreThemeArticles(articles) }, requestId);
  }

  if (typeof body.raw_themes === "string" || Array.isArray(body.raw_themes)) {
    return ok(
      {
        signal: scoreThemeArticle({
          id: body.id,
          title: body.title,
          raw_themes: body.raw_themes,
          context: body.context
        })
      },
      requestId
    );
  }

  return fail("Provide raw_themes or articles[].raw_themes.", "NO_THEMES", 400, requestId);
}
