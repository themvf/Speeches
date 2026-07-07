import {
  ensureSchema,
  getFeeds,
  getTopicRules,
  markFeedRefreshed,
  upsertFeedSource,
  upsertRssArticles,
  type RssFeed,
} from "@/lib/server/neon";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";
import { filterRssArticlesForIngestion } from "@/lib/server/rss-ingestion-filter";
import {
  fetchXTimelineBatch,
  isXTimelineFeedKey,
  parseXTimelineAccounts,
  xTimelineFeedKey,
  xTimelineFeedLabel,
  xTimelineFeedUrl,
  xTimelineUsernameFromFeed,
} from "@/lib/server/x-syndication";

export type XTimelineRefreshFeedResult = {
  feedKey: string;
  label: string;
  fetched: number;
  matched: number;
  filtered: number;
  inserted: number;
  provider?: string;
  error?: string;
};

export type XTimelineRefreshResult = {
  inserted: number;
  feeds: XTimelineRefreshFeedResult[];
  analysis_feed_keys: string[];
  analysis: {
    selected_count: number;
    saved_count: number;
    failed_count: number;
    failed: Array<{ article_id: number; title: string; error: string }>;
  };
};

function activeXFeeds(feeds: RssFeed[], onlyActive: boolean): RssFeed[] {
  return feeds.filter((feed) => isXTimelineFeedKey(feed.feed_key) && (!onlyActive || feed.active));
}

export async function listXTimelineFeeds(onlyActive = false, opts: { dueOnly?: boolean } = {}): Promise<RssFeed[]> {
  const feeds = onlyActive && opts.dueOnly ? await getFeeds(true, { dueOnly: true }) : await getFeeds(false);
  return activeXFeeds(feeds, onlyActive);
}

export async function configuredXTimelineAccounts(onlyActive = true, opts: { dueOnly?: boolean } = {}): Promise<string[]> {
  const feeds = await listXTimelineFeeds(onlyActive, opts);
  return parseXTimelineAccounts(feeds.map((feed) => xTimelineUsernameFromFeed(feed)).join(","));
}

export async function refreshXTimelines(opts: {
  accounts?: string[];
  limit?: number;
  analysisLimit?: number;
  refreshIntervalMinutes?: number;
  dueOnly?: boolean;
} = {}): Promise<XTimelineRefreshResult> {
  await ensureSchema();
  const explicitAccounts = parseXTimelineAccounts((opts.accounts || []).join(","));
  const accounts = explicitAccounts.length ? explicitAccounts : await configuredXTimelineAccounts(true, { dueOnly: opts.dueOnly });
  const limit = Math.max(1, Math.min(50, Math.round(Number(opts.limit || 20))));
  const analysisLimit = Math.max(0, Math.min(50, Math.round(Number(opts.analysisLimit || 0))));
  const refreshIntervalMinutes = Math.max(1, Math.round(Number(opts.refreshIntervalMinutes || 180)));
  const topicRules = await getTopicRules(true);
  const results = accounts.length ? await fetchXTimelineBatch(accounts, { limit }) : [];

  let totalInserted = 0;
  const feedKeys: string[] = [];
  const feeds: XTimelineRefreshFeedResult[] = [];

  for (const result of results) {
    const feedKey = xTimelineFeedKey(result.username);
    const label = xTimelineFeedLabel(result.username);
    feedKeys.push(feedKey);
    await upsertFeedSource(feedKey, label, xTimelineFeedUrl(result.username), refreshIntervalMinutes);
    const filtered = filterRssArticlesForIngestion(feedKey, result.articles, topicRules);
    const inserted = filtered.articles.length > 0 ? await upsertRssArticles(filtered.articles, feedKey) : 0;
    await markFeedRefreshed(feedKey).catch((error) => {
      console.error(`[x-timeline-ingestion] failed to mark ${feedKey} refreshed:`, error);
    });
    totalInserted += inserted;
    feeds.push({
      feedKey,
      label,
      fetched: result.fetched,
      matched: filtered.matched,
      filtered: filtered.filtered,
      inserted,
      provider: result.provider,
      error: result.error,
    });
  }

  const analysis = analysisLimit > 0 && feedKeys.length > 0
    ? await analyzeMissingRssArticles(analysisLimit, { feedKeys })
    : { selected_count: 0, saved_count: 0, failed_count: 0, failed: [] };

  return {
    inserted: totalInserted,
    feeds,
    analysis_feed_keys: feedKeys,
    analysis,
  };
}
