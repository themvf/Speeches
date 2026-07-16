export interface RssArticleIdentity {
  feedKey: string;
  guid: string;
}

export function rssArticleIdentity(feedKey: string, guid: string): RssArticleIdentity {
  return { feedKey: feedKey.trim(), guid: guid.trim() };
}

export function isSameStoredRssArticle(left: RssArticleIdentity, right: RssArticleIdentity): boolean {
  return left.feedKey === right.feedKey && left.guid === right.guid;
}
