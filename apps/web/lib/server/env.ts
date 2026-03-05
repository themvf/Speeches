export interface DataSourceConfig {
  mode: "auto" | "local" | "gcs";
  dataDirPath: string;
  gcsBucketName: string;
  gcsCredentialsJson: string;
  gcsCredentialsPath: string;
}

export interface GithubActionsConfig {
  enabled: boolean;
  token: string;
  owner: string;
  repo: string;
  ref: string;
  ingestWorkflow: string;
  enrichWorkflow: string;
  extractWorkflow: string;
}

function readEnv(name: string, fallback = ""): string {
  return String(process.env[name] ?? fallback).trim();
}

function parseMode(raw: string): DataSourceConfig["mode"] {
  const mode = raw.toLowerCase();
  if (mode === "local" || mode === "gcs" || mode === "auto") {
    return mode;
  }
  return "auto";
}

function parseBool(raw: string): boolean {
  return ["1", "true", "yes", "on"].includes(raw.toLowerCase());
}

export function getDataSourceConfig(): DataSourceConfig {
  return {
    mode: parseMode(readEnv("DATA_SOURCE_MODE", "auto")),
    dataDirPath: readEnv("DATA_DIR_PATH", ""),
    gcsBucketName: readEnv("GCS_BUCKET_NAME", ""),
    gcsCredentialsJson: readEnv("GCS_CREDENTIALS_JSON", ""),
    gcsCredentialsPath: readEnv("GCS_CREDENTIALS_PATH", "") || readEnv("GOOGLE_APPLICATION_CREDENTIALS", "")
  };
}

export function getGithubActionsConfig(): GithubActionsConfig {
  const token = readEnv("GITHUB_ACTIONS_TOKEN", "");
  const owner = readEnv("GITHUB_REPO_OWNER", "");
  const repo = readEnv("GITHUB_REPO_NAME", "");

  return {
    enabled:
      parseBool(readEnv("GITHUB_ACTIONS_ENABLED", "true")) &&
      Boolean(token) &&
      Boolean(owner) &&
      Boolean(repo),
    token,
    owner,
    repo,
    ref: readEnv("GITHUB_DEFAULT_REF", "main"),
    ingestWorkflow: readEnv("GITHUB_INGEST_WORKFLOW", "financial-news-ingest.yml"),
    enrichWorkflow: readEnv("GITHUB_ENRICH_WORKFLOW", "financial-news-enrich.yml"),
    extractWorkflow: readEnv("GITHUB_EXTRACT_WORKFLOW", "policy-extraction.yml")
  };
}

export function getApiRuntimeInfo() {
  const data = getDataSourceConfig();
  const jobs = getGithubActionsConfig();

  return {
    data_source_mode: data.mode,
    gcs_configured: Boolean(data.gcsBucketName),
    github_actions_enabled: jobs.enabled
  };
}
