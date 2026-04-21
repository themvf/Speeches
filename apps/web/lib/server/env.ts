export interface DataSourceConfig {
  mode: "auto" | "local" | "gcs";
  dataDirPath: string;
  gcsBucketName: string;
  gcsCredentialsJson: string;
  gcsCredentialsPath: string;
}
export type JobExecutionMode = "github_actions" | "local";

export interface OpenAiConfig {
  apiKey: string;
  model: string;
  baseUrl: string;
}

export interface GithubActionsConfig {
  enabled: boolean;
  enabledFlag: boolean;
  missingRequiredEnv: string[];
  token: string;
  owner: string;
  repo: string;
  ref: string;
  ingestWorkflow: string;
  enrichWorkflow: string;
  extractWorkflow: string;
}

export interface Neo4jConfig {
  configured: boolean;
  missingRequiredEnv: string[];
  url: string;
  username: string;
  password: string;
  database: string;
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

function parseJobExecutionMode(raw: string): JobExecutionMode {
  return raw.toLowerCase() === "local" ? "local" : "github_actions";
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
  const enabledFlag = parseBool(readEnv("GITHUB_ACTIONS_ENABLED", "true"));
  const token = readEnv("GITHUB_ACTIONS_TOKEN", "");
  const owner = readEnv("GITHUB_REPO_OWNER", "");
  const repo = readEnv("GITHUB_REPO_NAME", "");
  const missingRequiredEnv: string[] = [];
  if (!token) {
    missingRequiredEnv.push("GITHUB_ACTIONS_TOKEN");
  }
  if (!owner) {
    missingRequiredEnv.push("GITHUB_REPO_OWNER");
  }
  if (!repo) {
    missingRequiredEnv.push("GITHUB_REPO_NAME");
  }

  return {
    enabled: enabledFlag && missingRequiredEnv.length === 0,
    enabledFlag,
    missingRequiredEnv,
    token,
    owner,
    repo,
    ref: readEnv("GITHUB_DEFAULT_REF", "main"),
    ingestWorkflow: readEnv("GITHUB_INGEST_WORKFLOW", "financial-news-ingest.yml"),
    enrichWorkflow: readEnv("GITHUB_ENRICH_WORKFLOW", "financial-news-enrich.yml"),
    extractWorkflow: readEnv("GITHUB_EXTRACT_WORKFLOW", "policy-extraction.yml")
  };
}

export function getJobExecutionMode(): JobExecutionMode {
  return parseJobExecutionMode(readEnv("JOB_EXECUTION_MODE", "github_actions"));
}

export function getApiRuntimeInfo() {
  const data = getDataSourceConfig();
  const jobs = getGithubActionsConfig();
  const mode = getJobExecutionMode();

  return {
    job_execution_mode: mode,
    data_source_mode: data.mode,
    gcs_configured: Boolean(data.gcsBucketName),
    github_actions_enabled: jobs.enabled,
    github_actions_enabled_flag: jobs.enabledFlag,
    github_actions_missing_required_env: jobs.missingRequiredEnv
  };
}

export function getOpenAiConfig(): OpenAiConfig {
  return {
    apiKey: readEnv("OPENAI_API_KEY", ""),
    model: readEnv("OPENAI_CHAT_MODEL", "gpt-5.1"),
    baseUrl: readEnv("OPENAI_BASE_URL", "https://api.openai.com/v1")
  };
}

export function getNeo4jConfig(): Neo4jConfig {
  const url = readEnv("NEO4J_URL") || readEnv("NEO4J_URI");
  const username = readEnv("NEO4J_USERNAME") || readEnv("NEO4J_USER");
  const password = readEnv("NEO4J_PASSWORD");
  const missingRequiredEnv: string[] = [];

  if (!url) {
    missingRequiredEnv.push("NEO4J_URL");
  }
  if (!username) {
    missingRequiredEnv.push("NEO4J_USERNAME");
  }
  if (!password) {
    missingRequiredEnv.push("NEO4J_PASSWORD");
  }

  return {
    configured: missingRequiredEnv.length === 0,
    missingRequiredEnv,
    url,
    username,
    password,
    database: readEnv("NEO4J_DATABASE", "neo4j")
  };
}
