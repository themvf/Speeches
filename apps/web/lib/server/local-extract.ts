import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export interface LocalExtractPayload {
  connector: string;
  selection: string;
  limit: number;
  maxPages: number;
  baseUrl: string;
  includePdfs: boolean;
  includeRss: boolean;
}

export interface LocalExtractJobResult {
  job_id: string;
  provider: "local";
  status: "success";
  status_url: "";
  workflow: "local_extract";
  summary_path: string;
  summary: Record<string, unknown>;
  updated_at: string;
  conclusion: "success";
}

interface CommandResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

function nowIso(): string {
  return new Date().toISOString();
}

function findRepoRoot(startDir: string): string | null {
  let current = path.resolve(startDir);
  for (let i = 0; i < 8; i += 1) {
    const scriptPath = path.join(current, "run_connector_extraction_pipeline.py");
    if (fs.existsSync(scriptPath)) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  return null;
}

function runCommand(
  command: string,
  args: string[],
  cwd: string,
  timeoutMs: number
): Promise<CommandResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGKILL");
      } catch {
        // Ignore kill errors.
      }
    }, timeoutMs);

    child.stdout.on("data", (buf) => {
      stdout += String(buf || "");
    });
    child.stderr.on("data", (buf) => {
      stderr += String(buf || "");
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      if (settled) {
        return;
      }
      settled = true;
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (settled) {
        return;
      }
      settled = true;
      resolve({ code, stdout, stderr, timedOut });
    });
  });
}

function readJsonFile(filePath: string): Record<string, unknown> | null {
  try {
    if (!fs.existsSync(filePath)) {
      return null;
    }
    const raw = fs.readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export async function runLocalExtractJob(payload: LocalExtractPayload): Promise<LocalExtractJobResult> {
  const repoRoot = findRepoRoot(process.cwd());
  if (!repoRoot) {
    throw new Error("Local extraction is unavailable: cannot locate run_connector_extraction_pipeline.py.");
  }

  const scriptPath = path.join(repoRoot, "run_connector_extraction_pipeline.py");
  const dataDir = path.join(repoRoot, "data");
  fs.mkdirSync(dataDir, { recursive: true });

  const ts = Date.now();
  const summaryFile = path.join(dataDir, `local_extract_summary_${ts}.json`);
  const pythonBin = String(process.env.PYTHON_BIN || "python").trim() || "python";

  const args = [
    scriptPath,
    "--connector",
    String(payload.connector || "").trim(),
    "--selection",
    String(payload.selection || "new_or_updated").trim(),
    "--limit",
    String(Math.max(1, Number.parseInt(String(payload.limit || 25), 10) || 25)),
    "--max-pages",
    String(Math.max(1, Number.parseInt(String(payload.maxPages || 5), 10) || 5)),
    "--include-pdfs",
    payload.includePdfs ? "true" : "false",
    "--include-rss",
    payload.includeRss ? "true" : "false",
    "--require-remote-persistence",
    "--summary-path",
    summaryFile
  ];
  if (String(payload.baseUrl || "").trim()) {
    args.push("--base-url", String(payload.baseUrl).trim());
  }

  const command = await runCommand(pythonBin, args, repoRoot, 30 * 60 * 1000);
  const summary = readJsonFile(summaryFile);

  if (command.timedOut) {
    throw new Error("Local extraction timed out after 30 minutes.");
  }

  if (command.code !== 0) {
    const summaryError = typeof summary?.error === "string" ? summary.error : "";
    const stderr = command.stderr.trim();
    const stdout = command.stdout.trim();
    throw new Error(summaryError || stderr || stdout || `Local extraction failed with exit code ${String(command.code)}.`);
  }

  if (!summary) {
    throw new Error("Local extraction completed but no summary file was produced.");
  }

  const okValue = summary.ok;
  if (okValue === false) {
    const err = typeof summary.error === "string" ? summary.error : "Local extraction failed.";
    throw new Error(err);
  }

  return {
    job_id: `local_${ts}`,
    provider: "local",
    status: "success",
    status_url: "",
    workflow: "local_extract",
    summary_path: path.relative(repoRoot, summaryFile).replace(/\\/g, "/"),
    summary,
    updated_at: nowIso(),
    conclusion: "success"
  };
}

