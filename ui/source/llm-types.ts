import { type CalibrateCmd } from "./shared.js";

// ─── Step Types ───────────────────────────────────────────────
export type LlmStep =
  | "init"
  | "config-path"
  | "provider"
  | "enter-model"
  | "model-confirm"
  | "agent-mode"
  | "agent-model-entry"
  | "agent-model-confirm"
  | "agent-verify"
  | "output-dir"
  | "output-dir-confirm"
  | "api-keys"
  | "running"
  | "leaderboard";

// ─── Interfaces ───────────────────────────────────────────────
export interface ModelState {
  status: "waiting" | "running" | "done" | "error";
  logs: string[];
  metrics?: { passed?: number; failed?: number; total?: number };
}

export interface HistoryMessage {
  role: string;
  content: string;
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface TestResult {
  id: string;
  history: HistoryMessage[];
  evaluationType: string;
  evaluationCriteria: string;
  actualOutput: string;
  passed: boolean;
  reasoning: string;
}

export interface LlmConfig {
  configPath: string;
  models: string[];
  provider: string;
  outputDir: string;
  overwrite: boolean;
  envVars: Record<string, string>;
  calibrate: CalibrateCmd;
  agentUrl: string;
  agentHeaders: Record<string, string>;
  agentBenchmark: boolean;
  agentModels: string[];
}

// ─── Constants ────────────────────────────────────────────────
export const MAX_PARALLEL_MODELS = 2;

export const OPENAI_MODEL_EXAMPLES = [
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o1-mini",
  "o3-mini",
];

export const OPENROUTER_MODEL_EXAMPLES = [
  "openai/gpt-4.1",
  "anthropic/claude-sonnet-4",
  "google/gemini-2.0-flash-001",
];
