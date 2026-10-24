// One-call orchestration: session + injected tools + planner → runDeployAgent
// → deployment package. Non-React so it can run headless (tests, CLI).

import type { Model } from "@modelvisio/parsers";
import { runDeployAgent, type AgentRun, type AgentStep, type LlmFn } from "@modelvisio/ai";
import { HW, type DeviceSpec } from "../data/hardware";
import { agentContext, createDeploySession, createDeployTools, type DeploySession } from "./tools";
import { createDeployPlanner } from "./planner";
import { buildDeployPackage, type DeployPackage } from "./package";

export type RunDeploymentArgs = {
  model: Model;
  goal: string;
  /** null → deterministic planner only. */
  llm: LlmFn | null;
  maxSteps?: number;
  signal?: AbortSignal;
  onStep?: (s: AgentStep) => void;
  onStatus?: (s: string) => void;
  devices?: DeviceSpec[];
  /** Reason to show when llm is null (e.g. "No Gemini key set"). */
  noAiReason?: string;
};

export async function runDeployment(a: RunDeploymentArgs): Promise<{ run: AgentRun; pkg: DeployPackage; session: DeploySession }> {
  const session = createDeploySession(a.model, a.devices ?? HW);
  const run = await runDeployAgent({
    goal: a.goal,
    tools: createDeployTools(session),
    llm: a.llm,
    planner: createDeployPlanner(session),
    maxSteps: a.maxSteps,
    context: agentContext(a.model),
    signal: a.signal,
    onStep: a.onStep,
    onStatus: a.onStatus,
  });
  if (!a.llm && a.noAiReason) run.fallbackReason = a.noAiReason;
  a.onStatus?.("Building the package…");
  return { run, pkg: buildDeployPackage({ goal: a.goal, run, session }), session };
}
