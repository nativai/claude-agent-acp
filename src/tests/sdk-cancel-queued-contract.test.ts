// brick 21b73df0 — the SDK contract a person's Stop relies on, pinned against a REAL Claude Code.
//
// `cancel()` withdraws the prompts parked behind the stopped turn with `query.interrupt({ cancelQueued: true })`.
// That option, and the receipt's `cancelled` list, exist at runtime (SDK 0.3.322 / Claude Code 2.1.322, capability
// `interrupt_cancel_queued_v1`) but NOT in the SDK's typings, so a bump could drop them silently and the adapter would
// degrade to stopping the withdrawn message's turn only after it starts. This row goes RED if that happens.
//
// Needs real credentials: RUN_INTEGRATION_TESTS=1 CLAUDE_CONFIG_DIR=<a logged-in config dir>.

import { describe, it, expect } from "vitest";
import { query, type SDKMessage, type SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "crypto";
import { Pushable } from "../utils.js";
import { claudeCliPath } from "../acp-agent.js";

type InterruptReceipt = { still_queued?: string[]; cancelled?: string[] } | undefined;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitUntil(check: () => boolean, ms: number, what: string): Promise<void> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (check()) return;
    await sleep(50);
  }
  throw new Error(`precondition timed out: ${what}`);
}

function userMessage(text: string, uuid: string): SDKUserMessage {
  return {
    type: "user",
    message: { role: "user", content: [{ type: "text", text }] },
    parent_tool_use_id: null,
    uuid,
    session_id: "",
  } as SDKUserMessage;
}

// `chain3` (brick 21b73df0, the template-fork shape): A (the fork notice) runs a short tool; B arrives during A, is
// folded into A's turn at the tool boundary and runs a long command; C is queued behind it; then the Stop.
async function stopWithQueuedMessage(cancelQueued: boolean, chain3 = false) {
  const input = new Pushable<SDKUserMessage>();
  const q = query({
    prompt: input,
    options: {
      pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE ?? (await claudeCliPath()),
      extraArgs: { "replay-user-messages": "" },
      includePartialMessages: true,
      settingSources: [],
      env: { ...process.env, CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS: "1" },
      canUseTool: async (_name, toolInput) =>
        chain3
          ? { behavior: "allow", updatedInput: toolInput }
          : { behavior: "deny", message: "no tools in this test" },
    },
  });
  const seen: SDKMessage[] = [];
  const reader = (async () => {
    for await (const message of q) {
      seen.push(message);
    }
  })().catch(() => {});
  const queued = randomUUID();
  const nonce = `withdrawn-${randomUUID().slice(0, 8)}`;
  try {
    if (chain3) {
      const notice = randomUUID();
      input.push(
        userMessage(
          "Run this shell command with your Bash tool: echo fork-notice-read — then wait for further instructions.",
          notice,
        ),
      );
      await waitUntil(
        () =>
          seen.some((m) => {
            const lifecycle = m as { type: string; command_uuid?: string; state?: string };
            return (
              lifecycle.type === "command_lifecycle" &&
              lifecycle.command_uuid === notice &&
              lifecycle.state === "started"
            );
          }),
        90_000,
        "the notice's turn started",
      );
      input.push(
        userMessage(
          "Run this exact command in the FOREGROUND with your Bash tool and wait until it ends: timeout 90 tail -f /dev/null",
          randomUUID(),
        ),
      );
      await waitUntil(
        () =>
          seen.some(
            (m) =>
              m.type === "assistant" && JSON.stringify(m.message.content).includes("timeout 90"),
          ),
        90_000,
        "the second message's long command is running",
      );
      await sleep(1_500);
    } else {
      input.push(
        userMessage(
          "Write the numbers from one to four hundred as words, one per line. Use no tools.",
          randomUUID(),
        ),
      );
      await waitUntil(
        () => seen.some((m) => m.type === "stream_event" || m.type === "assistant"),
        90_000,
        "the first turn is generating",
      );
    }
    input.push(userMessage(`Reply with exactly this one word: ${nonce}`, queued));
    await waitUntil(
      () =>
        seen.some((m) => {
          const lifecycle = m as { type: string; command_uuid?: string; state?: string };
          return (
            lifecycle.type === "command_lifecycle" &&
            lifecycle.command_uuid === queued &&
            lifecycle.state === "queued"
          );
        }),
      15_000,
      "the second message is queued behind the running turn",
    );
    const interrupt = q.interrupt as unknown as (options?: {
      cancelQueued: true;
    }) => Promise<InterruptReceipt>;
    const receipt = cancelQueued
      ? await interrupt.call(q, { cancelQueued: true })
      : await interrupt.call(q);
    // Give a message that survived the interrupt the time to run as its own turn.
    await sleep(15_000);
    const init = seen.find((m) => m.type === "system" && m.subtype === "init") as
      | { capabilities?: string[] }
      | undefined;
    return {
      capabilities: init?.capabilities,
      receipt,
      replayed: seen.some((m) => m.type === "user" && (m as { uuid?: string }).uuid === queued),
      answered: JSON.stringify(
        seen.filter((m) => m.type === "assistant" || m.type === "stream_event"),
      ).includes(nonce),
      queued,
    };
  } finally {
    input.end();
    q.close();
    await reader;
  }
}

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)(
  "SDK contract (21b73df0): interrupt({ cancelQueued: true }) withdraws a queued message",
  () => {
    it("advertises interrupt_cancel_queued_v1, lists the queued uuid as cancelled, and never runs it", async () => {
      const run = await stopWithQueuedMessage(true);
      expect(run.capabilities).toContain("interrupt_cancel_queued_v1");
      expect(run.receipt?.cancelled).toContain(run.queued);
      expect(run.replayed).toBe(false);
      expect(run.answered).toBe(false);
    }, 180_000);

    // Positive control: the detector above SEES a run when one happens. A plain interrupt leaves the queued message
    // in place (`still_queued`), and Claude Code runs it right after: the very defect a Stop used to leave.
    // The template-fork chain (brick 21b73df0): the queued message is the THIRD, behind a message folded into the
    // notice's turn. Measured: withdrawn exactly as in the two-message case.
    it("three-message chain: the third message is listed cancelled and never runs", async () => {
      const run = await stopWithQueuedMessage(true, true);
      expect(run.receipt?.cancelled).toContain(run.queued);
      expect(run.replayed).toBe(false);
      expect(run.answered).toBe(false);
    }, 240_000);

    it("control: a plain interrupt lists it under still_queued, and Claude Code then runs it", async () => {
      const run = await stopWithQueuedMessage(false);
      expect(run.receipt?.still_queued).toContain(run.queued);
      expect(run.replayed).toBe(true);
      expect(run.answered).toBe(true);
    }, 180_000);
  },
);
