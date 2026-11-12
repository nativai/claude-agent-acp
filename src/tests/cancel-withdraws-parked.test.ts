// brick 21b73df0 — a person's Stop must stop a prompt that was steered (parked) behind the stopped turn.
//
// `prompt()` pushes a concurrent prompt's message into the SDK input at once and parks it; `cancel()` answers it
// `cancelled` and interrupts the running turn. Claude Code still holds the message and runs it as a turn of its own
// after the interrupt. Measured on the real adapter (8ae0a0bc, acpx 61e7e1d5 and b9c4a3d4): the steered tool call
// ran and its file was written seconds after the client had settled the turn idle.
//
// This harness plays the SDK by hand (as adapter-routing-hole.test.ts does): after the cancel it delivers what the
// real CLI produces for the withdrawn message: its replay, a tool use, text, the result, idle.

import { describe, it, expect, vi } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";
import { randomUUID } from "crypto";

const logger = { log: () => {}, error: () => {} };

function manualQuery() {
  const queue: any[] = [];
  const waiters: Array<(r: { value: any; done: boolean }) => void> = [];
  const query: any = {
    next() {
      if (queue.length) return Promise.resolve({ value: queue.shift(), done: false });
      return new Promise((res) => waiters.push(res));
    },
    interrupt: vi.fn(async () => {}),
    close: vi.fn(),
    [Symbol.asyncIterator]() {
      return this;
    },
  };
  return {
    query,
    push(msg: any) {
      const w = waiters.shift();
      if (w) w({ value: msg, done: false });
      else queue.push(msg);
    },
  };
}

function recordingClient() {
  const updates: any[] = [];
  const permissionRequests: any[] = [];
  return {
    client: {
      async sessionUpdate(n: any) {
        updates.push(n);
      },
      async extNotification() {},
      async requestPermission(r: any) {
        permissionRequests.push(r);
        return { outcome: { outcome: "selected", optionId: "allow" } };
      },
    } as any,
    updates,
    permissionRequests,
  };
}

function makeSession(query: any, input: any) {
  return {
    query,
    input,
    cancelled: false,
    cwd: "/test",
    sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
    modes: { currentModeId: "default", availableModes: [] },
    models: { currentModelId: "default", availableModels: [] },
    modelInfos: [],
    settingsManager: { dispose: vi.fn() } as any,
    accumulatedUsage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    },
    configOptions: [],
    promptRunning: false,
    pendingMessages: new Map(),
    nextPendingOrder: 0,
    abortController: new AbortController(),
    emitRawSDKMessages: false,
    activePromptResolve: null,
    pendingSdkMessages: [],
    backgroundLoopError: null,
    contextWindowSize: 200000,
    taskState: new Map(),
  };
}

const flush = () => new Promise((r) => setTimeout(r, 5));
const WEDGE = Symbol("WEDGED") as unknown as any;
const settledOr = (p: Promise<any>, ms = 300) =>
  Promise.race([p, new Promise((r) => setTimeout(() => r(WEDGE), ms))]);

const usage = () => ({
  input_tokens: 100,
  output_tokens: 20,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 },
  service_tier: null,
  cache_creation: { ephemeral_1h_input_tokens: 0, ephemeral_5m_input_tokens: 0 },
});

const replayOf = (pushed: any) => ({
  type: "user",
  message: pushed.message,
  parent_tool_use_id: null,
  uuid: pushed.uuid,
  session_id: "s",
  isReplay: true,
});

const textDelta = (text: string) => ({
  type: "stream_event",
  parent_tool_use_id: null,
  uuid: randomUUID(),
  session_id: "s",
  event: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } },
});

const toolUse = (command: string) => ({
  type: "assistant",
  parent_tool_use_id: null,
  uuid: randomUUID(),
  session_id: "s",
  message: {
    id: "m-" + randomUUID(),
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-20250514",
    content: [{ type: "tool_use", id: "tu-" + randomUUID(), name: "Bash", input: { command } }],
    stop_reason: "tool_use",
    stop_sequence: null,
    usage: usage(),
  } as any,
});

/** A tool result: a user message WITHOUT `isReplay` (it must not end a withdrawn turn's discard). */
const toolResult = () => ({
  type: "user",
  message: { role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: "ok" }] },
  parent_tool_use_id: null,
  uuid: randomUUID(),
  session_id: "s",
});

const resultMsg = () => ({
  type: "result",
  subtype: "success",
  stop_reason: null,
  is_error: false,
  result: "",
  errors: [],
  duration_ms: 0,
  duration_api_ms: 0,
  num_turns: 1,
  total_cost_usd: 0,
  usage: usage(),
  modelUsage: {},
  permission_denials: [],
  uuid: randomUUID(),
  session_id: "s",
});

const idleMsg = () => ({
  type: "system",
  subtype: "session_state_changed",
  state: "idle",
  session_id: "s",
});

function setup() {
  const mq = manualQuery();
  const rc = recordingClient();
  const pushed: any[] = [];
  const input = { push: (m: any) => pushed.push(m), end: vi.fn() } as any;
  const agent = new ClaudeAcpAgent(rc.client, logger);
  agent.sessions["s"] = makeSession(mq.query, input) as any;
  (agent as any).startBackgroundReaderLoop("s");
  const prompt = (text: string) =>
    agent.prompt({ sessionId: "s", prompt: [{ type: "text", text }] });
  return { agent, prompt, pushed, ...mq, ...rc };
}

/** M1 runs, M2 is steered in (parked), the person's Stop: both prompts answer `cancelled`. */
async function steerThenStop(t: ReturnType<typeof setup>) {
  const p1 = t.prompt("run the long check");
  await flush();
  const p2 = t.prompt("also write the file");
  await flush();
  const m2 = t.pushed[1];
  await t.agent.cancel({ sessionId: "s" });
  const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
  expect(r1).not.toBe(WEDGE);
  expect(r2).not.toBe(WEDGE);
  expect(r1.stopReason).toBe("cancelled");
  expect(r2.stopReason).toBe("cancelled");
  return { m2, r2, interruptsAtStop: t.query.interrupt.mock.calls.length };
}

/** What the real CLI emits after the interrupt: the withdrawn message's own turn. */
async function sdkRunsWithdrawnTurn(
  t: ReturnType<typeof setup>,
  m2: any,
  options = { idle: true },
) {
  t.push(replayOf(m2));
  await flush();
  t.push(toolUse("echo steered > file.txt"));
  await flush();
  t.push(toolResult());
  await flush();
  t.push(textDelta("I wrote the file."));
  await flush();
  t.push(resultMsg());
  await flush();
  if (options.idle) {
    t.push(idleMsg());
    await flush();
  }
}

describe("21b73df0 — Claude Code confirms the withdrawal (interrupt receipt)", () => {
  it("the Stop interrupts with cancelQueued, and a parked prompt Claude Code took back answers cancelled + _meta.withdrawn", async () => {
    const t = setup();
    t.query.interrupt.mockImplementation(async () => ({
      still_queued: [],
      cancelled: t.pushed.slice(1).map((m: any) => m.uuid),
    }));
    const { r2 } = await steerThenStop(t);
    expect(t.query.interrupt).toHaveBeenCalledWith({ cancelQueued: true });
    expect(r2).toEqual({ stopReason: "cancelled", _meta: { withdrawn: true } });
  });

  it("a parked prompt the receipt does not list answers a plain cancelled (never an inferred withdrawal)", async () => {
    const t = setup();
    t.query.interrupt.mockImplementation(async () => ({ still_queued: [], cancelled: [] }));
    const { r2, m2, interruptsAtStop } = await steerThenStop(t);
    expect(r2).toEqual({ stopReason: "cancelled" });
    // ...and if Claude Code does run it, the reader still stops that turn.
    const updatesAtStop = t.updates.length;
    await sdkRunsWithdrawnTurn(t, m2);
    expect(t.query.interrupt.mock.calls.length).toBe(interruptsAtStop + 1);
    expect(t.updates.slice(updatesAtStop)).toEqual([]);
  });

  it("a receipt that never comes does not wedge the parked prompt: it answers a plain cancelled after the bound", async () => {
    const t = setup();
    t.query.interrupt.mockImplementation(() => new Promise(() => {}));
    const p1 = t.prompt("run the long check");
    await flush();
    const p2 = t.prompt("also write the file");
    await flush();
    void t.agent.cancel({ sessionId: "s" });
    const r2 = await settledOr(p2, 3_000);
    expect(r2).not.toBe(WEDGE);
    expect(r2).toEqual({ stopReason: "cancelled" });
    expect(await settledOr(p1, 100)).not.toBe(WEDGE);
  });
});

describe("21b73df0 — fallback, no receipt (an older Claude Code): the withdrawn prompt's turn is stopped when it starts", () => {
  it("the parked prompt answers a plain cancelled", async () => {
    const t = setup();
    const { r2 } = await steerThenStop(t);
    expect(r2).toEqual({ stopReason: "cancelled" });
  });

  it("the turn the SDK runs for the withdrawn message is interrupted, and nothing of it reaches the client", async () => {
    const t = setup();
    const { m2, interruptsAtStop } = await steerThenStop(t);
    const updatesAtStop = t.updates.length;
    await sdkRunsWithdrawnTurn(t, m2);
    expect(t.query.interrupt.mock.calls.length).toBe(interruptsAtStop + 1);
    expect(t.updates.slice(updatesAtStop)).toEqual([]);
  });

  it("a tool the withdrawn turn asks for is denied, and the client is never asked", async () => {
    const t = setup();
    const { m2 } = await steerThenStop(t);
    t.push(replayOf(m2));
    await flush();
    const decision = await t.agent.canUseTool("s")("Bash", { command: "echo steered > file.txt" }, {
      signal: new AbortController().signal,
      suggestions: undefined,
      toolUseID: "tu-1",
    } as any);
    expect(decision?.behavior).toBe("deny");
    expect(t.permissionRequests).toEqual([]);
  });

  it("a prompt sent after the Stop runs normally, even when the withdrawn turn's messages (and its idle) come first", async () => {
    const t = setup();
    const { m2 } = await steerThenStop(t);
    const p3 = t.prompt("a new message after the Stop");
    await flush();
    const m3 = t.pushed[2];
    const updatesBefore = t.updates.length;
    await sdkRunsWithdrawnTurn(t, m2);
    // The withdrawn turn's idle must not end p3's turn.
    expect(await settledOr(p3, 50)).toBe(WEDGE);
    t.push(replayOf(m3));
    await flush();
    t.push(textDelta("answer to the new message"));
    await flush();
    t.push(resultMsg());
    await flush();
    t.push(idleMsg());
    await flush();
    const r3 = await settledOr(p3);
    expect(r3).not.toBe(WEDGE);
    expect(r3.stopReason).toBe("end_turn");
    const texts = t.updates
      .slice(updatesBefore)
      .filter((u) => u.update?.sessionUpdate === "agent_message_chunk")
      .map((u) => u.update.content?.text);
    expect(texts).toEqual(["answer to the new message"]);
  });

  it("a withdrawn turn that reaches no idle stops being discarded at the next prompt's replay", async () => {
    const t = setup();
    const { m2 } = await steerThenStop(t);
    const p3 = t.prompt("a new message after the Stop");
    await flush();
    const m3 = t.pushed[2];
    await sdkRunsWithdrawnTurn(t, m2, { idle: false });
    t.push(replayOf(m3));
    await flush();
    t.push(textDelta("answer to the new message"));
    await flush();
    t.push(resultMsg());
    await flush();
    t.push(idleMsg());
    await flush();
    const r3 = await settledOr(p3);
    expect(r3).not.toBe(WEDGE);
    expect(r3.stopReason).toBe("end_turn");
    expect(
      t.updates.some(
        (u) =>
          u.update?.sessionUpdate === "agent_message_chunk" &&
          u.update.content?.text === "answer to the new message",
      ),
    ).toBe(true);
  });

  it("negative: without a Stop the parked prompt hands off and its turn is forwarded", async () => {
    const t = setup();
    const p1 = t.prompt("run the long check");
    await flush();
    const p2 = t.prompt("also write the file");
    await flush();
    const m2 = t.pushed[1];
    t.push(replayOf(m2));
    await flush();
    t.push(textDelta("I wrote the file."));
    await flush();
    t.push(resultMsg());
    await flush();
    t.push(idleMsg());
    await flush();
    const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
    expect(r1.stopReason).toBe("end_turn");
    expect(r2.stopReason).toBe("end_turn");
    expect(t.query.interrupt).not.toHaveBeenCalled();
    expect(
      t.updates.some(
        (u) =>
          u.update?.sessionUpdate === "agent_message_chunk" &&
          u.update.content?.text === "I wrote the file.",
      ),
    ).toBe(true);
  });
});
