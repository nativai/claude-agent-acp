// Prompt lifecycle signals and the end of the mid-prompt turn-end marker —
// brick a147982f, CONCEPTION §4.1 (contract), §4.2 A1–A5, §6 adapter rows.
//
// acpx's turn watchdog armed on `_claude/lastTurnEndReason`, which this adapter
// put on EVERY `result`'s usage_update. A result ends one model loop, not the
// ACP prompt: a background Agent re-drives the model inside the same prompt,
// and the watchdog cut that live work 120 s after the marker. These rows pin:
// - A1: no marker on a mid-prompt usage_update (only on an error-completing one);
// - A3/A4: `_claude/promptLifecycle` `sdk_idle` (reader) and `completing` (loop),
//   attributed by the request's `_claude/promptId`, before the response;
// - A5: nothing for idle-time turns, hand-off returns, or prompts with no id.
//
// Every message reaches the reader through a hand-driven query, so a test
// controls exactly when a message arrives relative to the prompt loop's awaits,
// and every client write lands in ONE ordered `wire` log, so ordering
// (signal before response) is asserted rather than assumed.

import { describe, it, expect, vi } from "vitest";
import { randomUUID } from "crypto";
import {
  ClaudeAcpAgent,
  LAST_TURN_END_REASON_META_KEY,
  PROMPT_ID_META_KEY,
  PROMPT_LIFECYCLE_NOTIFICATION,
} from "../acp-agent.js";

const logger = { log: () => {}, error: () => {} };

type Wire =
  | { k: "update"; n: any }
  | { k: "ext"; method: string; params: any }
  | { k: "response"; promptId?: string; r: any }
  | { k: "reject"; promptId?: string; e: any };

/** One SDK message per `push()`, delivered to the reader's `query.next()`. */
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

function setup() {
  const wire: Wire[] = [];
  let gate: Promise<void> | null = null;
  let release: (() => void) | null = null;
  const client = {
    async sessionUpdate(n: any) {
      wire.push({ k: "update", n });
      if (gate) await gate;
    },
    async extNotification(method: string, params: any) {
      wire.push({ k: "ext", method, params });
    },
  } as any;
  const mq = manualQuery();
  const pushed: any[] = [];
  const agent = new ClaudeAcpAgent(client, logger);
  agent.sessions["s"] = {
    query: mq.query,
    input: { push: (m: any) => pushed.push(m), end: vi.fn() },
    cancelled: false,
    cwd: "/test",
    sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
    modes: { currentModeId: "default", availableModes: [] },
    models: { currentModelId: "default", availableModels: [] },
    modelInfos: [],
    settingsManager: { dispose: vi.fn() },
    accumulatedUsage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    },
    configOptions: [],
    availableOutputStyles: [],
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
  } as any;
  (agent as any).startBackgroundReaderLoop("s");

  /** Starts a prompt (with `promptId` in `_meta` unless undefined) and records
   *  its settlement into the wire log. Returns the settled promise. */
  const prompt = (text: string, promptId?: string) =>
    agent
      .prompt({
        sessionId: "s",
        prompt: [{ type: "text", text }],
        ...(promptId !== undefined && { _meta: { [PROMPT_ID_META_KEY]: promptId } }),
      })
      .then(
        (r) => {
          wire.push({ k: "response", promptId, r });
          return r;
        },
        (e) => {
          wire.push({ k: "reject", promptId, e });
          return e;
        },
      );

  return {
    agent,
    wire,
    pushed,
    push: mq.push,
    prompt,
    block() {
      gate = new Promise<void>((r) => (release = r));
    },
    unblock() {
      const r = release;
      gate = null;
      release = null;
      r?.();
    },
  };
}

const flush = () => new Promise((r) => setTimeout(r, 5));

const usage = (n = 100) => ({
  input_tokens: n,
  output_tokens: n ? 20 : 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const assistant = (opts: { tokens?: number; error?: string } = {}) => ({
  type: "assistant",
  parent_tool_use_id: null,
  uuid: randomUUID(),
  session_id: "s",
  ...(opts.error && { error: opts.error }),
  message: {
    id: "m-" + randomUUID(),
    type: "message",
    role: "assistant",
    model: "claude-haiku-4-5-20251001",
    content: [] as any[],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: usage(opts.tokens ?? 100),
  },
});

const result = (overrides: Record<string, unknown> = {}) => ({
  type: "result",
  subtype: "success",
  stop_reason: "end_turn",
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
  ...overrides,
});

const TASK_NOTIFICATION_ORIGIN = { kind: "task-notification", producer: "session-task" };
const sys = (subtype: string, extra: Record<string, unknown> = {}) => ({
  type: "system",
  subtype,
  session_id: "s",
  ...extra,
});
const idle = () => sys("session_state_changed", { state: "idle" });
const running = () => sys("session_state_changed", { state: "running" });
const replayOf = (pushedUserMessage: any) => ({
  type: "user",
  message: pushedUserMessage.message,
  parent_tool_use_id: null,
  uuid: pushedUserMessage.uuid,
  session_id: "s",
  isReplay: true,
});

/** Feed messages one at a time, letting the loop process each. */
async function feed(t: ReturnType<typeof setup>, msgs: any[]) {
  for (const m of msgs) {
    t.push(m);
    await flush();
  }
}

// ---- wire readers ----------------------------------------------------------

const lifecycle = (wire: Wire[], phase?: string) =>
  wire.flatMap((w, i) =>
    w.k === "ext" &&
    w.method === PROMPT_LIFECYCLE_NOTIFICATION &&
    (phase === undefined || w.params.phase === phase)
      ? [{ i, params: w.params }]
      : [],
  );
const usageUpdates = (wire: Wire[]) =>
  wire.flatMap((w, i) =>
    w.k === "update" && w.n.update?.sessionUpdate === "usage_update" ? [{ i, n: w.n }] : [],
  );
const markersOnUpdates = (wire: Wire[]) =>
  wire.filter(
    (w) => w.k === "update" && w.n.update?._meta?.[LAST_TURN_END_REASON_META_KEY] !== undefined,
  );
const settlementIndex = (wire: Wire[], promptId?: string) =>
  wire.findIndex((w) => (w.k === "response" || w.k === "reject") && w.promptId === promptId);
const phasesFor = (wire: Wire[], promptId: string) =>
  lifecycle(wire)
    .filter((s) => s.params.promptId === promptId)
    .map((s) => s.params.phase);

// Every "no signal here" row first asserts that the same harness DID carry the
// owning prompt's signals. Without that control an absence proves nothing — it
// holds just as well on an adapter that never signals at all.

// ---- §6 adapter rows -------------------------------------------------------

describe("prompt lifecycle signals (brick a147982f, §6 adapter rows)", () => {
  it("P1: plain turn with an id → sdk_idle, then completing{end_turn}, then the response; no marker on the usage_update", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("hi", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
    const r: any = await p;

    const sdkIdle = lifecycle(t.wire, "sdk_idle");
    const completing = lifecycle(t.wire, "completing");
    expect(sdkIdle.map((s) => s.params)).toEqual([
      { sessionId: "s", promptId: id, phase: "sdk_idle" },
    ]);
    expect(completing.map((s) => s.params)).toEqual([
      { sessionId: "s", promptId: id, phase: "completing", lastTurnEndReason: "end_turn" },
    ]);
    const responseAt = settlementIndex(t.wire, id);
    expect(sdkIdle[0].i).toBeLessThan(completing[0].i);
    expect(completing[0].i).toBeLessThan(responseAt);

    expect(r.stopReason).toBe("end_turn");
    expect(r._meta).toEqual({
      [LAST_TURN_END_REASON_META_KEY]: "end_turn",
      [PROMPT_ID_META_KEY]: id,
    });
    expect(usageUpdates(t.wire)).toHaveLength(1);
    expect(markersOnUpdates(t.wire)).toHaveLength(0);
  });

  it("P2: async Agent — result, task events, continuation result (task-notification), idle → one sdk_idle + one completing, both at the end; no marker on either result", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("start up", id);
    await flush();
    await feed(t, [
      replayOf(t.pushed[0]),
      assistant(),
      result(), // "I'll wait for the helper" — ends one model loop, NOT the prompt
      sys("background_tasks_changed", { tasks: [{ task_id: "a1", task_type: "local_agent" }] }),
      sys("task_started", { task_id: "a1", tool_use_id: "toolu_a1", description: "identify" }),
      sys("task_progress", { task_id: "a1", tool_use_id: "toolu_a1", last_tool_name: "Bash" }),
      sys("task_notification", { task_id: "a1", tool_use_id: "toolu_a1", status: "completed" }),
      sys("init"),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }), // the main agent's continuation
    ]);

    // Live continuation, still inside the prompt: nothing has signalled an end.
    expect(lifecycle(t.wire)).toHaveLength(0);
    expect(settlementIndex(t.wire, id)).toBe(-1);

    await feed(t, [idle()]);
    await p;

    const all = lifecycle(t.wire);
    expect(all.map((s) => s.params.phase)).toEqual(["sdk_idle", "completing"]);
    expect(all.every((s) => s.params.promptId === id)).toBe(true);
    const lastUsage = usageUpdates(t.wire).at(-1)!;
    expect(usageUpdates(t.wire)).toHaveLength(2);
    expect(all[0].i).toBeGreaterThan(lastUsage.i);
    expect(markersOnUpdates(t.wire)).toHaveLength(0);
  });

  it("P3: is_error result with the rate-limit shape (used:0) keeps the `error` marker on its usage_update, sends completing{error}, and rejects", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [
      replayOf(t.pushed[0]),
      assistant({ tokens: 0, error: "rate_limit" }),
      result({ is_error: true, result: "You've hit your limit · resets 8pm" }),
    ]);
    await p;

    const usageRows = usageUpdates(t.wire);
    expect(usageRows).toHaveLength(1);
    expect(usageRows[0].n.update.used).toBe(0);
    expect(usageRows[0].n.update._meta[LAST_TURN_END_REASON_META_KEY]).toBe("error");

    const completing = lifecycle(t.wire, "completing");
    expect(completing.map((s) => s.params)).toEqual([
      { sessionId: "s", promptId: id, phase: "completing", lastTurnEndReason: "error" },
    ]);
    const rejectAt = t.wire.findIndex((w) => w.k === "reject");
    expect(rejectAt).toBeGreaterThan(completing[0].i);
    expect((t.wire[rejectAt] as any).e.data).toEqual({ errorKind: "rate_limit" });
  });

  it("P4: a prompt without an id (a non-acpx client) gets no lifecycle signal at all", async () => {
    const t = setup();
    const p = t.prompt("hi");
    await flush();
    await feed(t, [
      replayOf(t.pushed[0]),
      assistant(),
      result(),
      sys("task_notification", { task_id: "a1", status: "completed" }),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
      idle(),
    ]);
    const r: any = await p;

    expect(t.wire.filter((w) => w.k === "ext")).toHaveLength(0);
    expect(r._meta).toEqual({ [LAST_TURN_END_REASON_META_KEY]: "end_turn" });
    expect(markersOnUpdates(t.wire)).toHaveLength(0);
  });

  it("P4: an empty or non-string promptId counts as no id (control: a valid one signals)", async () => {
    const valid = randomUUID();
    for (const [id, signals] of [
      ["", false],
      [42, false],
      [valid, true],
    ] as const) {
      const t = setup();
      const p = t.agent.prompt({
        sessionId: "s",
        prompt: [{ type: "text", text: "hi" }],
        _meta: { [PROMPT_ID_META_KEY]: id },
      });
      await flush();
      await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
      const r: any = await p;
      expect(lifecycle(t.wire).map((s) => s.params.phase)).toEqual(
        signals ? ["sdk_idle", "completing"] : [],
      );
      expect(r._meta[PROMPT_ID_META_KEY]).toBe(signals ? valid : undefined);
    }
  });

  it("P5: an idle the reader buffers while the loop is mid-await is signalled at once (before the loop drains it), and the loop still returns normally", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant()]);
    t.block();
    t.push(result());
    await flush(); // loop MID-AWAIT on the result's usage_update
    t.push(idle()); // buffered by the reader, not delivered
    await flush();

    // The routing-hole case the backstop exists for: the reader has seen SDK
    // idle, the loop has not. The attributed sdk_idle is already on the wire.
    expect(lifecycle(t.wire).map((s) => s.params.phase)).toEqual(["sdk_idle"]);
    expect(lifecycle(t.wire)[0].params.promptId).toBe(id);
    expect(settlementIndex(t.wire, id)).toBe(-1);

    t.unblock();
    const r: any = await p;
    expect(r.stopReason).toBe("end_turn");
    expect(lifecycle(t.wire).map((s) => s.params.phase)).toEqual(["sdk_idle", "completing"]);
    expect(lifecycle(t.wire, "completing")[0].i).toBeLessThan(settlementIndex(t.wire, id));
  });

  it("N4: a task-notification follow-up result after an earlier result re-stamps no marker on either usage_update", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [
      replayOf(t.pushed[0]),
      assistant(),
      result({ stop_reason: "max_tokens" }),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
      idle(),
    ]);
    const r: any = await p;

    const rows = usageUpdates(t.wire);
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      expect(row.n.update._meta?.[LAST_TURN_END_REASON_META_KEY]).toBeUndefined();
    }
    // The task-notification origin still rides on its own updates.
    expect(rows[1].n.update._meta["_claude/origin"]).toEqual(TASK_NOTIFICATION_ORIGIN);
    // The user turn's own reason survives to the response.
    expect(r._meta[LAST_TURN_END_REASON_META_KEY]).toBe("max_tokens");
    expect(lifecycle(t.wire, "completing")[0].params.lastTurnEndReason).toBe("max_tokens");
  });

  it("N5: messages handleIdleMessage forwards after the prompt returned carry no marker and no lifecycle signal", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
    await p;
    expect(phasesFor(t.wire, id)).toEqual(["sdk_idle", "completing"]); // control
    const before = t.wire.length;

    // Teammate / sub-agent activity after the turn ended.
    await feed(t, [
      sys("task_progress", { task_id: "a1", last_tool_name: "Read" }),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
      idle(),
    ]);
    const after = t.wire.slice(before);
    expect(after.length).toBeGreaterThan(0);
    expect(after.filter((w) => w.k === "ext")).toHaveLength(0);
    expect(markersOnUpdates(after)).toHaveLength(0);
  });

  it("N5: an idle-time turn (S3: background Bash finished after the prompt returned) gets no sdk_idle, no completing, no marker", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
    await p;
    expect(phasesFor(t.wire, id)).toEqual(["sdk_idle", "completing"]); // control
    const before = t.wire.length;

    await feed(t, [
      sys("background_tasks_changed", { tasks: [] }),
      sys("task_notification", { task_id: "b1", status: "completed" }),
      running(),
      sys("init"),
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
      idle(),
    ]);
    const after = t.wire.slice(before);
    expect(lifecycle(after)).toHaveLength(0);
    expect(markersOnUpdates(after)).toHaveLength(0);
  });

  it("N5: the hand-off return sends no signal for the handed-off prompt; its successor gets exactly one sdk_idle + completing", async () => {
    const t = setup();
    const idA = randomUUID();
    const idB = randomUUID();
    const pA = t.prompt("first", idA);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result()]);
    const pB = t.prompt("second", idB); // parks behind A
    await flush();
    await feed(t, [replayOf(t.pushed[1])]); // A hands off to B
    const rA: any = await pA;

    expect(rA.stopReason).toBe("end_turn");
    expect(lifecycle(t.wire)).toHaveLength(0);

    await feed(t, [assistant(), result(), idle()]);
    await pB;
    expect(lifecycle(t.wire).map((s) => [s.params.phase, s.params.promptId])).toEqual([
      ["sdk_idle", idB],
      ["completing", idB],
    ]);
  });

  it("N5: the cancel return sends no completing", async () => {
    const t = setup();
    const done = randomUUID();
    const p1 = t.prompt("x", done);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
    await p1;
    expect(phasesFor(t.wire, done)).toEqual(["sdk_idle", "completing"]); // control

    const cancelled = randomUUID();
    const p2 = t.prompt("y", cancelled);
    await flush();
    await feed(t, [replayOf(t.pushed[1]), assistant()]);
    await t.agent.cancel({ sessionId: "s" });
    const r: any = await p2;
    expect(r.stopReason).toBe("cancelled");
    expect(phasesFor(t.wire, cancelled)).toEqual([]);
  });

  it("A3: a cancel that the SDK acknowledges with idle reports completing{cancelled}, even after a pre-cancel end_turn result", async () => {
    const t = setup();
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result()]);
    // Mark cancelled without waking the loop through cancel()'s own return path.
    (t.agent.sessions["s"] as any).cancelled = true;
    await feed(t, [idle()]);
    const r: any = await p;

    expect(r.stopReason).toBe("cancelled");
    expect(r._meta[LAST_TURN_END_REASON_META_KEY]).toBe("cancelled");
    expect(lifecycle(t.wire, "completing").map((s) => s.params.lastTurnEndReason)).toEqual([
      "cancelled",
    ]);
  });

  it("A3: a failing lifecycle write never changes the prompt's outcome", async () => {
    const t = setup();
    (t.agent as any).client.extNotification = async () => {
      throw new Error("transport closed");
    };
    const id = randomUUID();
    const p = t.prompt("x", id);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]);
    const r: any = await p;
    expect(r.stopReason).toBe("end_turn");
    expect(r._meta[PROMPT_ID_META_KEY]).toBe(id);
  });
});

// ---- The edge case: a prompt arriving during an idle-time continuation -------
//
// S3: a background Bash finished after its prompt returned, so the SDK runs a
// turn on its own (`running` … `result{task-notification}` … `idle`) with no
// prompt. P2 arrives in the middle of it: P2 takes the stream (promptRunning,
// activePromptId = P2), so the rest of the continuation reaches P2's loop.
//
// The loop has ONE criterion for "this idle is my turn's end": it is the first
// idle routed to it while it owns the stream (the idle branch returns on any
// idle; there is no per-prompt check since upstream 23b3073 removed
// `promptReplayed`). A4 uses that same criterion, so `sdk_idle` fires exactly
// for the idle the loop returns on — never for an idle it does not take.
//
// Measured on CLI 2.1.287 (brick a147982f verification/IMPL-adapter.md, probes
// wdadapt-edge-1/-2): once P2 is queued the CLI emits NO separate continuation
// idle — P2 is either folded into the running continuation (its replay arrives
// mid-turn) or run as the next loop of the same `running` span. The only idle
// after P2's push follows P2's replay. Both shapes are pinned below.

describe("edge: a prompt arriving during an idle-time continuation", () => {
  async function promptOneThenStartContinuation(t: ReturnType<typeof setup>, id1: string) {
    const p1 = t.prompt("launch background bash", id1);
    await flush();
    await feed(t, [replayOf(t.pushed[0]), assistant(), result(), idle()]); // "WAITING"
    await p1;
    // The background command completes; the SDK starts a turn on its own.
    await feed(t, [
      sys("background_tasks_changed", { tasks: [] }),
      sys("task_notification", { task_id: "b1", status: "completed" }),
      running(),
      sys("init"),
    ]);
  }

  it("measured shape 1 (wdadapt-edge-1): P2 folded into the continuation mid-tool → no signal until P2's own end, then exactly one sdk_idle + one completing for P2", async () => {
    const t = setup();
    const id1 = randomUUID();
    const id2 = randomUUID();
    await promptOneThenStartContinuation(t, id1);
    const p1Signals = lifecycle(t.wire).length;

    const p2 = t.prompt("second", id2);
    await flush();
    await feed(t, [
      assistant(), // continuation's tool_use (foreground)
      sys("task_started", { task_id: "b2", description: "sleep" }),
      sys("task_notification", { task_id: "b2", status: "completed" }),
      replayOf(t.pushed[1]), // P2's message attached mid-turn
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }),
    ]);
    expect(lifecycle(t.wire).length).toBe(p1Signals);
    expect(settlementIndex(t.wire, id2)).toBe(-1);

    await feed(t, [idle()]);
    const r2: any = await p2;
    expect(r2.stopReason).toBe("end_turn");
    const p2Signals = lifecycle(t.wire).slice(p1Signals);
    expect(p2Signals.map((s) => [s.params.phase, s.params.promptId])).toEqual([
      ["sdk_idle", id2],
      ["completing", id2],
    ]);
    expect(p2Signals[1].i).toBeLessThan(settlementIndex(t.wire, id2));
    expect(markersOnUpdates(t.wire)).toHaveLength(0);
  });

  it("measured shape 2 (wdadapt-edge-2): P2 queued during tool-free generation → continuation result, no idle, P2's loop, one idle → exactly one sdk_idle + one completing for P2", async () => {
    const t = setup();
    const id1 = randomUUID();
    const id2 = randomUUID();
    await promptOneThenStartContinuation(t, id1);
    const p1Signals = lifecycle(t.wire).length;

    const p2 = t.prompt("second", id2);
    await flush();
    await feed(t, [
      assistant(),
      result({ origin: TASK_NOTIFICATION_ORIGIN }), // continuation ends; CLI emits NO idle
      sys("init"),
      replayOf(t.pushed[1]),
      assistant(),
      result(), // "SECOND"
    ]);
    expect(lifecycle(t.wire).length).toBe(p1Signals);

    await feed(t, [idle()]);
    await p2;
    expect(
      lifecycle(t.wire)
        .slice(p1Signals)
        .map((s) => [s.params.phase, s.params.promptId]),
    ).toEqual([
      ["sdk_idle", id2],
      ["completing", id2],
    ]);
    expect(markersOnUpdates(t.wire)).toHaveLength(0);
  });

  it("race (idle already in flight when P2 arrives): sdk_idle is never left unanswered — it fires only with the loop's own completing + response, and P2's later SDK turn emits nothing", async () => {
    const t = setup();
    const id1 = randomUUID();
    const id2 = randomUUID();
    await promptOneThenStartContinuation(t, id1);
    const p1Signals = lifecycle(t.wire).length;

    const p2 = t.prompt("second", id2);
    await flush();
    // The continuation's idle was written before the CLI dequeued P2.
    await feed(t, [assistant(), result({ origin: TASK_NOTIFICATION_ORIGIN }), idle()]);
    await p2;

    const p2Signals = lifecycle(t.wire).slice(p1Signals);
    expect(p2Signals.map((s) => [s.params.phase, s.params.promptId])).toEqual([
      ["sdk_idle", id2],
      ["completing", id2],
    ]);
    // sdk_idle → completing → response, with nothing in between that would
    // leave the watchdog armed on a live turn.
    const respAt = settlementIndex(t.wire, id2);
    expect(t.wire.slice(p2Signals[0].i + 1, respAt).map((w) => w.k)).toEqual(["ext"]);

    // P2's actual SDK turn now streams with no prompt owning the stream.
    const before = t.wire.length;
    await feed(t, [running(), replayOf(t.pushed[1]), assistant(), result(), idle()]);
    expect(lifecycle(t.wire.slice(before))).toHaveLength(0);
  });

  // ⚠️ KNOWN PRE-EXISTING DEFECT — reported to the HoD (brick a147982f,
  // verification/IMPL-adapter.md §Edge case), deliberately NOT fixed in this
  // lane. In the race above the loop returns P2's response on the
  // continuation's idle, before the SDK has even replayed P2: P2's real turn
  // then streams as unowned inter-turn output. This row asserts the CORRECT
  // behaviour and is expected to fail; whoever fixes the loop flips it to `it`.
  it.fails(
    "pre-existing: P2's response must not resolve on an idle that precedes P2's replay",
    async () => {
      const t = setup();
      await promptOneThenStartContinuation(t, randomUUID());
      const p2 = t.prompt("second", randomUUID());
      await flush();
      await feed(t, [assistant(), result({ origin: TASK_NOTIFICATION_ORIGIN }), idle()]);
      const settledBeforeReplay = await Promise.race([
        p2.then(() => true),
        flush().then(() => false),
      ]);
      expect(settledBeforeReplay).toBe(false);
    },
  );
});
