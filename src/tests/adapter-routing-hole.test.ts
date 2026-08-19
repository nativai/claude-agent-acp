// Repro-first tests for the `activePromptResolve` routing hole (A1) and the
// mid-turn usage clobber (A2) — brick d5f9c8bb / CONCEPTION §1.1, §1.4.
//
// The DECISIVE fidelity axis (CONCEPTION §1.4): *when* a turn-control message
// (session_state_changed:idle, or a user replay) arrives relative to the prompt
// loop's awaits. A rig that only delivers messages while the loop is PARKED at
// nextMessage() never exercises the bug — that is exactly why the pre-existing
// suite stayed green on it. So this harness delivers control messages while the
// loop is MID-AWAIT on a (gated) client.sessionUpdate() — the real production
// window (RCA §1.2). Each case is annotated with what it does on the unfixed
// adapter ("WITHOUT A1/A2"); all were confirmed to fail (wedge / zero-usage)
// against a reverted acp-agent.ts before the fix landed.

import { describe, it, expect, vi } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";
import { randomUUID } from "crypto";

const logger = { log: () => {}, error: () => {} };

/** A manually-driven stand-in for the SDK `Query`. The background reader awaits
 *  `query.next()`; each `push()` delivers EXACTLY ONE message, so a test controls
 *  precisely when a message reaches the reader relative to the prompt loop's
 *  awaits. `end()` signals stream termination. */
function manualQuery() {
  const queue: any[] = [];
  const waiters: Array<(r: { value: any; done: boolean }) => void> = [];
  let ended = false;
  const query: any = {
    next() {
      if (queue.length) return Promise.resolve({ value: queue.shift(), done: false });
      if (ended) return Promise.resolve({ value: undefined, done: true });
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
    end() {
      ended = true;
      let w;
      while ((w = waiters.shift())) w({ value: undefined, done: true });
    },
  };
}

/** A gated ACP client. `block()` makes every subsequent sessionUpdate() park on
 *  an unresolved promise, so the prompt loop sits MID-AWAIT (activePromptResolve
 *  null) — the routing-hole window. `unblock()` releases it. */
function gatedClient() {
  const updates: any[] = [];
  let gate: Promise<void> | null = null;
  let release: (() => void) | null = null;
  return {
    client: {
      async sessionUpdate(n: any) {
        updates.push(n);
        if (gate) await gate;
      },
      async extNotification() {},
    } as any,
    updates,
    block() {
      gate = new Promise<void>((r) => {
        release = r;
      });
    },
    unblock() {
      const r = release;
      gate = null;
      release = null;
      r?.();
    },
  };
}

/** Captures the SDKUserMessages the prompt loop pushes to session input, so a
 *  test can read the internally-generated promptUuid it must echo back as a
 *  replay. (The real SDK replays these; here we replay by hand.) */
function captureInput() {
  const pushed: any[] = [];
  return { input: { push: (m: any) => pushed.push(m), end: vi.fn() } as any, pushed };
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
const timeout = <T>(ms: number, val: T) => new Promise<T>((r) => setTimeout(() => r(val), ms));
const WEDGE = Symbol("WEDGED") as unknown as any;
/** Resolve `p` or, if it stays pending past `ms`, resolve to the WEDGE sentinel.
 *  A prompt() that never resolves is exactly the production wedge. */
const settledOr = (p: Promise<any>, ms = 200) => Promise.race([p, timeout(ms, WEDGE)]);

const usage = () => ({
  input_tokens: 100,
  output_tokens: 20,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 },
  service_tier: null,
  cache_creation: { ephemeral_1h_input_tokens: 0, ephemeral_5m_input_tokens: 0 },
});

/** A top-level assistant message with empty content: sets lastAssistantTotalUsage
 *  (so the following `result` awaits a usage_update — the gated window) without
 *  emitting any notification of its own. */
const assistantMsg = () => ({
  type: "assistant",
  parent_tool_use_id: null,
  uuid: randomUUID(),
  session_id: "s",
  message: {
    id: "m-" + randomUUID(),
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-20250514",
    content: [] as any[],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: usage(),
  } as any,
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

/** A local_command_output system message: a single, unconditional gated
 *  sessionUpdate() await — the simplest way to hold the loop MID-AWAIT. */
const localCmdMsg = () => ({
  type: "system",
  subtype: "local_command_output",
  session_id: "s",
  content: "output",
});

/** Echo a captured pushed user message back as the SDK's replay. */
const replayOf = (pushedUserMessage: any) => ({
  type: "user",
  message: pushedUserMessage.message,
  parent_tool_use_id: null,
  uuid: pushedUserMessage.uuid,
  session_id: "s",
  isReplay: true,
});

function setup() {
  const mq = manualQuery();
  const gc = gatedClient();
  const cap = captureInput();
  const agent = new ClaudeAcpAgent(gc.client, logger);
  agent.sessions["s"] = makeSession(mq.query, cap.input) as any;
  (agent as any).startBackgroundReaderLoop("s");
  const prompt = (text: string) =>
    agent.prompt({ sessionId: "s", prompt: [{ type: "text", text }] });
  return { agent, prompt, ...mq, ...gc, pushed: cap.pushed };
}

describe("A1 — activePromptResolve routing hole", () => {
  it("(a) turn whose idle-status lands mid-await still resolves (not one-request-lagged/wedged)", async () => {
    // WITHOUT A1: the idle-status is diverted to handleIdleMessage and discarded;
    // the loop parks forever and prompt() never resolves.
    const t = setup();
    const p = t.prompt("hi");
    await flush(); // loop parks at nextMessage

    t.push(assistantMsg());
    await flush(); // sets lastAssistantTotalUsage, parks again
    t.block();
    t.push(resultMsg());
    await flush(); // loop now MID-AWAIT on the result's usage_update
    t.push(idleMsg()); // control message arrives in the routing-hole window
    await flush();
    t.unblock();

    const r = await settledOr(p);
    expect(r).not.toBe(WEDGE);
    expect(r.stopReason).toBe("end_turn");
  });

  it("(b) a parked second prompt whose replay lands mid-await is not stranded forever", async () => {
    // WITHOUT A1: the replay is discarded (handleIdleMessage has no user case),
    // so turn 1 never hands off and the injected prompt is parked forever.
    const t = setup();
    const p1 = t.prompt("first");
    await flush(); // turn 1 running, parked
    const p2 = t.prompt("second"); // injected mid-turn → parks in pendingMessages
    await flush();
    const p2uuid = t.pushed[1];

    t.block();
    t.push(localCmdMsg());
    await flush(); // turn 1 MID-AWAIT
    t.push(replayOf(p2uuid)); // p2's replay in the routing-hole window
    await flush();
    t.unblock();
    await flush(); // turn 1 drains the replay → hands off to p2
    t.push(idleMsg()); // ends p2's turn
    await flush();

    const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
    expect(r1).not.toBe(WEDGE);
    expect(r2).not.toBe(WEDGE); // the parked-forever repro
    expect(r1.stopReason).toBe("end_turn");
    expect(r2.stopReason).toBe("end_turn");
  });

  it("(c) replay + idle both landing at turn-teardown: handoff, remaining buffer drained by successor", async () => {
    // WITHOUT A1: both control messages are eaten in the usage_update window; the
    // turn parks forever. WITH A1: the loop drains the replay (handoff, leaving
    // the idle buffered) and the successor prompt drains the idle to end.
    const t = setup();
    const p1 = t.prompt("first");
    await flush();
    const p2 = t.prompt("second");
    await flush();
    const p2uuid = t.pushed[1];

    t.push(assistantMsg());
    await flush();
    t.block();
    t.push(resultMsg());
    await flush(); // MID-AWAIT on usage_update
    t.push(replayOf(p2uuid)); // buffered [replay]
    await flush();
    t.push(idleMsg()); // buffered [replay, idle]
    await flush();
    t.unblock();
    await flush();

    const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
    expect(r1).not.toBe(WEDGE);
    expect(r2).not.toBe(WEDGE);
    expect(r1.stopReason).toBe("end_turn");
    expect(r2.stopReason).toBe("end_turn");
  });

  it("(d) two parked prompts handed off in a chain (turn1→p2→p3)", async () => {
    const t = setup();
    const p1 = t.prompt("first");
    await flush();
    const p2 = t.prompt("second");
    await flush();
    const p3 = t.prompt("third");
    await flush();
    const p2uuid = t.pushed[1];
    const p3uuid = t.pushed[2];

    // turn1 → p2 (replay eaten mid-await without the fix)
    t.block();
    t.push(localCmdMsg());
    await flush();
    t.push(replayOf(p2uuid));
    await flush();
    t.unblock();
    await flush();

    // p2 → p3 (again mid-await)
    t.block();
    t.push(localCmdMsg());
    await flush();
    t.push(replayOf(p3uuid));
    await flush();
    t.unblock();
    await flush();

    // p3 ends on idle
    t.push(idleMsg());
    await flush();

    const [r1, r2, r3] = await Promise.all([settledOr(p1), settledOr(p2), settledOr(p3)]);
    expect([r1, r2, r3].map((r) => r === WEDGE)).toEqual([false, false, false]);
    expect([r1.stopReason, r2.stopReason, r3.stopReason]).toEqual([
      "end_turn",
      "end_turn",
      "end_turn",
    ]);
  });

  it("(e) clean exit via idle-status still drains a buffered replay to release a parked prompt", async () => {
    // Exercises the finally clean-exit drain (the deterministic replacement for
    // the blind 'resolve one pending' band-aid). Turn 1 ends via its OWN
    // idle-status (not a handoff), but a parked prompt's replay was buffered at
    // teardown; the finally must drain it and hand off.
    // WITHOUT A1: idle + replay both eaten → turn 1 wedges, p2 stranded.
    const t = setup();
    const p1 = t.prompt("first");
    await flush();
    const p2 = t.prompt("second");
    await flush();
    const p2uuid = t.pushed[1];

    t.push(assistantMsg());
    await flush();
    t.block();
    t.push(resultMsg());
    await flush(); // MID-AWAIT
    t.push(idleMsg()); // buffered [idle]  — turn 1's own end
    await flush();
    t.push(replayOf(p2uuid)); // buffered [idle, replay]
    await flush();
    t.unblock();
    await flush(); // loop drains idle → returns end_turn; finally drains replay → hands off p2
    t.push(idleMsg()); // ends p2
    await flush();

    const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
    expect(r1).not.toBe(WEDGE);
    expect(r2).not.toBe(WEDGE);
    expect(r1.stopReason).toBe("end_turn");
    expect(r2.stopReason).toBe("end_turn");
  });
});

describe("A2 — mid-turn injection must not clobber the running turn's usage", () => {
  it("(f) a handed-off turn's response carries its pre-injection accumulated usage", async () => {
    // WITHOUT A2: the injected prompt's entry resets accumulatedUsage to 0 while
    // parking, so turn 1's handoff response reports zero usage (RCA §1.3).
    const t = setup();
    const p1 = t.prompt("first");
    await flush();

    t.push(assistantMsg());
    await flush();
    t.push(resultMsg()); // turn 1 accumulates usage (input_tokens 100)
    await flush();

    // Inject the second prompt AFTER turn 1 has done its work.
    const p2 = t.prompt("second");
    await flush();
    const p2uuid = t.pushed[1];

    t.push(replayOf(p2uuid)); // turn 1 hands off → response carries sessionUsage()
    await flush();
    t.push(idleMsg()); // end p2
    await flush();

    const [r1, r2] = await Promise.all([settledOr(p1), settledOr(p2)]);
    expect(r1).not.toBe(WEDGE);
    expect(r2).not.toBe(WEDGE); // the injected prompt still completes
    expect(r1.stopReason).toBe("end_turn");
    // The running turn's own accumulated usage — NOT zeroed by the injection.
    expect(r1.usage?.inputTokens).toBe(100);
    expect(r1.usage?.outputTokens).toBe(20);
  });
});
