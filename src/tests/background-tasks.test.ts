// `_claude/backgroundTasks` — brick cec4c064; acpx-ui design d4c1eb3f §4.1.
//
// The SDK's `background_tasks_changed` carries the session's whole live
// background set (REPLACE semantics) but no clock. acpx-ui needs the set to tell
// a healthy wait on a background helper from a stuck turn, and a clock to show
// how long the helper has run. These rows pin the forwarded contract:
// `{ sessionId, at, tasks: [{ taskId, taskType, description, startedAt, toolName? }] }`,
// ambient tasks dropped, `startedAt` stable across payloads, the empty set sent,
// and both routes (inside a prompt, and idle between turns).

import { describe, it, expect, vi } from "vitest";
import { randomUUID } from "crypto";
import { BACKGROUND_TASKS_NOTIFICATION, ClaudeAcpAgent } from "../acp-agent.js";

const logger = { log: () => {}, error: () => {} };

type Wire = { k: "update"; n: any } | { k: "ext"; method: string; params: any };

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
  const client = {
    async sessionUpdate(n: any) {
      wire.push({ k: "update", n });
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
    accumulatedUsage: { inputTokens: 0, outputTokens: 0, cachedReadTokens: 0, cachedWriteTokens: 0 },
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
  return { agent, wire, pushed, push: mq.push };
}

const flush = () => new Promise((r) => setTimeout(r, 5));

async function feed(t: ReturnType<typeof setup>, msgs: any[]) {
  for (const m of msgs) {
    t.push(m);
    await flush();
  }
}

const sys = (subtype: string, extra: Record<string, unknown> = {}) => ({
  type: "system",
  subtype,
  session_id: "s",
  uuid: randomUUID(),
  ...extra,
});
const changed = (tasks: Array<Record<string, unknown>>) => sys("background_tasks_changed", { tasks });
const task = (id: string, type = "local_agent", description = "helper " + id, extra = {}) => ({
  task_id: id,
  task_type: type,
  description,
  ...extra,
});

const bgPayloads = (wire: Wire[]) =>
  wire.flatMap((w) => (w.k === "ext" && w.method === BACKGROUND_TASKS_NOTIFICATION ? [w.params] : []));

const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

describe("_claude/backgroundTasks (brick cec4c064)", () => {
  it("forwards the live set idle, drops ambient tasks, stamps at + startedAt", async () => {
    const t = setup();
    await feed(t, [
      changed([task("a1", "local_agent", "background helper"), task("w1", "monitor_ws", "watcher", { ambient: true })]),
    ]);
    const sent = bgPayloads(t.wire);
    expect(sent).toHaveLength(1);
    expect(sent[0].sessionId).toBe("s");
    expect(sent[0].at).toMatch(ISO);
    expect(sent[0].tasks).toEqual([
      { taskId: "a1", taskType: "local_agent", description: "background helper", startedAt: sent[0].at },
    ]);
  });

  it("replace semantics: startedAt is kept for a task still live, a new task gets its own, the empty set is sent", async () => {
    const t = setup();
    await feed(t, [changed([task("a1")])]);
    await new Promise((r) => setTimeout(r, 15));
    await feed(t, [changed([task("a1"), task("b2", "local_bash", "sleep 150")])]);
    await feed(t, [changed([task("b2", "local_bash", "sleep 150")])]);
    await feed(t, [changed([])]);
    const sent = bgPayloads(t.wire);
    expect(sent.map((p) => p.tasks.map((x: any) => x.taskId))).toEqual([["a1"], ["a1", "b2"], ["b2"], []]);
    const a1Start = sent[0].tasks[0].startedAt;
    expect(sent[1].tasks[0].startedAt).toBe(a1Start);
    expect(sent[1].tasks[1].startedAt).toBe(sent[1].at);
    expect(sent[1].tasks[1].startedAt > a1Start).toBe(true);
    expect(sent[2].tasks[0].startedAt).toBe(sent[1].tasks[1].startedAt);
    expect(sent[3]).toEqual({ sessionId: "s", at: expect.stringMatching(ISO), tasks: [] });
  });

  it("a task that leaves and returns gets a fresh startedAt", async () => {
    const t = setup();
    await feed(t, [changed([task("a1")])]);
    await feed(t, [changed([])]);
    await new Promise((r) => setTimeout(r, 15));
    await feed(t, [changed([task("a1")])]);
    const sent = bgPayloads(t.wire);
    expect(sent).toHaveLength(3);
    expect(sent[2].tasks[0].startedAt > sent[0].tasks[0].startedAt).toBe(true);
  });

  it("a change in ambient tasks only sends nothing (the control payload before it was sent)", async () => {
    const t = setup();
    await feed(t, [changed([task("a1")])]);
    await feed(t, [changed([task("a1"), task("w1", "monitor_ws", "watcher", { ambient: true })])]);
    await feed(t, [changed([task("a1")])]);
    expect(bgPayloads(t.wire)).toHaveLength(1);
  });

  it("toolName from task_started: included when it came first, re-sent with the name when it came second", async () => {
    const t = setup();
    (t.agent as any).toolUseCache["toolu_mon"] = { type: "tool_use", id: "toolu_mon", name: "Monitor", input: {} };
    (t.agent as any).toolUseCache["toolu_bash"] = { type: "tool_use", id: "toolu_bash", name: "Bash", input: {} };
    // Monitor: task_started first.
    await feed(t, [sys("task_started", { task_id: "m1", tool_use_id: "toolu_mon", description: "tail log" })]);
    await feed(t, [changed([task("m1", "local_bash", "tail log")])]);
    // Bash: the set first, then task_started.
    await feed(t, [changed([task("m1", "local_bash", "tail log"), task("b1", "local_bash", "sleep 150")])]);
    await feed(t, [sys("task_started", { task_id: "b1", tool_use_id: "toolu_bash", description: "sleep 150" })]);
    const sent = bgPayloads(t.wire);
    expect(sent.map((p) => p.tasks.map((x: any) => [x.taskId, x.toolName ?? null]))).toEqual([
      [["m1", "Monitor"]],
      [
        ["m1", "Monitor"],
        ["b1", null],
      ],
      [
        ["m1", "Monitor"],
        ["b1", "Bash"],
      ],
    ]);
    // A task_started for a task not in the live set sends nothing on its own.
    expect(sent).toHaveLength(3);
  });

  it("is forwarded from inside a running prompt too", async () => {
    const t = setup();
    const p = t.agent.prompt({ sessionId: "s", prompt: [{ type: "text", text: "go" }] });
    await flush();
    const replay = {
      type: "user",
      message: t.pushed[0].message,
      parent_tool_use_id: null,
      uuid: t.pushed[0].uuid,
      session_id: "s",
      isReplay: true,
    };
    await feed(t, [replay, changed([task("a1", "local_agent", "background helper")])]);
    expect(bgPayloads(t.wire).map((x) => x.tasks.map((y: any) => y.taskId))).toEqual([["a1"]]);
    await feed(t, [changed([]), sys("session_state_changed", { state: "idle" })]);
    await p;
    expect(bgPayloads(t.wire).map((x) => x.tasks.length)).toEqual([1, 0]);
  });
});
