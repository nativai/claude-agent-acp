// brick 4a3c6bcb — task_progress / task_<status> updates name the tool that started the task.
//
// The adapter used to label every task update `toolName: "Agent"` with `subagentId = taskId`, so a
// background Bash or Monitor completion reached acpx and acpx-ui as a sub-agent completion. These
// rows pin: the originating tool is resolved from `toolUseCache`; the sub-agent fields exist only for
// Agent/Task; an origin that cannot be resolved keeps the legacy Agent shape (teammate tasks carry no
// tool_use_id and acpx routes their output by `subagentId`).

import { describe, it, expect } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";

const logger = { log: () => {}, error: () => {} };

function setup() {
  const updates: any[] = [];
  const client = {
    async sessionUpdate(n: any) {
      updates.push(n);
    },
    async extNotification() {},
  } as any;
  const agent = new ClaudeAcpAgent(client, logger);
  (agent as any).sessions["s"] = { cwd: "/test", taskState: new Map() };
  const cache = (agent as any).toolUseCache as Record<string, { name: string; input: unknown }>;
  const sys = (subtype: string, extra: Record<string, unknown>) =>
    (agent as any).handleIdleMessage({ type: "system", subtype, session_id: "s", ...extra }, "s");
  return { updates, cache, sys };
}

const meta = (u: any) => u.update._meta.claudeCode;

describe("task lifecycle updates resolve the originating tool", () => {
  it.each(["Bash", "Monitor"])("%s: toolName is the tool, no sub-agent fields", async (tool) => {
    const t = setup();
    t.cache["toolu_bg"] = { name: tool, input: {} };
    await t.sys("task_progress", { task_id: "b6jkc47ba", tool_use_id: "toolu_bg" });
    await t.sys("task_notification", {
      task_id: "b6jkc47ba",
      tool_use_id: "toolu_bg",
      status: "completed",
    });
    expect(t.updates).toHaveLength(2);
    for (const u of t.updates) {
      expect(u.update.toolCallId).toBe("toolu_bg");
      expect(meta(u).toolName).toBe(tool);
      expect(meta(u)).not.toHaveProperty("subagentId", "b6jkc47ba");
      expect(meta(u).subagentId).toBeUndefined();
      expect(meta(u).subagentName).toBeUndefined();
      expect(meta(u).subagentColor).toBeUndefined();
    }
    expect(meta(t.updates[0]).status).toBe("task_progress");
    expect(meta(t.updates[1]).status).toBe("task_completed");
  });

  it.each(["failed", "stopped"] as const)("Bash task_%s keeps the Bash label", async (status) => {
    const t = setup();
    t.cache["toolu_bg"] = { name: "Bash", input: {} };
    await t.sys("task_notification", { task_id: "tid", tool_use_id: "toolu_bg", status });
    expect(meta(t.updates[0])).toMatchObject({ toolName: "Bash", status: `task_${status}` });
    expect(meta(t.updates[0]).subagentId).toBeUndefined();
  });

  it("Agent: unchanged — Agent label, sub-agent fields from the spawn record", async () => {
    const t = setup();
    t.cache["toolu_a1"] = { name: "Agent", input: { name: "helper", color: "blue" } };
    await t.sys("task_started", { task_id: "a1", tool_use_id: "toolu_a1", description: "d" });
    await t.sys("task_progress", {
      task_id: "a1",
      tool_use_id: "toolu_a1",
      last_tool_name: "Bash",
    });
    await t.sys("task_notification", {
      task_id: "a1",
      tool_use_id: "toolu_a1",
      status: "completed",
    });
    const [spawned, progress, done] = t.updates.map(meta);
    expect(spawned).toMatchObject({
      toolName: "Agent",
      status: "teammate_spawned",
      subagentId: "a1",
    });
    expect(progress).toMatchObject({
      toolName: "Agent",
      status: "task_progress",
      subagentId: "a1",
      subagentName: "helper",
      subagentColor: "blue",
      taskLastToolName: "Bash",
    });
    expect(done).toMatchObject({
      toolName: "Agent",
      status: "task_completed",
      subagentId: "a1",
      subagentName: "helper",
      subagentColor: "blue",
    });
  });

  it("Task (the legacy sub-agent tool name) keeps its sub-agent fields", async () => {
    const t = setup();
    t.cache["toolu_t1"] = { name: "Task", input: { description: "legacy" } };
    await t.sys("task_notification", {
      task_id: "t1",
      tool_use_id: "toolu_t1",
      status: "completed",
    });
    expect(meta(t.updates[0])).toMatchObject({ toolName: "Task", subagentId: "t1" });
  });

  it("unresolvable origin keeps the legacy Agent shape (teammate task without tool_use_id; id absent from the cache)", async () => {
    const t = setup();
    await t.sys("task_notification", { task_id: "mate-1", status: "completed" });
    await t.sys("task_notification", {
      task_id: "gone",
      tool_use_id: "toolu_unknown",
      status: "stopped",
    });
    expect(meta(t.updates[0])).toMatchObject({ toolName: "Agent", subagentId: "mate-1" });
    expect(t.updates[0].update.toolCallId).toBe("mate-1");
    expect(meta(t.updates[1])).toMatchObject({ toolName: "Agent", subagentId: "gone" });
  });
});
