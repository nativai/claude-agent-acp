// brick://2aa59e00 — `_meta.claudeUuid` provenance on `session/update`.
//
// Coverage map (see conception/CLAUDE-AGENT-ACP-EMITTER.md §4/§5 for the
// Category A/B distinction and the §4c trailing-correction design):
//   - Category A (consolidated assistant/user messages, replay, idle-turn)
//     stamp `_meta.claudeUuid` with the real transcript uuid.
//   - Category B (stream_event / partial messages) never stamp claudeUuid —
//     enforced structurally here: `streamEventToAcpNotifications`'s options
//     type has no `claudeUuid` field, so this is a compile-time guarantee,
//     not just a runtime one; the tests below are the behavioral mirror.
//   - The §4c trailing correction fires for text/thinking-only assistant
//     turns (whose content is otherwise fully absorbed by streaming) and
//     carries the right uuid with empty visible content.
//   - Nothing about existing notification shapes changes for a consumer
//     that doesn't look at `_meta.claudeUuid` (additive-only).
import { describe, it, expect, vi } from "vitest";
import { randomUUID } from "crypto";
import { AgentSideConnection } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent, streamEventToAcpNotifications, toAcpNotifications } from "../acp-agent.js";
import { Pushable } from "../utils.js";

vi.mock("@anthropic-ai/claude-agent-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@anthropic-ai/claude-agent-sdk")>();
  return {
    ...actual,
    getSessionMessages: vi.fn(),
  };
});
import { getSessionMessages } from "@anthropic-ai/claude-agent-sdk";

function createMockAgentWithCapture() {
  const updates: any[] = [];
  const mockClient = {
    sessionUpdate: async (notification: any) => {
      updates.push(notification);
    },
  } as unknown as AgentSideConnection;
  const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  return { agent, updates };
}

function createResultMessage() {
  return {
    type: "result" as const,
    subtype: "success" as const,
    stop_reason: "end_turn",
    is_error: false,
    result: "",
    errors: [],
    duration_ms: 0,
    duration_api_ms: 0,
    num_turns: 1,
    total_cost_usd: 0,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
    modelUsage: {},
    permission_denials: [],
    uuid: randomUUID(),
    session_id: "test-session",
  };
}

// Same injectSession shape used by the "stop reason propagation" and
// "usage_update computation" describe blocks elsewhere in this file family —
// wires a scripted async-generator as the session's `query`, then drives it
// through the real prompt loop (the actual production code path).
function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
  const input = new Pushable<any>();
  async function* messageGenerator() {
    const iter = input[Symbol.asyncIterator]();
    const { value: userMessage, done } = await iter.next();
    if (!done && userMessage) {
      yield {
        type: "user",
        message: userMessage.message,
        parent_tool_use_id: null,
        uuid: userMessage.uuid,
        session_id: "test-session",
        isReplay: true,
      };
    }
    yield* messages;
  }
  agent.sessions["test-session"] = {
    query: messageGenerator() as any,
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
  };
  (agent as any).startBackgroundReaderLoop("test-session");
}

describe("claudeUuid provenance — Category A (consolidated messages)", () => {
  it("stamps _meta.claudeUuid on a tool_call from an assistant tool_use message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const toolUseUuid = randomUUID();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: toolUseUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [
            { type: "tool_use", id: "toolu_1", name: "Bash", input: { command: "echo hi" } },
          ],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall.update._meta?.claudeUuid).toBe(toolUseUuid);
  });

  it("stamps _meta.claudeUuid on a tool_call_update from a user tool_result message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const toolUseUuid = randomUUID();
    const toolResultUuid = randomUUID();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: toolUseUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [
            { type: "tool_use", id: "toolu_2", name: "Bash", input: { command: "echo hi" } },
          ],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      {
        type: "user" as const,
        parent_tool_use_id: null,
        uuid: toolResultUuid,
        session_id: "test-session",
        message: {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_2", content: "hi\n", is_error: false },
          ],
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const toolCallUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "tool_call_update",
    );
    // The assistant tool_use in this fixture arrives only once (no prior
    // streaming encounter to refine), so this is the tool_result completion.
    expect(toolCallUpdates.length).toBeGreaterThanOrEqual(1);
    const resultUpdate = toolCallUpdates.find((u: any) => u.update.status === "completed");
    expect(resultUpdate).toBeDefined();
    expect(resultUpdate.update._meta?.claudeUuid).toBe(toolResultUuid);
  });

  it("stamps _meta.claudeUuid on replayed history (SessionMessage.uuid)", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const assistantUuid = randomUUID();
    vi.mocked(getSessionMessages).mockResolvedValue([
      {
        type: "assistant",
        uuid: assistantUuid,
        session_id: "test-session",
        parent_tool_use_id: null,
        message: {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_3", name: "Bash", input: { command: "echo hi" } },
          ],
        },
      },
    ] as any);

    await (agent as any).replaySessionHistory("test-session");

    expect(updates.length).toBeGreaterThan(0);
    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall.update._meta?.claudeUuid).toBe(assistantUuid);
  });
});

describe("claudeUuid provenance — Category B (stream events never stamp)", () => {
  it("streamEventToAcpNotifications never carries _meta.claudeUuid on a text delta", () => {
    const streamUuid = randomUUID();
    const message = {
      type: "stream_event" as const,
      parent_tool_use_id: null,
      uuid: streamUuid,
      session_id: "test-session",
      event: {
        type: "content_block_delta" as const,
        index: 0,
        delta: { type: "text_delta" as const, text: "hi" },
      },
    } as Parameters<typeof streamEventToAcpNotifications>[0];

    const result = streamEventToAcpNotifications(
      message,
      "test-session",
      {},
      { sessionUpdate: async () => {} } as unknown as Parameters<
        typeof streamEventToAcpNotifications
      >[3],
      { log: () => {}, error: () => {} },
    );

    expect(result.length).toBeGreaterThan(0);
    for (const notification of result) {
      expect((notification.update as any)._meta?.claudeUuid).toBeUndefined();
    }
  });

  it("live stream_event notifications in a real prompt turn never carry _meta.claudeUuid", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: {
          type: "content_block_start" as const,
          index: 0,
          content_block: { type: "text" as const, text: "" },
        },
      },
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: {
          type: "content_block_delta" as const,
          index: 0,
          delta: { type: "text_delta" as const, text: "partial" },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    expect(chunkUpdates.length).toBeGreaterThan(0);
    for (const u of chunkUpdates) {
      expect(u.update._meta?.claudeUuid).toBeUndefined();
    }
  });
});

describe("claudeUuid provenance — §4c trailing correction for text/thinking-only turns", () => {
  it("sends one trailing empty agent_message_chunk carrying claudeUuid for a text-only assistant message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const textUuid = randomUUID();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: textUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [{ type: "text", text: "hello world" }],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    // The real text was already delivered via streaming (not simulated here);
    // the consolidated-message handler filters text out of `content`, so the
    // ONLY agent_message_chunk this turn produces is the trailing correction.
    expect(chunkUpdates.length).toBe(1);
    expect(chunkUpdates[0].update._meta?.claudeUuid).toBe(textUuid);
    expect(chunkUpdates[0].update.content).toEqual({ type: "text", text: "" });
  });

  it("sends the trailing correction for a thinking-only assistant message too", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const thinkingUuid = randomUUID();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: thinkingUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [{ type: "thinking", thinking: "reasoning...", signature: "sig" }],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    expect(chunkUpdates.length).toBe(1);
    expect(chunkUpdates[0].update._meta?.claudeUuid).toBe(thinkingUuid);
    expect(chunkUpdates[0].update.content).toEqual({ type: "text", text: "" });
  });

  it("does NOT send a trailing correction for a pure tool_use assistant message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [
            { type: "tool_use", id: "toolu_4", name: "Bash", input: { command: "echo hi" } },
          ],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    expect(chunkUpdates.length).toBe(0);
  });

  it("still sends the trailing correction when text is mixed with a tool_use in the same message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const mixedUuid = randomUUID();
    injectSession(agent, [
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: mixedUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [
            { type: "text", text: "let me check" },
            { type: "tool_use", id: "toolu_5", name: "Bash", input: { command: "echo hi" } },
          ],
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessage(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall.update._meta?.claudeUuid).toBe(mixedUuid);

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    expect(chunkUpdates.length).toBe(1);
    expect(chunkUpdates[0].update._meta?.claudeUuid).toBe(mixedUuid);
  });

  it("fires the same trailing correction for a text-only idle-turn (teammate-triggered) message", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const idleUuid = randomUUID();
    agent.sessions["test-session"] = {
      cwd: "/test",
      taskState: new Map(),
    } as any;

    await (agent as any).handleIdleMessage(
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: idleUuid,
        session_id: "test-session",
        message: {
          model: "claude-opus-4-20250514",
          content: [{ type: "text", text: "idle turn text" }],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      "test-session",
    );

    const chunkUpdates = updates.filter(
      (u: any) => u.update?.sessionUpdate === "agent_message_chunk",
    );
    expect(chunkUpdates.length).toBe(1);
    expect(chunkUpdates[0].update._meta?.claudeUuid).toBe(idleUuid);
    expect(chunkUpdates[0].update.content).toEqual({ type: "text", text: "" });
  });
});

describe("claudeUuid provenance — additive only, no regression for callers that ignore it", () => {
  it("toAcpNotifications without a claudeUuid option produces the exact same shape as before", () => {
    const mockClient = {} as AgentSideConnection;
    const logger = { log: () => {}, error: () => {} };

    const notifications = toAcpNotifications(
      [{ type: "tool_result", tool_use_id: "toolu_x", content: "ok", is_error: false }],
      "assistant",
      "test-session",
      { toolu_x: { type: "tool_use", id: "toolu_x", name: "Bash", input: {} } } as any,
      mockClient,
      logger,
    );

    expect(notifications).toHaveLength(1);
    expect(notifications[0].update._meta).toEqual({
      claudeCode: { toolName: "Bash" },
    });
    expect((notifications[0].update as any)._meta.claudeUuid).toBeUndefined();
  });
});
