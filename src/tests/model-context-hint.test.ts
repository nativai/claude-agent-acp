import { describe, it, expect, beforeEach, vi } from "vitest";
import { AgentSideConnection, SessionNotification } from "@agentclientprotocol/sdk";
import type { ModelInfo } from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAcpAgent as ClaudeAcpAgentType, effectiveRunModelId } from "../acp-agent.js";

// Regression coverage for brick 24e5df61: selecting a `[1m]` model must run a
// genuine 1M-context turn. The bug was that alias resolution stripped the
// `[1m]` suffix before it reached `query.setModel(...)`, so the Claude Code
// binary ran the base model at its 200k default (and never sent the
// `anthropic-beta: context-1m-2025-08-07` header) even though the picker/label
// promised 1M. The adapter must forward the `[1m]` form to the binary — which
// is the authority on the suffix (it strips it and toggles long_context) — and
// store it as currentModelId so the reported window is 1M.

const { registerHookCallbackSpy } = vi.hoisted(() => ({
  registerHookCallbackSpy: vi.fn(),
}));

// Mutable init-model list the mocked SDK `query` reports, so the resume-hint
// tests below can vary the session's model (and thus what the context-window
// heuristic would infer) without touching the injected-session tests above.
const { sdkInitModels } = vi.hoisted(() => ({
  sdkInitModels: {
    current: [
      {
        value: "claude-sonnet-4-6",
        displayName: "Claude Sonnet",
        description: "Fast",
        supportsAutoMode: true,
      },
    ] as unknown[],
  },
}));

vi.mock("../tools.js", async () => {
  const actual = await vi.importActual<typeof import("../tools.js")>("../tools.js");
  return {
    ...actual,
    registerHookCallback: registerHookCallbackSpy,
  };
});

// Stub the Claude Agent SDK `query` so `newSession` (→ shared `createSession`,
// the same seed path `loadSession`/resume uses) runs without a real subprocess.
vi.mock("@anthropic-ai/claude-agent-sdk", async () => {
  const actual = await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  );
  return {
    ...actual,
    query: () => ({
      initializationResult: async () => ({ models: sdkInitModels.current }),
      setModel: async () => {},
      setPermissionMode: async () => {},
      supportedCommands: async () => [],
      [Symbol.asyncIterator]: async function* () {},
    }),
  };
});

describe("effectiveRunModelId", () => {
  it("re-attaches an explicit [1m] hint the resolved base alias dropped", () => {
    // resolveModelPreference("sonnet[1m]") returns the base "sonnet" ModelInfo.
    expect(effectiveRunModelId("sonnet[1m]", "sonnet")).toBe("sonnet[1m]");
  });

  it("leaves a plain (non-hinted) pick untouched — never fabricates long-context", () => {
    expect(effectiveRunModelId("sonnet", "sonnet")).toBe("sonnet");
    expect(effectiveRunModelId("haiku", "haiku")).toBe("haiku");
    // A non-1M model applied without a hint stays a 200k pick.
    expect(effectiveRunModelId("opus", "claude-opus-4-6")).toBe("claude-opus-4-6");
  });

  it("does not double-append when the resolved base already denotes the tier", () => {
    // Bracketed form already carries the hint.
    expect(effectiveRunModelId("sonnet[1m]", "sonnet[1m]")).toBe("sonnet[1m]");
    // Concrete resolved id encodes 1m as "-1m", not "[1m]" — must NOT become
    // "claude-opus-4-6-1m[1m]" (which would defeat the binary's suffix strip).
    expect(effectiveRunModelId("opus[1m]", "claude-opus-4-6-1m")).toBe("claude-opus-4-6-1m");
  });

  it("attaches the hint to a concrete base id lacking the tier (valid bracket form)", () => {
    // "claude-sonnet-4-6[1m]" is a form the codebase already treats as valid;
    // the binary strips the bracket and enables long_context on the base id.
    expect(effectiveRunModelId("sonnet[1m]", "claude-sonnet-4-6")).toBe("claude-sonnet-4-6[1m]");
  });
});

const SESSION_ID = "test-session-id";

// Incident-shaped session: the session started on the box default (Opus 1M) and
// modelInfos holds only the BASE aliases — no "sonnet[1m]" entry — exactly the
// fork/switch path from the live incident (parent brick bbdbd56d).
const MOCK_MODES = {
  currentModeId: "default",
  availableModes: [
    { id: "auto", name: "Auto", description: "Classifier-approved permissions" },
    { id: "default", name: "Default", description: "Standard behavior" },
    { id: "plan", name: "Plan Mode", description: "Planning mode" },
  ],
};

const MODEL_INFOS: ModelInfo[] = [
  { value: "default", displayName: "Default", description: "Opus 4.8 (1M context)" },
  {
    value: "sonnet",
    displayName: "Sonnet",
    description: "Balanced",
    supportsEffort: true,
    supportedEffortLevels: ["low", "medium", "high"],
    supportsAutoMode: true,
  },
  { value: "haiku", displayName: "Haiku", description: "Fast" },
];

const MOCK_MODELS = {
  currentModelId: "default",
  availableModels: MODEL_INFOS.map((m) => ({
    modelId: m.value,
    name: m.displayName,
    description: m.description,
  })),
};

const MOCK_CONFIG_OPTIONS = [
  {
    id: "model",
    name: "Model",
    type: "select",
    category: "model",
    currentValue: "default",
    options: MOCK_MODELS.availableModels.map((m) => ({
      value: m.modelId,
      name: m.name,
      description: m.description,
    })),
  },
];

describe("1M context hint preservation (integration)", () => {
  let agent: ClaudeAcpAgentType;
  let ClaudeAcpAgent: typeof ClaudeAcpAgentType;
  let sessionUpdates: SessionNotification[];
  let setModelSpy: ReturnType<typeof vi.fn>;
  let setPermissionModeSpy: ReturnType<typeof vi.fn>;
  let applyFlagSettingsSpy: ReturnType<typeof vi.fn>;

  function createMockClient(): AgentSideConnection {
    return {
      sessionUpdate: async (notification: SessionNotification) => {
        sessionUpdates.push(notification);
      },
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as unknown as AgentSideConnection;
  }

  function populateSession() {
    setModelSpy = vi.fn();
    setPermissionModeSpy = vi.fn();
    applyFlagSettingsSpy = vi.fn();

    (agent as unknown as { sessions: Record<string, unknown> }).sessions[SESSION_ID] = {
      query: {
        setModel: setModelSpy,
        setPermissionMode: setPermissionModeSpy,
        applyFlagSettings: applyFlagSettingsSpy,
        supportedCommands: async () => [],
      },
      input: null,
      cancelled: false,
      permissionMode: "default",
      settingsManager: {},
      modes: structuredClone(MOCK_MODES),
      models: structuredClone(MOCK_MODELS),
      modelInfos: structuredClone(MODEL_INFOS),
      configOptions: structuredClone(MOCK_CONFIG_OPTIONS),
      contextWindowSize: 200000,
    };
  }

  function session() {
    return (
      agent as unknown as {
        sessions: Record<string, { models: { currentModelId: string }; contextWindowSize: number }>;
      }
    ).sessions[SESSION_ID];
  }

  beforeEach(async () => {
    sessionUpdates = [];
    registerHookCallbackSpy.mockClear();
    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;
    agent = new ClaudeAcpAgent(createMockClient());
    populateSession();
  });

  describe("unstable_setSessionModel (live picker path)", () => {
    it("forwards the [1m] hint to setModel and reports a 1M window", async () => {
      await agent.unstable_setSessionModel({ sessionId: SESSION_ID, modelId: "sonnet[1m]" });

      // The binary receives the [1m] form (it strips the suffix + enables the
      // long_context beta) instead of the stripped base "sonnet".
      expect(setModelSpy).toHaveBeenCalledWith("sonnet[1m]");
      // currentModelId keeps the [1m] form so the advertised set round-trips
      // through acpx's replay gate and the reported window is 1M.
      expect(session().models.currentModelId).toBe("sonnet[1m]");
      expect(session().contextWindowSize).toBe(1_000_000);
    });

    it("leaves a plain model application unaffected (still 200k)", async () => {
      await agent.unstable_setSessionModel({ sessionId: SESSION_ID, modelId: "sonnet" });

      expect(setModelSpy).toHaveBeenCalledWith("sonnet");
      expect(session().models.currentModelId).toBe("sonnet");
      expect(session().contextWindowSize).toBe(200000);
    });

    it("does not give a non-1M-capable pick long_context", async () => {
      await agent.unstable_setSessionModel({ sessionId: SESSION_ID, modelId: "haiku" });

      // No [1m] fabricated: the binary runs plain haiku at its 200k default.
      expect(setModelSpy).toHaveBeenCalledWith("haiku");
      expect(session().models.currentModelId).toBe("haiku");
      expect(session().contextWindowSize).toBe(200000);
    });
  });

  describe("setSessionConfigOption model branch (config-option path)", () => {
    it("forwards the [1m] hint to setModel and reports a 1M window", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "sonnet[1m]",
      });

      expect(setModelSpy).toHaveBeenCalledWith("sonnet[1m]");
      expect(session().models.currentModelId).toBe("sonnet[1m]");
      expect(session().contextWindowSize).toBe(1_000_000);
    });

    it("leaves a plain model application unaffected (still 200k)", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "sonnet",
      });

      expect(setModelSpy).toHaveBeenCalledWith("sonnet");
      expect(session().models.currentModelId).toBe("sonnet");
      expect(session().contextWindowSize).toBe(200000);
    });
  });
});

// Fix A (brick 92a994a0): a resumed session must restore the authoritative
// context window a prior run already learned, instead of re-running the
// heuristic and re-guessing 200k. acpx remembers the last authoritative `size`
// per session-model and passes it back on resume as
// `_meta.claudeCode.contextWindowSizeHint`; `createSession` (shared by
// newSession and loadSession/resume) seeds `contextWindowSize` from it, taking
// precedence over the heuristic. This is the true-source fix for the reported
// recurrence (a genuine 1M `opus` resume flashing ~172k/200k for its whole
// first post-resume turn).
describe("context window hint restoration on resume (fix A)", () => {
  let agent: ClaudeAcpAgentType;
  let ClaudeAcpAgent: typeof ClaudeAcpAgentType;

  function createMockClient(): AgentSideConnection {
    return {
      sessionUpdate: async () => {},
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as unknown as AgentSideConnection;
  }

  function windowFor(sessionId: string): number {
    return (agent as unknown as { sessions: Record<string, { contextWindowSize: number }> })
      .sessions[sessionId].contextWindowSize;
  }

  beforeEach(async () => {
    registerHookCallbackSpy.mockClear();
    // Default to a plain 200k-heuristic model so the hint is the only thing
    // that can lift the window to 1M.
    sdkInitModels.current = [
      {
        value: "claude-sonnet-4-6",
        displayName: "Claude Sonnet",
        description: "Fast",
        supportsAutoMode: true,
      },
    ];
    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;
    agent = new ClaudeAcpAgent(createMockClient());
  });

  it("seeds contextWindowSize from the restored hint even though the model heuristic would guess 200k", async () => {
    const { sessionId } = await agent.newSession({
      cwd: "/test",
      mcpServers: [],
      _meta: { claudeCode: { contextWindowSizeHint: 1_000_000 } },
    });
    // Without fix A this would be 200000 (the heuristic's guess for sonnet) —
    // exactly the wrong number the reported session showed post-resume.
    expect(windowFor(sessionId)).toBe(1_000_000);
  });

  it("without a hint, the same 200k-heuristic model falls back to the default window", async () => {
    const { sessionId } = await agent.newSession({ cwd: "/test", mcpServers: [] });
    expect(windowFor(sessionId)).toBe(200000);
  });

  it("ignores a non-positive or non-finite hint and falls back to the heuristic default", async () => {
    for (const bad of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      const { sessionId } = await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { contextWindowSizeHint: bad } },
      });
      expect(windowFor(sessionId)).toBe(200000);
    }
  });

  it("hint takes precedence over (and is consistent with) a positive 1M heuristic — idempotent restore", async () => {
    // The box-default `default` model already infers 1M from its description;
    // restoring the same 1M hint must not regress it.
    sdkInitModels.current = [
      { value: "default", displayName: "Default", description: "Opus 4.8 with 1M context" },
    ];
    const { sessionId } = await agent.newSession({
      cwd: "/test",
      mcpServers: [],
      _meta: { claudeCode: { contextWindowSizeHint: 1_000_000 } },
    });
    expect(windowFor(sessionId)).toBe(1_000_000);
  });

  it("stores the restored window tagged with its model so the replay switch can re-apply it", async () => {
    // Model-aware restore: the hint+model are captured on the session so that
    // when a resume replays the pinned model (after advertising `default`),
    // the model-switch branch re-applies the restored window instead of the
    // plain-alias heuristic. Without the model tag, that replay clobbers 1M.
    const { sessionId } = await agent.newSession({
      cwd: "/test",
      mcpServers: [],
      _meta: {
        claudeCode: { contextWindowSizeHint: 1_000_000, contextWindowSizeHintModel: "opus" },
      },
    });
    const restored = (
      agent as unknown as {
        sessions: Record<
          string,
          { restoredContextWindow: { size: number; modelId: string } | null }
        >;
      }
    ).sessions[sessionId].restoredContextWindow;
    expect(restored).toEqual({ size: 1_000_000, modelId: "opus" });
  });

  it("does not store a restored window when the hint has no model tag", async () => {
    const { sessionId } = await agent.newSession({
      cwd: "/test",
      mcpServers: [],
      _meta: { claudeCode: { contextWindowSizeHint: 1_000_000 } },
    });
    const restored = (
      agent as unknown as {
        sessions: Record<
          string,
          { restoredContextWindow: { size: number; modelId: string } | null }
        >;
      }
    ).sessions[sessionId].restoredContextWindow;
    expect(restored).toBeNull();
  });
});
