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

vi.mock("../tools.js", async () => {
  const actual = await vi.importActual<typeof import("../tools.js")>("../tools.js");
  return {
    ...actual,
    registerHookCallback: registerHookCallbackSpy,
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
