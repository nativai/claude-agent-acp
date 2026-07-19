import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  AgentSideConnection,
  SessionNotification,
  type SessionModelState,
} from "@agentclientprotocol/sdk";
import type { ModelInfo } from "@anthropic-ai/claude-agent-sdk";
import type { ClaudeAcpAgent as ClaudeAcpAgentType } from "../acp-agent.js";
import { injectFableModel, injectOpusModel } from "../acp-agent.js";

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

const SESSION_ID = "test-session-id";

const MOCK_MODES = {
  currentModeId: "default",
  availableModes: [
    { id: "default", name: "Default", description: "Standard behavior" },
    { id: "plan", name: "Plan Mode", description: "Planning mode" },
    { id: "acceptEdits", name: "Accept Edits", description: "Auto-accept edits" },
  ],
};

const MOCK_MODELS = {
  currentModelId: "claude-opus-4-5",
  availableModels: [
    { modelId: "claude-opus-4-5", name: "Claude Opus", description: "Most capable" },
    { modelId: "claude-sonnet-4-6", name: "Claude Sonnet", description: "Balanced" },
  ],
};

const MOCK_CONFIG_OPTIONS = [
  {
    id: "mode",
    name: "Mode",
    type: "select",
    category: "mode",
    currentValue: "default",
    options: MOCK_MODES.availableModes.map((m) => ({
      value: m.id,
      name: m.name,
      description: m.description,
    })),
  },
  {
    id: "model",
    name: "Model",
    type: "select",
    category: "model",
    currentValue: "claude-opus-4-5",
    options: MOCK_MODELS.availableModels.map((m) => ({
      value: m.modelId,
      name: m.name,
      description: m.description,
    })),
  },
  {
    id: "effort",
    name: "Effort",
    description: "Available effort levels for this model",
    type: "select",
    category: "effort",
    currentValue: "default",
    options: [
      { value: "default", name: "Default" },
      { value: "low", name: "Low" },
      { value: "medium", name: "Medium" },
      { value: "high", name: "High" },
    ],
  },
];

describe("session config options", () => {
  let agent: ClaudeAcpAgentType;
  let ClaudeAcpAgent: typeof ClaudeAcpAgentType;
  let sessionUpdates: SessionNotification[];
  let createSessionSpy: ReturnType<typeof vi.fn>;
  let setPermissionModeSpy: ReturnType<typeof vi.fn>;
  let setModelSpy: ReturnType<typeof vi.fn>;
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
    setPermissionModeSpy = vi.fn();
    setModelSpy = vi.fn();
    applyFlagSettingsSpy = vi.fn();

    (agent as unknown as { sessions: Record<string, unknown> }).sessions[SESSION_ID] = {
      query: {
        setPermissionMode: setPermissionModeSpy,
        setModel: setModelSpy,
        applyFlagSettings: applyFlagSettingsSpy,
        supportedCommands: async () => [],
      },
      input: null,
      cancelled: false,
      permissionMode: "default",
      settingsManager: {},
      modes: structuredClone(MOCK_MODES),
      models: structuredClone(MOCK_MODELS),
      modelInfos: MOCK_MODELS.availableModels.map(
        (m): ModelInfo => ({
          value: m.modelId,
          displayName: m.name,
          description: m.description,
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        }),
      ),
      configOptions: structuredClone(MOCK_CONFIG_OPTIONS),
      contextWindowSize: 200000,
    };
  }

  beforeEach(async () => {
    sessionUpdates = [];
    registerHookCallbackSpy.mockClear();

    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;

    agent = new ClaudeAcpAgent(createMockClient());
    createSessionSpy = vi.fn(async () => ({
      sessionId: SESSION_ID,
      modes: MOCK_MODES,
      models: MOCK_MODELS,
      configOptions: MOCK_CONFIG_OPTIONS,
    }));
    (agent as unknown as { createSession: typeof createSessionSpy }).createSession =
      createSessionSpy;
  });

  describe("newSession returns configOptions", () => {
    it("includes configOptions in the response", async () => {
      const response = await agent.newSession({ cwd: "/test", mcpServers: [] });
      expect(response.configOptions).toBeDefined();
      expect(response.configOptions).toEqual(MOCK_CONFIG_OPTIONS);
    });

    it("includes mode and model config options", async () => {
      const response = await agent.newSession({ cwd: "/test", mcpServers: [] });
      const modeOption = response.configOptions?.find((o) => o.id === "mode");
      const modelOption = response.configOptions?.find((o) => o.id === "model");
      expect(modeOption).toBeDefined();
      expect(modelOption).toBeDefined();
    });
  });

  describe("loadSession returns configOptions", () => {
    it("includes configOptions from createSession", async () => {
      // loadSession calls findSessionFile first - override the whole method
      const loadSessionSpy = vi.fn(async () => ({
        modes: MOCK_MODES,
        models: MOCK_MODELS,
        configOptions: MOCK_CONFIG_OPTIONS,
      }));
      (agent as unknown as { loadSession: typeof loadSessionSpy }).loadSession = loadSessionSpy;

      const response = await agent.loadSession({
        cwd: "/test",
        sessionId: SESSION_ID,
        mcpServers: [],
      });
      expect(response.configOptions).toEqual(MOCK_CONFIG_OPTIONS);
    });
  });

  describe("setSessionConfigOption", () => {
    beforeEach(() => {
      populateSession();
    });

    it("throws when session not found", async () => {
      await expect(
        agent.setSessionConfigOption({
          sessionId: "nonexistent",
          configId: "mode",
          value: "plan",
        }),
      ).rejects.toThrow("Session not found");
    });

    it("throws when config option not found", async () => {
      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "unknown-option",
          value: "some-value",
        }),
      ).rejects.toThrow("Unknown config option: unknown-option");
    });

    it("throws when value is not valid for the option", async () => {
      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "mode",
          value: "invalid-mode",
        }),
      ).rejects.toThrow("Invalid value for config option mode: invalid-mode");
    });

    it("changes mode, sends current_mode_update but not config_option_update", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "mode",
        value: "plan",
      });

      expect(setPermissionModeSpy).toHaveBeenCalledWith("plan");

      const modeUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "current_mode_update",
      );
      expect(modeUpdate?.update).toMatchObject({
        sessionUpdate: "current_mode_update",
        currentModeId: "plan",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeUndefined();
    });

    it("changes model and does not send a config_option_update notification", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-sonnet-4-6");

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeUndefined();
    });

    it("resolves model alias 'opus' to full model ID", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "opus",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-opus-4-5");

      const modelOption = response.configOptions.find((o) => o.id === "model");
      expect(modelOption?.currentValue).toBe("claude-opus-4-5");
    });

    it("resolves model alias 'sonnet' to full model ID", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "sonnet",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-sonnet-4-6");
    });

    it("resolves display name to model ID", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "Claude Sonnet",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-sonnet-4-6");
    });

    it("still works with exact model ID", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-sonnet-4-6");
      const modelOption = response.configOptions.find((o) => o.id === "model");
      expect(modelOption?.currentValue).toBe("claude-sonnet-4-6");
    });

    it("throws for completely invalid model value", async () => {
      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "model",
          value: "gpt-4",
        }),
      ).rejects.toThrow("Invalid value for config option model: gpt-4");
    });

    it("returns full configOptions in the response", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "mode",
        value: "plan",
      });

      expect(response.configOptions).toHaveLength(MOCK_CONFIG_OPTIONS.length);
      const modeOption = response.configOptions.find((o) => o.id === "mode");
      expect(modeOption?.currentValue).toBe("plan");
    });

    it("other options are unchanged when one is updated", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "mode",
        value: "plan",
      });

      const modelOption = response.configOptions.find((o) => o.id === "model");
      expect(modelOption?.currentValue).toBe("claude-opus-4-5");
    });
  });

  describe("setSessionMode sends config_option_update", () => {
    beforeEach(() => {
      populateSession();
    });

    it("sends config_option_update when mode is changed via setSessionMode", async () => {
      await agent.setSessionMode({ sessionId: SESSION_ID, modeId: "acceptEdits" });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeDefined();
      expect(configUpdate?.update).toMatchObject({
        sessionUpdate: "config_option_update",
        configOptions: expect.arrayContaining([
          expect.objectContaining({ id: "mode", currentValue: "acceptEdits" }),
        ]),
      });
    });

    it("updates stored configOptions currentValue when mode changes", async () => {
      await agent.setSessionMode({ sessionId: SESSION_ID, modeId: "plan" });

      const session = (
        agent as unknown as {
          sessions: Record<string, { configOptions: typeof MOCK_CONFIG_OPTIONS }>;
        }
      ).sessions[SESSION_ID];
      const modeOption = session.configOptions.find((o) => o.id === "mode");
      expect(modeOption?.currentValue).toBe("plan");
    });

    it("does not send config_option_update for an invalid mode", async () => {
      await expect(
        agent.setSessionMode({ sessionId: SESSION_ID, modeId: "not-a-mode" as any }),
      ).rejects.toThrow("Invalid Mode");

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeUndefined();
    });
  });

  describe("unstable_setSessionModel sends config_option_update", () => {
    beforeEach(() => {
      populateSession();
    });

    it("sends config_option_update when model is changed via setSessionModel", async () => {
      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeDefined();
      expect(configUpdate?.update).toMatchObject({
        sessionUpdate: "config_option_update",
        configOptions: expect.arrayContaining([
          expect.objectContaining({ id: "model", currentValue: "claude-sonnet-4-6" }),
        ]),
      });
    });

    it("updates stored configOptions currentValue when model changes", async () => {
      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const session = (
        agent as unknown as {
          sessions: Record<string, { configOptions: typeof MOCK_CONFIG_OPTIONS }>;
        }
      ).sessions[SESSION_ID];
      const modelOption = session.configOptions.find((o) => o.id === "model");
      expect(modelOption?.currentValue).toBe("claude-sonnet-4-6");
    });

    it("includes updated effort in config_option_update when model drops effort support", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: false,
        },
      ];

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeDefined();
      const effortOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "effort",
      );
      expect(effortOption).toBeUndefined();
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({ env: null, effortLevel: null });
    });

    it("clamps effort in config_option_update when new model has different supported levels", async () => {
      // Set current effort to "max" which the new model won't support
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      if (effortOpt) effortOpt.currentValue = "max";

      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high", "max"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
      ];

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      const effortOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "effort",
      );
      expect(effortOption).toBeDefined();
      expect(effortOption.currentValue).toBe("default");
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({ env: null, effortLevel: null });
    });

    it("preserves effort in config_option_update when new model supports same level", async () => {
      // Set effort to "low" first
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      if (effortOpt) effortOpt.currentValue = "low";

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      const effortOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "effort",
      );
      expect(effortOption?.currentValue).toBe("low");
      // Effort didn't change, so applyFlagSettings should NOT be called
      expect(applyFlagSettingsSpy).not.toHaveBeenCalled();
    });
  });

  describe("no config_option_update notification when using setSessionConfigOption", () => {
    beforeEach(() => {
      populateSession();
    });

    it("sends no config_option_update when setting mode via config option", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "mode",
        value: "plan",
      });

      const configUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdates).toHaveLength(0);
    });

    it("sends no config_option_update when setting model via config option", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      const configUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdates).toHaveLength(0);
    });
  });

  describe("setSessionConfigOption for effort", () => {
    beforeEach(() => {
      populateSession();
    });

    it("calls applyFlagSettings with effortLevel", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });

      // The pinned effort now rides the FLAG-tier `env` block (authoritative,
      // beats the box's re-applied user-tier CLAUDE_CODE_EFFORT_LEVEL); the
      // effortLevel key is kept coherent for in-union values. (brick 5f35da58)
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({
        env: { CLAUDE_CODE_EFFORT_LEVEL: "low" },
        effortLevel: "low",
      });
    });

    it("routes an out-of-union 'max' pin through the env path only (no effortLevel key)", async () => {
      // Arrange: make "max" a selectable effort level for this session.
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      effortOpt.options.push({ value: "max", name: "Max" });

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "max",
      });

      // "max" is outside the SDK `effortLevel` union, so it must ride ONLY the
      // authoritative env path — the effortLevel key is omitted, never set to a
      // bogus out-of-union "max". This is precisely what makes a
      // `--reasoning-effort max` pin win over the box's user-tier
      // CLAUDE_CODE_EFFORT_LEVEL=high (the hostile ambient env is exercised
      // end-to-end in the self-test). (brick 5f35da58)
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({
        env: { CLAUDE_CODE_EFFORT_LEVEL: "max" },
      });
      const calls = applyFlagSettingsSpy.mock.calls;
      const lastCall = calls[calls.length - 1]?.[0];
      expect(lastCall).not.toHaveProperty("effortLevel");
    });

    it("sets both env and effortLevel for an in-union 'xhigh' pin", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      effortOpt.options.push({ value: "xhigh", name: "Extra High" });

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "xhigh",
      });

      // "xhigh" is in the SDK union, so both channels are set: the env block
      // (authoritative) and the effortLevel key (label/clamp coherence).
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({
        env: { CLAUDE_CODE_EFFORT_LEVEL: "xhigh" },
        effortLevel: "xhigh",
      });
    });

    it("calls applyFlagSettings with null effortLevel for 'default'", async () => {
      // Set effort to a non-default value first
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      if (effortOpt) effortOpt.currentValue = "high";

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "default",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({ env: null, effortLevel: null });

      // The SDK's applyFlagSettings travels over a JSON pipe and only clears a
      // flag-layer key when an explicit `null` is sent — `undefined` is
      // dropped during JSON.stringify, which would leave the previous effort
      // override in place. Round-trip the call args through JSON to make sure
      // the key actually reaches the SDK.
      const calls = applyFlagSettingsSpy.mock.calls;
      const lastCallArgs = calls[calls.length - 1]?.[0];
      const serialized = JSON.parse(JSON.stringify(lastCallArgs));
      expect(serialized).toHaveProperty("effortLevel", null);
    });

    it("updates effort currentValue in returned configOptions", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "medium",
      });

      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption?.currentValue).toBe("medium");
    });

    it("throws for invalid effort value", async () => {
      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "effort",
          value: "turbo",
        }),
      ).rejects.toThrow("Invalid value for config option effort: turbo");
    });

    it("does not send config_option_update notification", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });

      const configUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdates).toHaveLength(0);
    });

    it("other options are unchanged when effort is updated", async () => {
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });

      const modeOption = response.configOptions.find((o) => o.id === "mode");
      expect(modeOption?.currentValue).toBe("default");
      const modelOption = response.configOptions.find((o) => o.id === "model");
      expect(modelOption?.currentValue).toBe("claude-opus-4-5");
    });
  });

  describe("effort level and model switch interactions", () => {
    beforeEach(() => {
      populateSession();
    });

    it("drops effort option when switching to a model without effort support", async () => {
      // Make sonnet not support effort
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: false,
        },
      ];

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeUndefined();
    });

    it("clears effort via applyFlagSettings when switching to a model without effort", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: false,
        },
      ];

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({ env: null, effortLevel: null });
    });

    it("adds effort option when switching to a model that supports effort", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      // Start with sonnet (no effort) as current
      session.models = { ...session.models, currentModelId: "claude-sonnet-4-6" };
      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: false,
        },
      ];
      // Remove effort from current config options
      session.configOptions = session.configOptions.filter((o: any) => o.id !== "effort");

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-opus-4-5",
      });

      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      // No previous effort, so defaults to "default" (no effort override)
      expect(effortOption?.currentValue).toBe("default");
    });

    it("clamps effort to valid value when new model has different supported levels", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      // Set current effort to "max" (not supported by sonnet in our mock)
      const effortOpt = session.configOptions.find((o: any) => o.id === "effort");
      if (effortOpt) effortOpt.currentValue = "max";

      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high", "max"],
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
        },
      ];

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      // "max" is not in sonnet's levels, so should fall back to "default" (no effort override)
      expect(effortOption?.currentValue).toBe("default");
      // SDK should be told to clear the effort override
      expect(applyFlagSettingsSpy).toHaveBeenCalledWith({ env: null, effortLevel: null });
    });

    it("preserves effort value when new model supports the same level", async () => {
      // Set effort to "low"
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });

      // Switch model — both support "low"
      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption?.currentValue).toBe("low");
      // applyFlagSettings was called once for the effort change, but not again for the model switch
      expect(applyFlagSettingsSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("bidirectional consistency", () => {
    beforeEach(() => {
      populateSession();
    });

    it("setSessionConfigOption for mode also calls underlying setPermissionMode", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "mode",
        value: "acceptEdits",
      });

      expect(setPermissionModeSpy).toHaveBeenCalledWith("acceptEdits");
    });

    it("setSessionConfigOption for model also calls underlying setModel", async () => {
      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      expect(setModelSpy).toHaveBeenCalledWith("claude-sonnet-4-6");
    });

    it("setSessionMode also syncs configOptions", async () => {
      await agent.setSessionMode({ sessionId: SESSION_ID, modeId: "plan" });

      const session = (
        agent as unknown as {
          sessions: Record<string, { configOptions: typeof MOCK_CONFIG_OPTIONS }>;
        }
      ).sessions[SESSION_ID];
      expect(session.configOptions.find((o) => o.id === "mode")?.currentValue).toBe("plan");
    });

    it("setSessionModel also syncs configOptions", async () => {
      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-sonnet-4-6",
      });

      const session = (
        agent as unknown as {
          sessions: Record<string, { configOptions: typeof MOCK_CONFIG_OPTIONS }>;
        }
      ).sessions[SESSION_ID];
      expect(session.configOptions.find((o) => o.id === "model")?.currentValue).toBe(
        "claude-sonnet-4-6",
      );
    });
  });

  describe("auto mode availability per model", () => {
    /**
     * Augment the session populated by `populateSession()` with a Haiku entry
     * (no `supportsAutoMode`), Opus + Sonnet entries with `supportsAutoMode:
     * true`, and seed `availableModes` so it currently includes `auto`. This
     * exercises the per-model recomputation done by `applyConfigOptionValue`
     * on a model switch.
     */
    function setupHaikuOpusSession(currentModeId: string = "default") {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modelInfos = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          supportsAutoMode: true,
        },
        {
          value: "claude-sonnet-4-6",
          displayName: "Claude Sonnet",
          description: "Balanced",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          supportsAutoMode: true,
        },
        {
          value: "claude-haiku-4-5",
          displayName: "Claude Haiku",
          description: "Fast",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          // supportsAutoMode intentionally omitted
        },
      ];
      session.models = {
        currentModelId: "claude-opus-4-5",
        availableModels: [
          { modelId: "claude-opus-4-5", name: "Claude Opus", description: "Most capable" },
          { modelId: "claude-sonnet-4-6", name: "Claude Sonnet", description: "Balanced" },
          { modelId: "claude-haiku-4-5", name: "Claude Haiku", description: "Fast" },
        ],
      };
      session.modes = {
        currentModeId,
        availableModes: [
          {
            id: "auto",
            name: "Auto",
            description: "Use a model classifier to approve/deny permission prompts",
          },
          {
            id: "default",
            name: "Default",
            description: "Standard behavior, prompts for dangerous operations",
          },
          {
            id: "acceptEdits",
            name: "Accept Edits",
            description: "Auto-accept file edit operations",
          },
          { id: "plan", name: "Plan Mode", description: "Planning mode" },
          {
            id: "dontAsk",
            name: "Don't Ask",
            description: "Don't prompt for permissions, deny if not pre-approved",
          },
        ],
      };
      // Reflect the seeded availableModes/availableModels in configOptions so
      // the pre-state matches what `createSession` would have produced for
      // Opus, and `setSessionConfigOption` validation can accept the seeded
      // model ids (notably the new Haiku entry).
      session.configOptions = session.configOptions.map((o: any) => {
        if (o.id === "mode") {
          return {
            ...o,
            currentValue: currentModeId,
            options: session.modes.availableModes.map((m: any) => ({
              value: m.id,
              name: m.name,
              description: m.description,
            })),
          };
        }
        if (o.id === "model") {
          return {
            ...o,
            currentValue: session.models.currentModelId,
            options: session.models.availableModels.map((m: any) => ({
              value: m.modelId,
              name: m.name,
              description: m.description,
            })),
          };
        }
        return o;
      });
      return session;
    }

    beforeEach(() => {
      populateSession();
    });

    it("drops `auto` from available modes when switching to Haiku", async () => {
      setupHaikuOpusSession("default");

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-haiku-4-5",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeDefined();
      const modeOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "mode",
      );
      expect(modeOption).toBeDefined();
      const modeValues = modeOption.options.map((o: any) => o.value);
      expect(modeValues).not.toContain("auto");
      expect(modeValues).toEqual(
        expect.arrayContaining(["default", "acceptEdits", "plan", "dontAsk"]),
      );
    });

    it("clamps to `default` and emits current_mode_update when Opus(auto) → Haiku", async () => {
      setupHaikuOpusSession("auto");

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-haiku-4-5",
      });

      // SDK was synced to "default".
      expect(setPermissionModeSpy).toHaveBeenCalledWith("default");

      // current_mode_update was emitted before config_option_update so a
      // client applying notifications in order observes the mode change
      // before re-rendering the config-option list.
      const modeUpdateIdx = sessionUpdates.findIndex(
        (n) => n.update.sessionUpdate === "current_mode_update",
      );
      const configUpdateIdx = sessionUpdates.findIndex(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(modeUpdateIdx).toBeGreaterThanOrEqual(0);
      expect(configUpdateIdx).toBeGreaterThanOrEqual(0);
      expect(modeUpdateIdx).toBeLessThan(configUpdateIdx);
      expect((sessionUpdates[modeUpdateIdx].update as any).currentModeId).toBe("default");

      // configOptions reflect the clamped mode.
      const modeOption = (sessionUpdates[configUpdateIdx].update as any).configOptions.find(
        (o: any) => o.id === "mode",
      );
      expect(modeOption.currentValue).toBe("default");
    });

    it("re-adds `auto` when switching from Haiku back to Opus", async () => {
      const session = setupHaikuOpusSession("default");
      // Pretend Haiku is the current model with no `auto`.
      session.models.currentModelId = "claude-haiku-4-5";
      session.modes.availableModes = session.modes.availableModes.filter(
        (m: any) => m.id !== "auto",
      );
      const modeOpt = session.configOptions.find((o: any) => o.id === "mode");
      modeOpt.options = session.modes.availableModes.map((m: any) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      }));

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-opus-4-5",
      });

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdate).toBeDefined();
      const modeOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "mode",
      );
      const modeValues = modeOption.options.map((o: any) => o.value);
      expect(modeValues).toContain("auto");

      // The current mode ("default") is still valid on Opus, so no
      // current_mode_update should have been emitted by the model switch.
      const modeUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "current_mode_update",
      );
      expect(modeUpdates).toHaveLength(0);
    });

    it("preserves the current mode when it remains valid after a model switch", async () => {
      setupHaikuOpusSession("plan");

      await agent.unstable_setSessionModel({
        sessionId: SESSION_ID,
        modelId: "claude-haiku-4-5",
      });

      // `plan` is in availableModes for both Opus and Haiku, so no clamp.
      expect(setPermissionModeSpy).not.toHaveBeenCalledWith("default");

      const modeUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "current_mode_update",
      );
      expect(modeUpdates).toHaveLength(0);

      const configUpdate = sessionUpdates.find(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      const modeOption = (configUpdate?.update as any).configOptions.find(
        (o: any) => o.id === "mode",
      );
      expect(modeOption.currentValue).toBe("plan");
    });

    it("clamps mode and emits current_mode_update via setSessionConfigOption(model)", async () => {
      // Mirrors the unstable_setSessionModel(auto → Haiku) test, but goes
      // through the request/response API. The `current_mode_update` side
      // effect must still fire so clients learn about the clamp regardless of
      // which entry point triggered the model switch.
      setupHaikuOpusSession("auto");

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-haiku-4-5",
      });

      expect(setPermissionModeSpy).toHaveBeenCalledWith("default");

      const modeUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "current_mode_update",
      );
      expect(modeUpdates).toHaveLength(1);
      expect((modeUpdates[0].update as any).currentModeId).toBe("default");

      // setSessionConfigOption is a request/response API: it returns the new
      // configOptions in the response rather than emitting a
      // config_option_update notification.
      const configUpdates = sessionUpdates.filter(
        (n) => n.update.sessionUpdate === "config_option_update",
      );
      expect(configUpdates).toHaveLength(0);

      const modeOption = response.configOptions.find((o: any) => o.id === "mode");
      expect(modeOption).toBeDefined();
      expect((modeOption as any).currentValue).toBe("default");
      expect((modeOption as any).options.map((o: any) => o.value)).not.toContain("auto");
    });

    it("rejects direct setSessionMode to `auto` when the active model does not offer it", async () => {
      const session = setupHaikuOpusSession("default");
      session.models.currentModelId = "claude-haiku-4-5";
      session.modes.availableModes = session.modes.availableModes.filter(
        (mode: any) => mode.id !== "auto",
      );

      await expect(agent.setSessionMode({ sessionId: SESSION_ID, modeId: "auto" })).rejects.toThrow(
        "Mode auto is not available in this session",
      );

      expect(setPermissionModeSpy).not.toHaveBeenCalledWith("auto");
      expect(sessionUpdates).toHaveLength(0);
    });
  });

  describe("ExitPlanMode permission options filtered by availableModes", () => {
    let capturedPermissionRequest: any;
    let permissionResponse: any;

    beforeEach(() => {
      capturedPermissionRequest = null;
      permissionResponse = { outcome: { outcome: "cancelled" } };
      // Replace the default mock client with one that captures the
      // requestPermission call so we can assert on the offered options.
      (agent as any).client = {
        sessionUpdate: async (notification: SessionNotification) => {
          sessionUpdates.push(notification);
        },
        requestPermission: async (params: any) => {
          capturedPermissionRequest = params;
          return permissionResponse;
        },
        readTextFile: async () => ({ content: "" }),
        writeTextFile: async () => ({}),
      };
      populateSession();
    });

    it("omits the `auto` option on a model without supportsAutoMode", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      // Haiku-shaped session: availableModes does NOT include `auto`.
      session.modes = {
        currentModeId: "plan",
        availableModes: [
          { id: "default", name: "Default", description: "Standard" },
          { id: "acceptEdits", name: "Accept Edits", description: "Auto-accept edits" },
          { id: "plan", name: "Plan Mode", description: "Planning mode" },
          { id: "dontAsk", name: "Don't Ask", description: "Deny if not pre-approved" },
        ],
      };

      const canUseTool = (agent as any).canUseTool(SESSION_ID);
      const signal = new AbortController().signal;
      try {
        await canUseTool(
          "ExitPlanMode",
          { plan: "do stuff" },
          { signal, suggestions: undefined, toolUseID: "toolu_1" },
        );
      } catch {
        // The mock client returns `cancelled`, which makes canUseTool throw.
        // We only care about the captured requestPermission options.
      }

      expect(capturedPermissionRequest).not.toBeNull();
      const optionIds = capturedPermissionRequest.options.map((o: any) => o.optionId);
      expect(optionIds).not.toContain("auto");
      expect(optionIds).toEqual(expect.arrayContaining(["default", "acceptEdits", "plan"]));
    });

    it("denies a selected `auto` option if the client did not receive that option", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modes = {
        currentModeId: "plan",
        availableModes: [
          { id: "default", name: "Default", description: "Standard" },
          { id: "acceptEdits", name: "Accept Edits", description: "Auto-accept edits" },
          { id: "plan", name: "Plan Mode", description: "Planning mode" },
          { id: "dontAsk", name: "Don't Ask", description: "Deny if not pre-approved" },
        ],
      };
      permissionResponse = { outcome: { outcome: "selected", optionId: "auto" } };

      const canUseTool = (agent as any).canUseTool(SESSION_ID);
      const result = await canUseTool(
        "ExitPlanMode",
        { plan: "do stuff" },
        { signal: new AbortController().signal, suggestions: undefined, toolUseID: "toolu_2" },
      );

      expect(capturedPermissionRequest).not.toBeNull();
      const optionIds = capturedPermissionRequest.options.map((o: any) => o.optionId);
      expect(optionIds).not.toContain("auto");
      expect(result.behavior).toBe("deny");
      expect(sessionUpdates).toHaveLength(0);
    });

    it("includes the `auto` option on a model with supportsAutoMode", async () => {
      const session = (agent as unknown as { sessions: Record<string, any> }).sessions[SESSION_ID];
      session.modes = {
        currentModeId: "plan",
        availableModes: [
          { id: "auto", name: "Auto", description: "Use a model classifier" },
          { id: "default", name: "Default", description: "Standard" },
          { id: "acceptEdits", name: "Accept Edits", description: "Auto-accept edits" },
          { id: "plan", name: "Plan Mode", description: "Planning mode" },
          { id: "dontAsk", name: "Don't Ask", description: "Deny if not pre-approved" },
        ],
      };

      const canUseTool = (agent as any).canUseTool(SESSION_ID);
      const signal = new AbortController().signal;
      try {
        await canUseTool(
          "ExitPlanMode",
          { plan: "do stuff" },
          { signal, suggestions: undefined, toolUseID: "toolu_3" },
        );
      } catch {
        // mock returns cancelled
      }

      expect(capturedPermissionRequest).not.toBeNull();
      const optionIds = capturedPermissionRequest.options.map((o: any) => o.optionId);
      expect(optionIds).toContain("auto");
    });
  });

  // Regression repro for brick 2a928fd7: switching a running session Sonnet→Fable
  // (or →Opus) then setting thinking depth high→medium/low threw "Unknown config
  // option: effort" (surfaced to the user as "internal error"). Root cause: the
  // adapter INJECTS fable/opus into the advertised model list (the SDK never
  // surfaces fable, and opus only when the resolved list lacks it) with a PARTIAL
  // ModelInfo that omitted `supportsEffort`/`supportedEffortLevels`. After the
  // mid-session switch, buildConfigOptions dropped the effort option, and
  // setSessionConfigOption then threw. These tests drive the REAL injection
  // functions so they are coupled to production: a partial (effort-less) injected
  // ModelInfo makes them red; the capability fix makes them green with no test
  // change. Fable 5 / Opus 4.x support the full ladder (claude-api skill), so the
  // fix is to advertise it — there is no legible-restriction path.
  const FULL_EFFORT_LADDER = ["low", "medium", "high", "xhigh", "max"];

  describe("injected fable/opus models carry effort capability (brick 2a928fd7)", () => {
    // Build a session whose advertised model list was produced by the REAL
    // injection function `inject`, starting from an effort-capable Opus-4.5 base
    // (so the effort option is present pre-switch, exactly as createSession would
    // have produced it).
    function populateInjectedSession(inject: typeof injectFableModel) {
      setPermissionModeSpy = vi.fn();
      setModelSpy = vi.fn();
      applyFlagSettingsSpy = vi.fn();

      const baseModelInfos: ModelInfo[] = [
        {
          value: "claude-opus-4-5",
          displayName: "Claude Opus 4.5",
          description: "Most capable",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high", "xhigh", "max"],
        },
      ];
      const baseState: SessionModelState = {
        currentModelId: "claude-opus-4-5",
        availableModels: [
          { modelId: "claude-opus-4-5", name: "Claude Opus 4.5", description: "Most capable" },
        ],
      };

      // Drive the REAL injection to add the fable/opus entry to the advertised list.
      const { state, modelInfos } = inject(baseState, baseModelInfos);

      const configOptions = [
        {
          id: "mode",
          name: "Mode",
          type: "select",
          category: "mode",
          currentValue: "default",
          options: MOCK_MODES.availableModes.map((m) => ({
            value: m.id,
            name: m.name,
            description: m.description,
          })),
        },
        {
          id: "model",
          name: "Model",
          type: "select",
          category: "model",
          currentValue: "claude-opus-4-5",
          options: state.availableModels.map((m) => ({
            value: m.modelId,
            name: m.name,
            description: m.description,
          })),
        },
        {
          id: "effort",
          name: "Effort",
          type: "select",
          category: "thought_level",
          currentValue: "default",
          options: [
            { value: "default", name: "Default" },
            { value: "low", name: "Low" },
            { value: "medium", name: "Medium" },
            { value: "high", name: "High" },
            { value: "xhigh", name: "Xhigh" },
            { value: "max", name: "Max" },
          ],
        },
      ];

      (agent as unknown as { sessions: Record<string, unknown> }).sessions[SESSION_ID] = {
        query: {
          setPermissionMode: setPermissionModeSpy,
          setModel: setModelSpy,
          applyFlagSettings: applyFlagSettingsSpy,
          supportedCommands: async () => [],
        },
        input: null,
        cancelled: false,
        permissionMode: "default",
        settingsManager: {},
        modes: structuredClone(MOCK_MODES),
        models: structuredClone(state),
        modelInfos,
        configOptions,
        contextWindowSize: 200000,
      };
    }

    it("injectFableModel advertises the full effort ladder", () => {
      const { modelInfos } = injectFableModel(
        { currentModelId: "claude-opus-4-5", availableModels: [] },
        [],
      );
      const fable = modelInfos.find((m) => m.value === "fable");
      expect(fable?.supportsEffort).toBe(true);
      expect(fable?.supportedEffortLevels).toEqual(FULL_EFFORT_LADDER);
    });

    it("injectOpusModel advertises the full effort ladder", () => {
      const { modelInfos } = injectOpusModel(
        { currentModelId: "claude-sonnet-4-6", availableModels: [] },
        [],
      );
      const opus = modelInfos.find((m) => m.value === "opus");
      expect(opus?.supportsEffort).toBe(true);
      expect(opus?.supportedEffortLevels).toEqual(FULL_EFFORT_LADDER);
    });

    it("keeps the effort option after switching to fable, and effort medium/low succeed", async () => {
      populateInjectedSession(injectFableModel);

      // Switch the session's model to the injected fable (control op) — drives
      // the same applyConfigOptionValue model path Daniel hit.
      const afterSwitch = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "fable",
      });
      expect(setModelSpy).toHaveBeenCalledWith("fable");

      // Pre-fix this was undefined (option dropped) → the next set threw
      // "Unknown config option: effort". Post-fix it persists with the full ladder.
      const effortOption = afterSwitch.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      expect((effortOption as any).options.map((o: any) => o.value)).toEqual([
        "default",
        ...FULL_EFFORT_LADDER,
      ]);

      // Daniel's exact sequence: high → medium, then → low. Both must succeed.
      const afterMedium = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "medium",
      });
      expect(afterMedium.configOptions.find((o) => o.id === "effort")?.currentValue).toBe("medium");

      const afterLow = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });
      expect(afterLow.configOptions.find((o) => o.id === "effort")?.currentValue).toBe("low");
    });

    it("keeps the effort option after switching to opus, and effort medium/low succeed", async () => {
      populateInjectedSession(injectOpusModel);

      const afterSwitch = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "opus",
      });
      expect(setModelSpy).toHaveBeenCalledWith("opus");

      const effortOption = afterSwitch.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      expect((effortOption as any).options.map((o: any) => o.value)).toEqual([
        "default",
        ...FULL_EFFORT_LADDER,
      ]);

      const afterMedium = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "medium",
      });
      expect(afterMedium.configOptions.find((o) => o.id === "effort")?.currentValue).toBe("medium");

      const afterLow = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "low",
      });
      expect(afterLow.configOptions.find((o) => o.id === "effort")?.currentValue).toBe("low");
    });
  });

  // ── Brick c1cde4bd ──────────────────────────────────────────────────────────
  // On session/resume the adapter transiently has currentModelId = box-default
  // (non-effort-capable) while acpx fires set_config_option{effort} BEFORE
  // set_model(fable) settles.  buildConfigOptions produces no effort entry →
  // the lookup at acp-agent.ts:1959 returns undefined → line 1961 throws
  // "Unknown config option: effort".  The tolerance guard must intercept this
  // case and accept the effort op regardless of the transient model state.
  describe("effort on resume — tolerance guard (brick c1cde4bd)", () => {
    function populateResumeSession({
      withEffortCapableModelInfo,
    }: {
      withEffortCapableModelInfo: boolean;
    }) {
      setPermissionModeSpy = vi.fn();
      setModelSpy = vi.fn();
      applyFlagSettingsSpy = vi.fn();

      const sonnetInfo: ModelInfo = {
        value: "claude-sonnet-4-6",
        displayName: "Claude Sonnet",
        description: "Balanced",
        supportsEffort: false,
      };
      const fableInfo: ModelInfo = {
        value: "fable",
        displayName: "Claude Fable 5",
        description: "Most capable",
        supportsEffort: true,
        supportedEffortLevels: ["low", "medium", "high", "xhigh", "max"],
      };

      // configOptions WITHOUT an effort entry — this is the transient state:
      // currentModelId is the non-effort-capable box-default at resume time.
      const configOptionsNoEffort = [
        {
          id: "mode",
          name: "Mode",
          type: "select",
          category: "mode",
          currentValue: "default",
          options: MOCK_MODES.availableModes.map((m) => ({
            value: m.id,
            name: m.name,
            description: m.description,
          })),
        },
        {
          id: "model",
          name: "Model",
          type: "select",
          category: "model",
          currentValue: "claude-sonnet-4-6",
          options: [
            { value: "claude-sonnet-4-6", name: "Claude Sonnet", description: "Balanced" },
            { value: "fable", name: "Claude Fable 5", description: "Most capable" },
          ],
        },
        // No "effort" option — the race-exposed transient
      ];

      (agent as unknown as { sessions: Record<string, unknown> }).sessions[SESSION_ID] = {
        query: {
          setPermissionMode: setPermissionModeSpy,
          setModel: setModelSpy,
          applyFlagSettings: applyFlagSettingsSpy,
          supportedCommands: async () => [],
        },
        input: null,
        cancelled: false,
        permissionMode: "default",
        settingsManager: {},
        modes: structuredClone(MOCK_MODES),
        models: {
          currentModelId: "claude-sonnet-4-6",
          availableModels: [
            { modelId: "claude-sonnet-4-6", name: "Claude Sonnet", description: "Balanced" },
            { modelId: "fable", name: "Claude Fable 5", description: "Most capable" },
          ],
        },
        modelInfos: withEffortCapableModelInfo ? [sonnetInfo, fableInfo] : [sonnetInfo],
        configOptions: configOptionsNoEffort,
        contextWindowSize: 200000,
      };
    }

    it("accepts effort set before set_model(fable) when modelInfos advertises effort-capable model", async () => {
      populateResumeSession({ withEffortCapableModelInfo: true });

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "medium",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalled();
      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      expect(effortOption?.currentValue).toBe("medium");
    });

    it("accepts effort even when no effort-capable model is yet in modelInfos (worst-case race)", async () => {
      populateResumeSession({ withEffortCapableModelInfo: false });

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "high",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalled();
      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      expect(effortOption?.currentValue).toBe("high");
    });

    it("accepts effort=default when effort option is transiently absent", async () => {
      populateResumeSession({ withEffortCapableModelInfo: true });

      const response = await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "default",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalled();
      const effortOption = response.configOptions.find((o) => o.id === "effort");
      expect(effortOption).toBeDefined();
      expect(effortOption?.currentValue).toBe("default");
    });

    it("rejects an unknown effort value with Invalid-value error (not Unknown-option)", async () => {
      populateResumeSession({ withEffortCapableModelInfo: true });

      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "effort",
          value: "super-high",
        }),
      ).rejects.toThrow("Invalid value for config option effort: super-high");
    });

    it("still throws Unknown config option for non-effort unknown configId", async () => {
      populateResumeSession({ withEffortCapableModelInfo: true });

      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "thinking_depth",
          value: "medium",
        }),
      ).rejects.toThrow("Unknown config option: thinking_depth");
    });
  });
});
