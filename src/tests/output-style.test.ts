/**
 * Output style — unit U1 of https://acpx.devbox.nativai.de/?brick=4d16ab8b
 *
 * PRODUCTION SHAPES PROBED HERE (all measured on this box, `claude 2.1.239`,
 * Claude Agent SDK 0.3.219 — brick 4d16ab8b `conception/PROBES.md`). These are
 * not invented fixtures; each one is a shape the real harness produced:
 *
 *   • `available_output_styles` is FOUR built-ins, not five — `Concise` is
 *     documented upstream but absent here — and the casing is NOT uniform
 *     (`default` lowercase, the rest capitalised). P1.
 *   • A custom style's id is its `name:` frontmatter and MAY CONTAIN SPACES
 *     ("Nativai Probe Shared"). P4/P6.
 *   • ⚠️ Claude Code DOES NOT VALIDATE the style: `{outputStyle:"NoSuchStyle"}`
 *     comes back as `output_style: "NoSuchStyle"` with the same 4-item list.
 *     P3. That exact response is reproduced below; it is the shape our clamp
 *     and our setter validation exist for, and the only reason a session
 *     cannot silently claim a style it does not have.
 *
 * The regression lock this file exists for is
 * `does not touch the live query`: output style must NEVER be pushed into a
 * running SDK query. Its positive control sits directly beside it — `effort`
 * on the same session DOES call `applyFlagSettings` — so a spy that could
 * never have observed a call cannot masquerade as a passing prohibition.
 *
 * These tests and the code they guard were written by the same agent, so a
 * green run is not evidence on its own. Every guard below was shown to FAIL
 * when its production code is gutted, by the mutation probe at
 * `/wisdom/Bricks/68508424-c845-46fa-96a1-14a16c8e64ca/verification/output-style.mutation-probe.mjs`
 * (run it from this worktree's root; 10/10 caught, 0 survived, 2026-08-28).
 * If you change this file or the guards it pins, re-run it — a mutation that
 * starts surviving means an assertion has stopped biting. The probe reports a
 * stale anchor as a survivor rather than silently skipping it.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  AgentSideConnection,
  SessionNotification,
  type SessionConfigOption,
} from "@agentclientprotocol/sdk";
import type { ModelInfo, Options } from "@anthropic-ai/claude-agent-sdk";
import type { ClaudeAcpAgent as ClaudeAcpAgentType } from "../acp-agent.js";

/** Measured: the built-ins this box's harness enumerates. Order preserved. */
const BUILT_IN_STYLES = ["default", "Proactive", "Explanatory", "Learning"];
/** Measured: a custom style id is its `name:` frontmatter, spaces and all. */
const CUSTOM_STYLE = "Nativai Probe Shared";

type InitResult = {
  output_style?: unknown;
  available_output_styles?: unknown;
};

let capturedOptions: Options | undefined;
let mockInit: InitResult = {};

vi.mock("@anthropic-ai/claude-agent-sdk", async () => {
  const actual = await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  );
  return {
    ...actual,
    query: (args: { prompt: unknown; options: Options }) => {
      capturedOptions = args.options;
      return {
        initializationResult: async () => ({
          models: [
            {
              value: "claude-sonnet-4-6",
              displayName: "Claude Sonnet",
              description: "Fast",
              supportsAutoMode: true,
            },
          ],
          ...mockInit,
        }),
        setModel: async () => {},
        setPermissionMode: async () => {},
        applyFlagSettings: async () => {},
        supportedCommands: async () => [],
        [Symbol.asyncIterator]: async function* () {},
      };
    },
  };
});

vi.mock("../tools.js", async () => {
  const actual = await vi.importActual<typeof import("../tools.js")>("../tools.js");
  return {
    ...actual,
    registerHookCallback: vi.fn(),
  };
});

const SESSION_ID = "output-style-session";

describe("output style (U1)", () => {
  let agent: ClaudeAcpAgentType;
  let ClaudeAcpAgent: typeof ClaudeAcpAgentType;

  function createMockClient(): AgentSideConnection {
    return {
      sessionUpdate: async (_notification: SessionNotification) => {},
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as unknown as AgentSideConnection;
  }

  beforeEach(async () => {
    capturedOptions = undefined;
    // Default to the measured production response.
    mockInit = { output_style: "default", available_output_styles: [...BUILT_IN_STYLES] };

    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;
    agent = new ClaudeAcpAgent(createMockClient());
  });

  /** The `settings` object the adapter actually handed the SDK. */
  function capturedSettings(): Record<string, unknown> | undefined {
    return capturedOptions?.settings as Record<string, unknown> | undefined;
  }

  /** Flatten a select option's values, groups included — the same flatten the
   *  production validation path in `setSessionConfigOption` performs, so this
   *  helper cannot disagree with what the setter will actually accept. */
  function optionValues(option: SessionConfigOption | undefined): string[] {
    if (!option || !("options" in option)) {
      return [];
    }
    return option.options.flatMap((o) => ("options" in o ? o.options : [o])).map((o) => o.value);
  }

  // ------------------------------------------------------------------
  // A1 / A2 / A3 — the style reaches the harness through the flag-tier
  // creation settings, and only from its own `_meta` field.
  // ------------------------------------------------------------------
  describe("creation settings (A1/A2)", () => {
    it("folds _meta.claudeCode.outputStyle into the flag-tier settings object", async () => {
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "Explanatory" } },
      });

      expect(capturedSettings()?.outputStyle).toBe("Explanatory");
    });

    it("sets no outputStyle key when the caller asks for none", async () => {
      await agent.newSession({ cwd: "/test", mcpServers: [] });

      // Either no settings object at all, or one without the key — never a
      // key present with an empty/undefined value, which would pin the
      // session to "" instead of letting the settings cascade resolve.
      expect(capturedSettings()?.outputStyle).toBeUndefined();
      expect(Object.keys(capturedSettings() ?? {})).not.toContain("outputStyle");
    });

    it("preserves a custom style id verbatim, spaces included", async () => {
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: CUSTOM_STYLE } },
      });

      expect(capturedSettings()?.outputStyle).toBe(CUSTOM_STYLE);
    });

    it("trims surrounding whitespace but NEVER case-folds", async () => {
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "  Explanatory  " } },
      });
      expect(capturedSettings()?.outputStyle).toBe("Explanatory");

      capturedOptions = undefined;
      await agent.newSession({
        cwd: "/test2",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "explanatory" } },
      });
      // Lowercase in, lowercase out. Case-folding either direction breaks one
      // end of the list: `default` is lowercase, the rest are capitalised.
      expect(capturedSettings()?.outputStyle).toBe("explanatory");
    });

    it("ignores an empty or whitespace-only style", async () => {
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "   " } },
      });

      expect(capturedSettings()?.outputStyle).toBeUndefined();
    });

    it("A3: a caller-supplied settings object opts out of the style AND the effort pin", async () => {
      // The documented all-or-nothing contract: the adapter drops its entire
      // `creationSettings` when the caller supplies `settings`. Pinned here
      // because routing the style through `options.settings` instead of its
      // own `_meta` field is the trap that would silently kill the effort pin.
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: {
          claudeCode: {
            outputStyle: "Explanatory",
            options: { settings: { outputStyle: "Learning" } },
          },
        },
      });

      expect(capturedSettings()?.outputStyle).toBe("Learning");
      expect(capturedSettings()?.env).toBeUndefined();
    });
  });

  // ------------------------------------------------------------------
  // A6c — the resume seam. This is what the live control rests on.
  // ------------------------------------------------------------------
  describe("resume/load carry the style (A6c)", () => {
    it("loadSession composes the style into the resumed query's creation settings", async () => {
      await agent.loadSession({
        cwd: "/test",
        sessionId: SESSION_ID,
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "Learning" } },
      });

      expect(capturedSettings()?.outputStyle).toBe("Learning");
      // It really is the resume path, not a fresh session.
      expect(capturedOptions?.resume).toBe(SESSION_ID);
    });

    it("resumeSession composes the style the same way", async () => {
      await agent.resumeSession({
        cwd: "/test",
        sessionId: SESSION_ID,
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: CUSTOM_STYLE } },
      });

      expect(capturedSettings()?.outputStyle).toBe(CUSTOM_STYLE);
      expect(capturedOptions?.resume).toBe(SESSION_ID);
    });

    it("a resume carrying a DIFFERENT style rebuilds the query with the new one", async () => {
      await agent.newSession({
        cwd: "/test",
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "Explanatory" } },
      });
      expect(capturedSettings()?.outputStyle).toBe("Explanatory");

      // A cold resume — the shape acpx produces after recycling the queue
      // owner, which kills this adapter process along with it.
      vi.resetModules();
      const fresh = await import("../acp-agent.js");
      const respawned = new fresh.ClaudeAcpAgent(createMockClient());
      capturedOptions = undefined;
      await respawned.loadSession({
        cwd: "/test",
        sessionId: SESSION_ID,
        mcpServers: [],
        _meta: { claudeCode: { outputStyle: "Learning" } },
      });

      expect(capturedSettings()?.outputStyle).toBe("Learning");
    });
  });

  // ------------------------------------------------------------------
  // A4 / A5 / A9 — the advertisement, and the clamp that keeps it honest.
  // ------------------------------------------------------------------
  describe("advertisement (A4/A5)", () => {
    async function optionFor(init: InitResult) {
      mockInit = init;
      const response = await agent.newSession({ cwd: "/test", mcpServers: [] });
      return response.configOptions?.find((o) => o.id === "outputStyle");
    }

    it("advertises the harness's own list, in order, with the reported current value", async () => {
      const option = await optionFor({
        output_style: "Explanatory",
        available_output_styles: [...BUILT_IN_STYLES],
      });

      expect(option).toBeDefined();
      expect(option).toMatchObject({
        id: "outputStyle",
        name: "Output style",
        type: "select",
        currentValue: "Explanatory",
      });
      expect(optionValues(option)).toEqual(BUILT_IN_STYLES);
    });

    it("does not hardcode the built-ins — a custom style is advertised too", async () => {
      const option = await optionFor({
        output_style: CUSTOM_STYLE,
        available_output_styles: [...BUILT_IN_STYLES, CUSTOM_STYLE],
      });

      expect(option?.currentValue).toBe(CUSTOM_STYLE);
      expect(optionValues(option)).toContain(CUSTOM_STYLE);
    });

    it("advertises NOTHING when the harness enumerates no styles", async () => {
      // An empty dropdown reads as "supported, no choices". The absence of the
      // option is the honest "this harness has no output styles" signal that
      // acpx and acpx-ui derive support from.
      expect(
        await optionFor({ output_style: "default", available_output_styles: [] }),
      ).toBeUndefined();
    });

    it("advertises NOTHING when the harness omits the fields entirely", async () => {
      expect(await optionFor({})).toBeUndefined();
    });

    it("⚠️ P3: clamps a reported style that is not in the advertised list", async () => {
      // The measured Claude Code response to `{outputStyle:"NoSuchStyle"}` —
      // accepted, echoed back as active, list unchanged. Advertising that
      // verbatim would put a currentValue outside `options` on the wire and
      // report a style the session does not have.
      const option = await optionFor({
        output_style: "NoSuchStyle",
        available_output_styles: [...BUILT_IN_STYLES],
      });

      expect(option?.currentValue).toBe("default");
      expect(optionValues(option)).not.toContain("NoSuchStyle");
    });

    it("clamps to the first advertised style when the harness offers no `default`", async () => {
      const option = await optionFor({
        output_style: "NoSuchStyle",
        available_output_styles: ["Proactive", "Learning"],
      });

      expect(option?.currentValue).toBe("Proactive");
    });

    it("the advertised currentValue is ALWAYS a member of the advertised options", async () => {
      // The invariant the two clamp cases above are instances of. A select
      // whose currentValue is not in its own options is a control that lies.
      for (const init of [
        { output_style: "default", available_output_styles: [...BUILT_IN_STYLES] },
        { output_style: "NoSuchStyle", available_output_styles: [...BUILT_IN_STYLES] },
        { output_style: "", available_output_styles: [...BUILT_IN_STYLES] },
        { output_style: undefined, available_output_styles: [CUSTOM_STYLE] },
      ]) {
        const option = await optionFor(init);
        expect(option).toBeDefined();
        const values = optionValues(option);
        expect(values).toContain(option?.currentValue);
      }
    });
  });

  // ------------------------------------------------------------------
  // A7 / A9 — the setter. Validation, and the prohibition on live apply.
  // ------------------------------------------------------------------
  describe("setSessionConfigOption (A7/A9)", () => {
    let applyFlagSettingsSpy: ReturnType<typeof vi.fn>;

    function populateSession(styles: string[] = BUILT_IN_STYLES, current = "default") {
      applyFlagSettingsSpy = vi.fn();
      const modes = {
        currentModeId: "default",
        availableModes: [{ id: "default", name: "Default", description: "Standard" }],
      };
      const models = {
        currentModelId: "claude-opus-4-5",
        availableModels: [
          { modelId: "claude-opus-4-5", name: "Claude Opus", description: "Most capable" },
          { modelId: "claude-sonnet-4-6", name: "Claude Sonnet", description: "Balanced" },
        ],
      };
      const configOptions: unknown[] = [
        {
          id: "mode",
          name: "Mode",
          type: "select",
          category: "mode",
          currentValue: "default",
          options: [{ value: "default", name: "Default" }],
        },
        {
          id: "model",
          name: "Model",
          type: "select",
          category: "model",
          currentValue: "claude-opus-4-5",
          options: models.availableModels.map((m) => ({ value: m.modelId, name: m.name })),
        },
        {
          id: "effort",
          name: "Effort",
          type: "select",
          category: "thought_level",
          currentValue: "default",
          options: [
            { value: "default", name: "Default" },
            { value: "high", name: "High" },
          ],
        },
      ];
      if (styles.length > 0) {
        configOptions.push({
          id: "outputStyle",
          name: "Output style",
          description: "Response role, tone and format",
          type: "select",
          category: "mode",
          currentValue: current,
          options: styles.map((s) => ({ value: s, name: s })),
        });
      }

      (agent as unknown as { sessions: Record<string, unknown> }).sessions[SESSION_ID] = {
        query: {
          setPermissionMode: vi.fn(),
          setModel: vi.fn(),
          applyFlagSettings: applyFlagSettingsSpy,
          supportedCommands: async () => [],
        },
        input: null,
        cancelled: false,
        settingsManager: {},
        modes,
        models,
        modelInfos: models.availableModels.map(
          (m): ModelInfo => ({
            value: m.modelId,
            displayName: m.name,
            description: m.description,
            supportsEffort: true,
            supportedEffortLevels: ["low", "medium", "high"],
          }),
        ),
        configOptions,
        availableOutputStyles: styles,
        contextWindowSize: 200000,
      };
    }

    function currentAdvertised(): string | undefined {
      const session = (
        agent as unknown as {
          sessions: Record<string, { configOptions: { id: string; currentValue?: unknown }[] }>;
        }
      ).sessions[SESSION_ID];
      const option = session.configOptions.find((o) => o.id === "outputStyle");
      return typeof option?.currentValue === "string" ? option.currentValue : undefined;
    }

    it("accepts an advertised style and records it", async () => {
      populateSession();

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "outputStyle",
        value: "Explanatory",
      });

      expect(currentAdvertised()).toBe("Explanatory");
    });

    it("⚠️ AC-5: refuses a style outside available_output_styles", async () => {
      // Claude Code itself accepts this value (measured, P3). This assertion
      // therefore tests OUR guard, and it is the only thing standing between a
      // typo and a session that reports a style it does not have.
      populateSession();

      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "outputStyle",
          value: "NoSuchStyle",
        }),
      ).rejects.toThrow("Invalid value for config option outputStyle: NoSuchStyle");

      expect(currentAdvertised()).toBe("default");
    });

    it("is case-sensitive — a case variant of a real style is refused", async () => {
      populateSession();

      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "outputStyle",
          value: "explanatory",
        }),
      ).rejects.toThrow("Invalid value for config option outputStyle: explanatory");
    });

    it("refuses the option entirely on a session that advertises no styles", async () => {
      // The honest-degradation contract: support is derived from the
      // advertisement, so an unadvertised id is a visible error, never a
      // silent success.
      populateSession([]);

      await expect(
        agent.setSessionConfigOption({
          sessionId: SESSION_ID,
          configId: "outputStyle",
          value: "Explanatory",
        }),
      ).rejects.toThrow("Unknown config option: outputStyle");
    });

    it('clears via the literal "default" id (R-6 inversion #2 — never null)', async () => {
      // `applyFlagSettings({outputStyle:null})` — the documented clear idiom
      // the sibling effort code uses — CANNOT clear a style set at creation:
      // create-time and live settings occupy different flag-tier slots
      // (measured). "default" is an ordinary advertised member, so clearing is
      // an ordinary set that the next resume composes.
      populateSession(BUILT_IN_STYLES, "Explanatory");

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "outputStyle",
        value: "default",
      });

      expect(currentAdvertised()).toBe("default");
      expect(applyFlagSettingsSpy).not.toHaveBeenCalled();
    });

    it("⚠️ R-6 inversion #1: NEVER pushes the style into the live query", async () => {
      // THE regression lock. Moving the live config without a system-prompt
      // recompose leaves the model told it is in a style whose instructions it
      // has never seen — measured, and worse than a no-op. A new
      // `applyOutputStyleToSdk` (or any applyFlagSettings call on this path) is
      // the DEFECT, not the missing piece. See `applyConfigOptionValue`.
      populateSession();

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "outputStyle",
        value: "Learning",
      });

      expect(applyFlagSettingsSpy).not.toHaveBeenCalled();
      // The value is still recorded — the style binds on the next resume.
      expect(currentAdvertised()).toBe("Learning");
    });

    it("POSITIVE CONTROL: effort on the same session DOES reach the live query", async () => {
      // Without this, the prohibition above is indistinguishable from a spy
      // that could never have observed a call at all.
      populateSession();

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "effort",
        value: "high",
      });

      expect(applyFlagSettingsSpy).toHaveBeenCalled();
    });

    it("survives a model switch — the option is not dropped by the rebuild", async () => {
      // `buildConfigOptions` reconstructs the whole list on a model change; an
      // option not threaded through it disappears, and support is derived from
      // the advertisement, so the session would read as no longer supporting
      // styles.
      populateSession(BUILT_IN_STYLES, "Explanatory");

      await agent.setSessionConfigOption({
        sessionId: SESSION_ID,
        configId: "model",
        value: "claude-sonnet-4-6",
      });

      expect(currentAdvertised()).toBe("Explanatory");
    });
  });
});
