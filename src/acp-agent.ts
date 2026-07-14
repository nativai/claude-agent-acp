import {
  Agent,
  AgentSideConnection,
  AuthenticateRequest,
  AuthMethod,
  AvailableCommand,
  CancelNotification,
  ClientCapabilities,
  ForkSessionRequest,
  ForkSessionResponse,
  InitializeRequest,
  InitializeResponse,
  ListSessionsRequest,
  ListSessionsResponse,
  LoadSessionRequest,
  LoadSessionResponse,
  ndJsonStream,
  NewSessionRequest,
  NewSessionResponse,
  PermissionOption,
  PromptRequest,
  PromptResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestError,
  ResumeSessionRequest,
  ResumeSessionResponse,
  SessionConfigOption,
  SessionModelState,
  SessionModeState,
  SessionNotification,
  SetSessionConfigOptionRequest,
  SetSessionConfigOptionResponse,
  SetSessionModelRequest,
  SetSessionModelResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  CloseSessionRequest,
  CloseSessionResponse,
  DeleteSessionRequest,
  DeleteSessionResponse,
  TerminalHandle,
  TerminalOutputResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,
  StopReason,
} from "@agentclientprotocol/sdk";
import {
  CanUseTool,
  deleteSession,
  getSessionMessages,
  listSessions,
  McpServerConfig,
  ModelInfo,
  ModelUsage,
  Options,
  PermissionMode,
  PermissionUpdate,
  Query,
  query,
  Settings,
  SDKAssistantMessageError,
  SDKMessage,
  SDKMessageOrigin,
  SDKPartialAssistantMessage,
  SDKUserMessage,
  SlashCommand,
} from "@anthropic-ai/claude-agent-sdk";
import { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import { BetaContentBlock, BetaRawContentBlockDelta } from "@anthropic-ai/sdk/resources/beta.mjs";
import { randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";
import packageJson from "../package.json" with { type: "json" };
import { SettingsManager } from "./settings.js";
import {
  applyTaskCreate,
  applyTaskUpdate,
  ClaudePlanEntry,
  createPostToolUseHook,
  createTaskHook,
  parseTaskCreateOutput,
  planEntries,
  registerHookCallback,
  TaskState,
  taskStateToPlanEntries,
  toolInfoFromToolUse,
  toolUpdateFromDiffToolResponse,
  toolUpdateFromToolResult,
} from "./tools.js";
import { nodeToWebReadable, nodeToWebWritable, Pushable, unreachable } from "./utils.js";

export const CLAUDE_CONFIG_DIR =
  process.env.CLAUDE_CONFIG_DIR ?? path.join(os.homedir(), ".claude");

const MAX_TITLE_LENGTH = 256;

function sanitizeTitle(text: string): string {
  // Replace newlines and collapse whitespace
  const sanitized = text
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (sanitized.length <= MAX_TITLE_LENGTH) {
    return sanitized;
  }
  return sanitized.slice(0, MAX_TITLE_LENGTH - 1) + "…";
}

/**
 * Logger interface for customizing logging output
 */
export interface Logger {
  log: (...args: any[]) => void;
  error: (...args: any[]) => void;
}

type AccumulatedUsage = {
  inputTokens: number;
  outputTokens: number;
  cachedReadTokens: number;
  cachedWriteTokens: number;
};

type UsageSnapshot = {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
};

const ZERO_USAGE = Object.freeze({
  input_tokens: 0,
  output_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const DEFAULT_CONTEXT_WINDOW = 200000;

// Coalescing bounds for the thinking-token progress signal. The SDK emits a
// `thinking_tokens` system message per stream frame during the (often
// redacted) thinking phase; forwarding each one would flood the client. We
// emit at most one `_meta` update per THINKING_TOKENS_THROTTLE_MS, or sooner
// if the running estimate jumped by at least THINKING_TOKENS_MIN_DELTA tokens.
const THINKING_TOKENS_THROTTLE_MS = 400;
const THINKING_TOKENS_MIN_DELTA = 64;

/**
 * Reliability surfacing constants — SHARED CONTRACT with the acpx-ui server and
 * the acpx CLI (conception §0). Keep these values in lockstep across the three
 * repos; they are intentionally named so a single edit retunes them.
 *
 *   RESUME_HEARTBEAT_MS  cadence of init/resume "still resuming — N s" heartbeats.
 *   INIT_HARD_MS         hard ceiling on init/resume; past this we throw a
 *                        structured terminal error instead of hanging forever.
 *   WEDGE_DISPLAY_MS     turn no-activity surfacing threshold ("no activity for N s").
 *
 * These make the silent hang VISIBLE; they are NOT a cure. A natively-blocked
 * child makes the SDK's `query.next()` never yield, so the guaranteed un-wedge
 * is an EXTERNAL process force-restart (owned by acpx). See CONCEPTION §3, §4.3,
 * §4.6.
 */
export const RESUME_HEARTBEAT_MS = 5_000;
export const INIT_HARD_MS = 300_000;
export const WEDGE_DISPLAY_MS = 90_000;

/**
 * Hard cap on `session.pendingSdkMessages` (the mid-await SDK-message buffer
 * introduced to close the `activePromptResolve` routing hole — CONCEPTION §1.1
 * A1). The buffer only grows while the prompt loop is mid-`await` between parks,
 * so in practice it holds a handful of messages at most. A count this large means
 * the loop is wedged and the buffer is masking it — so overflow is a LOUD failure
 * (stage `backgroundLoopError`, terminate the turn via the existing error path),
 * never a silent drop (silent drop is the bug being fixed).
 */
export const MAX_PENDING_SDK_MESSAGES = 4096;

/**
 * Opt-in, best-effort Tier-1 self-abort (CONCEPTION §4.3 Tier 1). When the env
 * var `ACP_TURN_NOACTIVITY_ABORT_MS` is set to a positive integer, a turn with
 * no delivered message for that many ms tears itself down (abort + close) and
 * the prompt throws a terminal error. OFF by default: this only frees wedges the
 * SDK can preempt — a natively-blocked child still needs the external restart.
 */
function turnNoActivityAbortMs(): number | null {
  const raw = process.env.ACP_TURN_NOACTIVITY_ABORT_MS;
  if (!raw) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Terminal reason for the last turn, forwarded to the client so the acpx-ui
 *  server can surface it as `activity.lastTurnEndReason`. Mirrors the stop
 *  reasons already captured by the prompt loop. */
type LastTurnEndReason = "end_turn" | "max_tokens" | "max_turns" | "error" | "cancelled";

/** ext-notification method carrying ephemeral adapter status (resume heartbeat /
 *  turn no-activity). A side channel — NOT a `session/update` — so it advances
 *  the client's stream (defeating staleness) without changing the stream-tail
 *  kind the server uses to tell `resuming` from `working`. */
export const SESSION_STATUS_NOTIFICATION = "_claude/sessionStatus";
/** Out-of-band signal (NOT a `session/update`) emitted when the SDK reports a
 *  `model_refusal_fallback`: the active model refused the turn and the SDK
 *  transparently fell back to another model, retracting the already-streamed
 *  refused partial. Carries the retraction record so a client that tracks
 *  message identity can evict the superseded messages; non-handling clients
 *  ignore it (no transcript impact). */
export const MODEL_REFUSAL_FALLBACK_NOTIFICATION = "_claude/modelRefusalFallback";
/** `_meta` key carrying the terminal turn reason on the final `usage_update`
 *  `session/update` and on the `PromptResponse`. */
export const LAST_TURN_END_REASON_META_KEY = "_claude/lastTurnEndReason";

type Session = {
  query: Query;
  input: Pushable<SDKUserMessage>;
  cancelled: boolean;
  cwd: string;
  /** Serialized snapshot of session-defining params (cwd, mcpServers) used to
   *  detect when loadSession/resumeSession is called with changed values. */
  sessionFingerprint: string;
  settingsManager: SettingsManager;
  accumulatedUsage: AccumulatedUsage;
  modes: SessionModeState;
  models: SessionModelState;
  modelInfos: ModelInfo[];
  configOptions: SessionConfigOption[];
  promptRunning: boolean;
  pendingMessages: Map<string, { resolve: (cancelled: boolean) => void; order: number }>;
  nextPendingOrder: number;
  abortController: AbortController;
  emitRawSDKMessages: boolean | SDKMessageFilter[];
  /** Resolve callback for the active prompt's current nextMessage() call. null when idle. */
  activePromptResolve: ((msg: SDKMessage | null) => void) | null;
  /** SDK messages that arrived while a prompt owns the stream (`promptRunning`)
   *  but its loop was mid-`await` (no resolver parked). The background reader
   *  buffers them here instead of diverting them to `handleIdleMessage`, which
   *  silently discarded turn-control messages (`session_state_changed`, user
   *  replays) and withheld the turn's response (RCA §1.2, CONCEPTION §1.1 A1).
   *  `nextMessage()` drains this FIFO before parking. A `null` entry is a
   *  stream-end / error sentinel so the loop still observes termination. */
  pendingSdkMessages: (SDKMessage | null)[];
  /** Error captured by the background reader loop, to be re-thrown by the prompt. */
  backgroundLoopError: Error | null;
  /** Context window size of the last top-level assistant model, carried across
   *  prompts so mid-stream usage_update notifications report a correct `size`
   *  before the turn's first result message arrives. Defaults to
   *  DEFAULT_CONTEXT_WINDOW, refreshed from each result's modelUsage, and
   *  invalidated when the user switches the session's model. */
  contextWindowSize: number;
  /** Accumulated task list for the session, keyed by task ID. Task IDs are
   *  per-session, so this state must not be shared across sessions. */
  taskState: TaskState;
};

/** Compute a stable fingerprint of the session-defining params so we can
 *  detect when a loadSession/resumeSession call requires tearing down and
 *  recreating the underlying Query process.  MCP servers are sorted by name
 *  so that ordering differences don't trigger unnecessary recreations. */
function computeSessionFingerprint(params: {
  cwd: string;
  mcpServers?: NewSessionRequest["mcpServers"];
}): string {
  const servers = [...(params.mcpServers ?? [])].sort((a, b) => a.name.localeCompare(b.name));
  return JSON.stringify({ cwd: params.cwd, mcpServers: servers });
}

type BackgroundTerminal =
  | {
      handle: TerminalHandle;
      status: "started";
      lastOutput: TerminalOutputResponse | null;
    }
  | {
      status: "aborted" | "exited" | "killed" | "timedOut";
      pendingOutput: TerminalOutputResponse;
    };

export type SDKMessageFilter = {
  type: string;
  subtype?: string;
  origin?: SDKMessageOrigin["kind"];
};

/**
 * Extra metadata that can be given when creating a new session.
 */
export type NewSessionMeta = {
  claudeCode?: {
    /**
     * Options forwarded to Claude Code when starting a new session.
     * Those parameters will be ignored and managed by ACP:
     *   - cwd
     *   - includePartialMessages
     *   - allowDangerouslySkipPermissions
     *   - permissionMode
     *   - canUseTool
     *   - executable
     * Those parameters will be used and updated to work with ACP:
     *   - hooks (merged with ACP's hooks)
     *   - mcpServers (merged with ACP's mcpServers)
     *   - disallowedTools (merged with ACP's disallowedTools)
     *   - tools (passed through; defaults to claude_code preset if not provided)
     */
    options?: Options;
    /**
     * When set, raw SDK messages are emitted as extNotification("_claude/sdkMessage", message)
     * in addition to normal processing.
     * - true: emit all messages
     * - false/undefined: emit nothing (default)
     * - SDKMessageFilter[]: emit only messages matching at least one filter
     */
    emitRawSDKMessages?: boolean | SDKMessageFilter[];
  };
  additionalRoots?: string[];
};

/**
 * Extra metadata for 'gateway' authentication requests.
 */
type GatewayAuthMeta = {
  /**
   * These parameters are mapped to environment variables to:
   * - Redirect API calls via baseUrl
   * - Inject custom headers
   * - Bypass the default Claude login requirement
   */
  gateway: {
    baseUrl: string;
    headers: Record<string, string>;
  };
};

type GatewayAuthRequest = AuthenticateRequest & { _meta?: GatewayAuthMeta };

/**
 * Subagent info cached when a teammate is spawned (keyed by parent tool use ID).
 */
type SubagentInfo = {
  agentId: string;
  name: string;
  color?: string;
};

/**
 * Extra metadata that the agent provides for each tool_call / tool_update update.
 */
export type ToolUpdateMeta = {
  claudeCode?: {
    /* The name of the tool that was used in Claude Code. */
    toolName: string;
    /* The structured output provided by Claude Code. */
    toolResponse?: unknown;
    /* Status tag, e.g. 'teammate_spawned' when a subagent is launched. */
    status?: string;
    /* The parent tool use ID when this message originated from a subagent. */
    parentToolUseId?: string;
    /* Subagent identifier, e.g. 'poet-a@haiku-demo'. */
    subagentId?: string;
    /* Subagent display name, e.g. 'poet-a'. */
    subagentName?: string;
    /* Subagent color, e.g. 'blue'. */
    subagentColor?: string;
    /* Last tool name used by the subagent, from task_progress. */
    taskLastToolName?: string;
  };
  /* Terminal metadata for Bash tool execution, matching codex-acp's _meta protocol. */
  terminal_info?: {
    terminal_id: string;
  };
  terminal_output?: {
    terminal_id: string;
    data: string;
  };
  terminal_exit?: {
    terminal_id: string;
    exit_code: number;
    signal: string | null;
  };
};

export type ToolUseCache = {
  [key: string]: {
    type: "tool_use" | "server_tool_use" | "mcp_tool_use";
    id: string;
    name: string;
    input: unknown;
  };
};

export async function claudeCliPath(): Promise<string> {
  if (process.env.CLAUDE_CODE_EXECUTABLE) {
    return process.env.CLAUDE_CODE_EXECUTABLE;
  }
  // The SDK's CLI is a native binary shipped as a platform-specific optional
  // dependency of @anthropic-ai/claude-agent-sdk. Resolve via a require bound
  // to the SDK so nested installs are found even when npm doesn't hoist.
  const { createRequire } = await import("node:module");
  const req = createRequire(import.meta.resolve("@anthropic-ai/claude-agent-sdk"));
  const ext = process.platform === "win32" ? ".exe" : "";
  // On linux, both glibc and musl variants may be installed side-by-side
  // (e.g. bunx hydrates every optional dep), so picking one by trial is
  // unreliable: the wrong binary segfaults at runtime instead of failing to
  // spawn. Detect the runtime libc and prefer the matching variant, falling
  // back to the other only if the preferred one isn't installed.
  const candidates =
    process.platform === "linux"
      ? isMuslLibc()
        ? [
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}-musl/claude${ext}`,
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}/claude${ext}`,
          ]
        : [
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}/claude${ext}`,
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}-musl/claude${ext}`,
          ]
      : [`@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}/claude${ext}`];
  for (const candidate of candidates) {
    try {
      return req.resolve(candidate);
    } catch {
      // try next candidate
    }
  }
  throw new Error(
    `Claude native binary not found for ${process.platform}-${process.arch}. ` +
      `Reinstall @anthropic-ai/claude-agent-sdk without --omit=optional, or set CLAUDE_CODE_EXECUTABLE.`,
  );
}

function isMuslLibc(): boolean {
  // process.report.getReport().header.glibcVersionRuntime is populated when
  // Node is dynamically linked against glibc, and absent on musl.
  const report = process.report?.getReport() as
    | { header?: { glibcVersionRuntime?: string } }
    | undefined;
  return !report?.header?.glibcVersionRuntime;
}

function shouldHideClaudeAuth(): boolean {
  return process.argv.includes("--hide-claude-auth");
}

// Bypass Permissions doesn't work if we are a root/sudo user
const IS_ROOT = (process.geteuid?.() ?? process.getuid?.()) === 0;
const ALLOW_BYPASS = !IS_ROOT || !!process.env.IS_SANDBOX;

// Slash commands that the SDK handles locally without replaying the user
// message and without invoking the model.
const LOCAL_ONLY_COMMANDS = new Set(["/context", "/heapdump", "/extra-usage"]);

// The Claude SDK persists local slash command invocations (e.g. `/model`) and
// their output as user messages in the session transcript, wrapping the
// payload in these XML-like markers that the CLI uses for its own display.
// The live prompt loop drops them; replay must strip them too or they leak
// into the UI on session/load.
const LOCAL_COMMAND_TAG_PATTERN =
  /<(command-name|command-message|command-args|local-command-stdout|local-command-stderr)>[\s\S]*?<\/\1>/g;

function stripMarkerTags(text: string): string {
  return text.replace(LOCAL_COMMAND_TAG_PATTERN, "");
}

/**
 * Return user-message content with local-command marker tags removed, or
 * `null` if nothing meaningful remains (caller should skip the message).
 * Preserves real prose that's mixed in alongside the markers — e.g. a
 * message like `<command-name>…</command-name>hi` becomes `hi`.
 */
export function stripLocalCommandMetadata(content: unknown): unknown | null {
  if (typeof content === "string") {
    const stripped = stripMarkerTags(content);
    return stripped.trim() === "" ? null : stripped;
  }
  if (!Array.isArray(content)) return content;

  const kept: unknown[] = [];
  for (const block of content) {
    if (
      block &&
      typeof block === "object" &&
      "type" in block &&
      (block as { type: unknown }).type === "text" &&
      "text" in block &&
      typeof (block as { text: unknown }).text === "string"
    ) {
      const stripped = stripMarkerTags((block as { text: string }).text);
      if (stripped.trim() === "") continue;
      kept.push({ ...(block as object), text: stripped });
    } else {
      kept.push(block);
    }
  }
  if (kept.length === 0) return null;
  return kept;
}

export function isLocalCommandMetadata(content: unknown): boolean {
  return stripLocalCommandMetadata(content) === null;
}

const PERMISSION_MODE_ALIASES: Record<string, PermissionMode> = {
  auto: "auto",
  default: "default",
  acceptedits: "acceptEdits",
  dontask: "dontAsk",
  plan: "plan",
  bypasspermissions: "bypassPermissions",
  bypass: "bypassPermissions",
};

export function resolvePermissionMode(
  defaultMode?: unknown,
  logger: Logger = console,
): PermissionMode {
  if (defaultMode === undefined) {
    return "default";
  }

  if (typeof defaultMode !== "string") {
    logger.error("Ignoring permissions.defaultMode from settings: expected a string.");
    return "default";
  }

  const normalized = defaultMode.trim().toLowerCase();
  if (normalized === "") {
    logger.error("Ignoring permissions.defaultMode from settings: expected a non-empty string.");
    return "default";
  }

  const mapped = PERMISSION_MODE_ALIASES[normalized];
  if (!mapped) {
    logger.error(`Ignoring permissions.defaultMode from settings: unknown value '${defaultMode}'.`);
    return "default";
  }

  if (mapped === "bypassPermissions" && !ALLOW_BYPASS) {
    logger.error(
      "Ignoring permissions.defaultMode from settings: bypassPermissions is not available when running as root.",
    );
    return "default";
  }

  return mapped;
}

/**
 * Builds the label for the "Always Allow" permission option so the user can see
 * the exact scope they are committing to. Uses the SDK-provided suggestions
 * when available (e.g. `Bash(npm test:*)`) and falls back to naming the whole
 * tool so "Always Allow" is never a blank check without disclosure.
 */
export function describeAlwaysAllow(
  suggestions: PermissionUpdate[] | undefined,
  toolName: string,
): string {
  if (!suggestions || suggestions.length === 0) {
    return `Always Allow all ${toolName}`;
  }

  const ruleLabels: string[] = [];
  const directories: string[] = [];

  for (const update of suggestions) {
    if (update.type === "addRules" && update.behavior === "allow") {
      for (const rule of update.rules) {
        ruleLabels.push(
          rule.ruleContent ? `${rule.toolName}(${rule.ruleContent})` : `all ${rule.toolName}`,
        );
      }
    } else if (update.type === "addDirectories") {
      directories.push(...update.directories);
    }
  }

  const parts: string[] = [];
  if (ruleLabels.length > 0) {
    parts.push(ruleLabels.join(", "));
  }
  if (directories.length > 0) {
    parts.push(`access to ${directories.join(", ")}`);
  }

  if (parts.length === 0) {
    return `Always Allow all ${toolName}`;
  }

  return `Always Allow ${parts.join(" and ")}`;
}

// Implement the ACP Agent interface
export class ClaudeAcpAgent implements Agent {
  sessions: {
    [key: string]: Session;
  };
  client: AgentSideConnection;
  toolUseCache: ToolUseCache;
  backgroundTerminals: { [key: string]: BackgroundTerminal } = {};
  clientCapabilities?: ClientCapabilities;
  logger: Logger;
  gatewayAuthRequest?: GatewayAuthRequest;
  /** Maps parent tool use ID → subagent info for spawned teammates. */
  subagentCache: Map<string, SubagentInfo> = new Map();

  /**
   * Creation params of the most recent `session/fork` on this connection.
   *
   * When acpx copies a Claude session it does NOT trust the id our
   * `unstable_forkSession` returns: it materializes its own *durable* forked
   * transcript (a fresh, independent session id) and then drives
   * `session/set_model` / `session/set_config_option` on that id over the same
   * connection — without a preceding `session/resume`. That id was therefore
   * never registered in `sessions`, so the config op would fail "Session not
   * found". We keep the fork's creation context here so those ops can lazily
   * resume the durable transcript from disk (see `resolveSessionForConfigOp`).
   */
  private lastForkContext?: {
    cwd: string;
    mcpServers: NewSessionRequest["mcpServers"];
    additionalDirectories?: NewSessionRequest["additionalDirectories"];
    _meta?: NewSessionRequest["_meta"];
  };

  constructor(client: AgentSideConnection, logger?: Logger) {
    this.sessions = {};
    this.client = client;
    this.toolUseCache = {};
    this.logger = logger ?? console;
  }

  async initialize(request: InitializeRequest): Promise<InitializeResponse> {
    this.clientCapabilities = request.clientCapabilities;

    // Bypasses standard auth by routing requests through a custom Anthropic-protocol gateway.
    // Only offered when the client advertises `auth._meta.gateway` capability.
    const supportsGatewayAuth = request.clientCapabilities?.auth?._meta?.gateway === true;

    const gatewayAuthMethod: AuthMethod = {
      id: "gateway",
      name: "Custom model gateway",
      description: "Use a custom gateway to authenticate and access models",
      _meta: {
        gateway: {
          protocol: "anthropic",
        },
      },
    };

    const gatewayBedrockAuthMethod: AuthMethod = {
      id: "gateway-bedrock",
      name: "Custom model gateway",
      description: "Use a custom gateway to authenticate and access models",
      _meta: {
        gateway: {
          protocol: "bedrock",
        },
      },
    };

    const supportsTerminalAuth = request.clientCapabilities?.auth?.terminal === true;
    const supportsMetaTerminalAuth = request.clientCapabilities?._meta?.["terminal-auth"] === true;

    // Detect remote environments where the OAuth browser redirect to localhost
    // won't work. This matches the SDK's internal isRemote check. In these cases,
    // the `auth login` subcommand would fall back to a device-code-like manual
    // flow, which doesn't work well over ACP, so we offer the TUI login instead.
    const isRemote = !!(
      process.env.NO_BROWSER ||
      process.env.SSH_CONNECTION ||
      process.env.SSH_CLIENT ||
      process.env.SSH_TTY ||
      process.env.CLAUDE_CODE_REMOTE
    );
    const terminalAuthMethods: AuthMethod[] = [];

    if (isRemote) {
      const remoteLoginMethod: AuthMethod = {
        description: "Run `claude /login` in the terminal",
        name: "Log in with Claude",
        id: "claude-login",
        type: "terminal",
        args: ["--cli"],
      };

      if (supportsMetaTerminalAuth) {
        remoteLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...process.argv.slice(1), "--cli"],
            label: "Claude Login",
          },
        };
      }

      if (!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)) {
        terminalAuthMethods.push(remoteLoginMethod);
      }
    } else {
      const claudeLoginMethod: AuthMethod = {
        description: "Use Claude subscription ",
        name: "Claude Subscription",
        id: "claude-ai-login",
        type: "terminal",
        args: ["--cli", "auth", "login", "--claudeai"],
      };

      const consoleLoginMethod: AuthMethod = {
        description: "Use Anthropic Console (API usage billing)",
        name: "Anthropic Console",
        id: "console-login",
        type: "terminal",
        args: ["--cli", "auth", "login", "--console"],
      };

      if (supportsMetaTerminalAuth) {
        const baseArgs = process.argv.slice(1);
        claudeLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...baseArgs, "--cli", "auth", "login", "--claudeai"],
            label: "Claude Login",
          },
        };
        consoleLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...baseArgs, "--cli", "auth", "login", "--console"],
            label: "Anthropic Console Login",
          },
        };
      }

      if (!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)) {
        terminalAuthMethods.push(claudeLoginMethod);
      }
      if (supportsTerminalAuth || supportsMetaTerminalAuth) {
        terminalAuthMethods.push(consoleLoginMethod);
      }
    }

    return {
      protocolVersion: 1,
      agentCapabilities: {
        _meta: {
          claudeCode: {
            promptQueueing: true,
          },
        },
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
        mcpCapabilities: {
          http: true,
          sse: true,
        },
        loadSession: true,
        sessionCapabilities: {
          additionalDirectories: {},
          close: {},
          delete: {},
          fork: {},
          list: {},
          resume: {},
        },
      },
      agentInfo: {
        name: packageJson.name,
        title: "Claude Agent",
        version: packageJson.version,
      },
      authMethods: [
        ...terminalAuthMethods,
        ...(supportsGatewayAuth ? [gatewayAuthMethod, gatewayBedrockAuthMethod] : []),
      ],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const response = await this.createSession(params, {
      // Revisit these meta values once we support resume
      resume: (params._meta as NewSessionMeta | undefined)?.claudeCode?.options?.resume,
    });
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async unstable_forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    // Remember the fork's creation context so a follow-up set_model/config on
    // acpx's out-of-band durable fork id can lazily resume it (see
    // `lastForkContext` / `resolveSessionForConfigOp`).
    this.lastForkContext = {
      cwd: params.cwd,
      mcpServers: params.mcpServers ?? [],
      additionalDirectories: params.additionalDirectories,
      _meta: params._meta,
    };
    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        additionalDirectories: params.additionalDirectories,
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
        forkSession: true,
      },
    );
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const result = await this.getOrCreateSession(params);

    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 0);
    return result;
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const result = await this.getOrCreateSession(params);

    await this.replaySessionHistory(params.sessionId);

    // Send available commands after replay so it doesn't interleave with history
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 0);

    return result;
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const sdk_sessions = await listSessions({ dir: params.cwd ?? undefined });
    const sessions = [];

    for (const session of sdk_sessions) {
      if (!session.cwd) continue;
      sessions.push({
        sessionId: session.sessionId,
        cwd: session.cwd,
        title: sanitizeTitle(session.summary),
        updatedAt: new Date(session.lastModified).toISOString(),
      });
    }
    return {
      sessions,
    };
  }

  async authenticate(_params: AuthenticateRequest): Promise<void> {
    if (_params.methodId === "gateway" || _params.methodId === "gateway-bedrock") {
      this.gatewayAuthRequest = _params as GatewayAuthRequest;
      return;
    }
    throw new Error("Method not implemented.");
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }

    session.cancelled = false;
    // A2 (RCA §1.3): the `accumulatedUsage` reset is NOT done here at entry.
    // A concurrent prompt() that resets on entry — then immediately parks behind
    // the running turn (below) — zeroes the RUNNING turn's eventual
    // `sessionUsage(session)` response. Instead each ownership branch below resets
    // only when this prompt actually takes the stream.

    let lastAssistantTotalUsage: number | null = null;
    let lastAssistantUsage: UsageSnapshot | null = null;
    let lastAssistantModel: string | null = null;
    // When the Claude SDK classifies a turn as failed (e.g. rate limit, auth
    // problem, billing), it sets a categorical `error` field on the
    // `SDKAssistantMessage` that precedes the final `result` message. We
    // capture it here so the subsequent `RequestError.internalError` can
    // forward it to clients as structured `data`, sparing them from
    // pattern-matching on the human-readable message text.
    let lastAssistantError: SDKAssistantMessageError | undefined;
    // Tracks whether we're inside a compaction. The SDK emits the terminal
    // `status` (compact_result success/failed) twice for a single failed
    // compaction, and the two messages are indistinguishable — so we report the
    // outcome only while a compaction is in progress, then clear this. A fresh
    // `compacting` status sets it again, so every distinct compaction (e.g.
    // repeated auto-compactions in a long turn) is still shown.
    let compactionInProgress = false;

    const userMessage = promptToClaude(params);

    const promptUuid = randomUUID();
    userMessage.uuid = promptUuid;

    // These local-only commands return a result without replaying the user
    // message. Mark promptReplayed=true so their result isn't consumed as a
    // background task result.
    const firstText = params.prompt[0]?.type === "text" ? params.prompt[0].text : "";
    const isLocalOnlyCommand =
      firstText.startsWith("/") && LOCAL_ONLY_COMMANDS.has(firstText.split(" ", 1)[0]);

    if (session.promptRunning) {
      session.input.push(userMessage);
      const order = session.nextPendingOrder++;
      const cancelled = await new Promise<boolean>((resolve) => {
        session.pendingMessages.set(promptUuid, { resolve, order });
      });
      if (cancelled) {
        return { stopReason: "cancelled" };
      }
      // A2: reset now — the running turn has ended and this prompt is taking
      // over the stream (handoff). Resetting at entry would have zeroed that
      // turn's usage while we were still parked behind it.
      session.accumulatedUsage = {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      };
    } else {
      // Fresh turn taking the stream immediately: reset before pushing so no
      // concurrently-read result can be zeroed after it is counted.
      session.accumulatedUsage = {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      };
      session.input.push(userMessage);
    }

    session.promptRunning = true;
    let handedOff = false;
    let errored = false;
    let stopReason: StopReason = "end_turn";
    // Terminal reason for THIS turn, forwarded to the client (CONCEPTION §3, D3).
    // Set when a `result` message arrives; consumed by the final `usage_update`
    // and the `PromptResponse`.
    let lastTurnEndReason: LastTurnEndReason | undefined;

    // Per-turn throttle state for the thinking-token progress signal (consumed
    // by the `thinking_tokens` case below). Reset implicitly each prompt.
    let lastThinkingTokensAt = 0;
    let lastThinkingTokensValue = -1;

    /** Waits for the background reader loop to deliver the next SDK message.
     *  Drains any messages the reader buffered while this loop was mid-`await`
     *  (A1) before parking. FIFO is structurally guaranteed: the reader only
     *  buffers when no resolver is parked, and this loop never parks while the
     *  buffer is non-empty — single event loop, no interleaving window. */
    const nextMessage = (): Promise<SDKMessage | null> => {
      if (session.pendingSdkMessages.length > 0) {
        return Promise.resolve(session.pendingSdkMessages.shift() ?? null);
      }
      return new Promise((resolve) => {
        session.activePromptResolve = resolve;
      });
    };

    // Turn no-activity surfacing (CONCEPTION §3) + optional Tier-1 self-abort
    // (§4.3). Reset on every delivered message; if WEDGE_DISPLAY_MS elapses with
    // no message we emit "no activity for N s". This runs on the adapter's event
    // loop, which stays ALIVE during a wedge (only the SDK child / query.next()
    // is blocked), so the timer still fires. SURFACING ONLY — it does not abort
    // the wedge unless the opt-in self-abort env var is set; the guaranteed cure
    // is an external force-restart.
    //
    // ONE-SHOT per silent stretch (re-armed only when real activity resumes).
    // CRITICAL: acpx persists every adapter notification to the session stream
    // and bumps `event_log.last_write_at` on each write (raw tap — acpx
    // runtime.ts onAcpMessage / queue-owner-runtime.ts). The acpx-ui server's
    // wedge clock is `now - last_write_at`, so a RECURRING marker would reset
    // that clock every WEDGE_DISPLAY_MS and MASK the wedge forever (never
    // detected → watchdog never restarts). One-shot bounds any clock disturbance
    // to a single bump. (acpx guards `_claude/sessionStatus` from the
    // last_write_at bump as the complete cross-repo fix.)
    let lastActivityAt = Date.now();
    let noActivityNotified = false;
    const selfAbortMs = turnNoActivityAbortMs();
    const noActivityTimer = setInterval(() => {
      const elapsedMs = Date.now() - lastActivityAt;
      if (elapsedMs < WEDGE_DISPLAY_MS) return;
      const seconds = Math.round(elapsedMs / 1000);
      if (!noActivityNotified) {
        noActivityNotified = true;
        void this.client
          .extNotification(SESSION_STATUS_NOTIFICATION, {
            sessionId: params.sessionId,
            phase: "turn_no_activity",
            elapsedMs,
            message: `no activity for ${seconds}s`,
          })
          .catch(() => {});
      }
      if (selfAbortMs !== null && elapsedMs >= selfAbortMs) {
        // Best-effort Tier-1 self-abort (OFF unless ACP_TURN_NOACTIVITY_ABORT_MS
        // is set). Stage a terminal error, then teardown (abort + close). The
        // teardown's cancel() resolves the parked nextMessage() so the loop
        // re-throws this error. NOT guaranteed to free a natively-blocked child.
        this.logger.error(
          `Session ${params.sessionId}: no activity for ${seconds}s — best-effort ` +
            `self-abort (external restart is the guaranteed recovery).`,
        );
        session.backgroundLoopError ??= new Error(
          `Turn aborted: no activity for ${seconds}s. Best-effort self-abort; if the ` +
            `agent is natively blocked, an external process restart is required.`,
        );
        clearInterval(noActivityTimer);
        void this.teardownSession(params.sessionId);
      }
    }, WEDGE_DISPLAY_MS);

    try {
      while (true) {
        const message = await nextMessage();
        // Reset the no-activity timer: the background loop just delivered.
        // Re-arm the one-shot so a fresh silent stretch can surface again.
        lastActivityAt = Date.now();
        noActivityNotified = false;

        if (!message) {
          // Background loop signalled done or errored.
          if (session.backgroundLoopError) {
            throw session.backgroundLoopError;
          }
          if (session.cancelled) {
            return { stopReason: "cancelled" };
          }
          break;
        }

        if (
          session.emitRawSDKMessages &&
          shouldEmitRawMessage(session.emitRawSDKMessages, message)
        ) {
          await this.client.extNotification("_claude/sdkMessage", {
            sessionId: params.sessionId,
            message: message as Record<string, unknown>,
          });
        }

        switch (message.type) {
          case "system":
            switch (message.subtype) {
              case "init":
                break;
              case "status": {
                if (message.status === "compacting") {
                  compactionInProgress = true;
                  await this.client.sessionUpdate({
                    sessionId: message.session_id,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: "Compacting..." },
                    },
                  });
                } else if (message.compact_result === "success" && compactionInProgress) {
                  // The SDK signals manual `/compact` completion with a status
                  // message carrying `compact_result`, not the `compact_boundary`
                  // message (which only fires when there's content to compact).
                  compactionInProgress = false;
                  await this.client.sessionUpdate({
                    sessionId: message.session_id,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: "\n\nCompacting completed." },
                    },
                  });
                } else if (message.compact_result === "failed" && compactionInProgress) {
                  compactionInProgress = false;
                  const reason = message.compact_error ? `: ${message.compact_error}` : ".";
                  await this.client.sessionUpdate({
                    sessionId: message.session_id,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: `\n\nCompacting failed${reason}` },
                    },
                  });
                }
                break;
              }
              case "compact_boundary": {
                // Send used:0 immediately so the client doesn't keep showing
                // the stale pre-compaction context size until the next turn.
                //
                // This is a deliberate approximation: we don't know the exact
                // post-compaction token count (only the SDK's next API call
                // reveals that). But used:0 is directionally correct — context
                // just dropped dramatically — and the real value replaces it
                // within seconds when the next result message arrives.
                // The alternative (no update) leaves the client showing e.g.
                // "944k/1m" right after the user sees "Compacting completed",
                // which is confusing and wrong.
                //
                // The "Compacting completed." text is emitted from the `status`
                // handler (keyed on `compact_result`), not here, so the failure
                // path gets a message too.
                lastAssistantTotalUsage = 0;
                lastAssistantUsage = null;
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "usage_update",
                    used: 0,
                    size: session.contextWindowSize,
                  },
                });
                break;
              }
              case "local_command_output": {
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: message.content },
                  },
                });
                break;
              }
              case "session_state_changed": {
                if (message.state === "idle") {
                  if (session.cancelled) {
                    stopReason = "cancelled";
                  }
                  // Forward the terminal reason as a structured PromptResponse
                  // field too (CONCEPTION §3 "and/or a structured prompt-result
                  // field"). Falls back to mapping the final stopReason if no
                  // result message captured it.
                  const endReason: LastTurnEndReason =
                    lastTurnEndReason ??
                    (stopReason === "max_tokens"
                      ? "max_tokens"
                      : stopReason === "max_turn_requests"
                        ? "max_turns"
                        : stopReason === "cancelled"
                          ? "cancelled"
                          : "end_turn");
                  return {
                    stopReason,
                    usage: sessionUsage(session),
                    _meta: { [LAST_TURN_END_REASON_META_KEY]: endReason },
                  };
                }
                break;
              }
              case "task_started": {
                if (message.tool_use_id) {
                  await this.onTeammateSpawned(
                    message.tool_use_id,
                    message.task_id,
                    message.description,
                    params.sessionId,
                  );
                }
                break;
              }
              case "task_progress": {
                await this.onTaskProgress(
                  message.tool_use_id,
                  message.task_id,
                  message.last_tool_name,
                  params.sessionId,
                );
                break;
              }
              case "task_notification": {
                await this.onTaskNotification(
                  message.tool_use_id,
                  message.task_id,
                  message.status,
                  params.sessionId,
                );
                break;
              }
              case "memory_recall": {
                const isSynthesis = message.mode === "synthesize";
                const locations = isSynthesis
                  ? []
                  : message.memories.map((m) => ({ path: m.path }));
                const content = isSynthesis
                  ? message.memories
                      .filter(
                        (m): m is (typeof message.memories)[number] & { content: string } =>
                          typeof m.content === "string",
                      )
                      .map((m) => ({
                        type: "content" as const,
                        content: { type: "text" as const, text: m.content },
                      }))
                  : [];
                const count = message.memories.length;
                const title = isSynthesis
                  ? "Recalled synthesized memory"
                  : `Recalled ${count} ${count === 1 ? "memory" : "memories"}`;
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "tool_call",
                    toolCallId: message.uuid,
                    title,
                    kind: "read",
                    status: "completed",
                    ...(locations.length > 0 && { locations }),
                    ...(content.length > 0 && { content }),
                    _meta: {
                      claudeCode: {
                        toolName: "memory_recall",
                        toolResponse: { mode: message.mode },
                      },
                    } satisfies ToolUpdateMeta,
                  },
                });
                break;
              }
              case "hook_started":
              case "hook_progress":
              case "hook_response":
              case "files_persisted":
              case "task_updated":
              case "elicitation_complete":
              case "plugin_install":
              case "notification":
              case "api_retry":
              case "mirror_error":
              case "permission_denied":
              case "commands_changed":
                // `commands_changed` (new in the bundled CC ≥2.1.157): the SDK
                // pushes the full slash-command list after a mid-session change
                // (e.g. skills discovered dynamically while working in a
                // subdirectory). We snapshot commands once at init and surface
                // them via `sendAvailableCommandsUpdate`; live re-sync of the
                // command palette is not yet wired through ACP, so this is an
                // intentional no-op. No client-visible regression — the init
                // command list stays valid; commands discovered mid-session
                // simply aren't advertised until the next session load.
                // Todo: process via status api: https://docs.claude.com/en/docs/claude-code/hooks#hook-output
                break;
              case "model_refusal_fallback": {
                // New in the bundled CC ≥2.1.157. The active model refused this
                // turn and the SDK transparently fell back to another model
                // (`direction`: retry/revert/sticky). The refused partial was
                // already streamed to the client as append-only
                // `agent_message_chunk` text; `retracted_message_uuids` names the
                // SDK wire uuids the engine has now evicted so they aren't shown
                // as a real answer.
                //
                // The adapter streams assistant text WITHOUT exposing the SDK
                // wire uuids to the client, so a generic ACP client cannot today
                // key off `retracted_message_uuids` to evict the stale partial.
                // We therefore (a) log the fallback richly for operability — this
                // SDK drives every Claude session on the box, not just fable — and
                // (b) propagate the full retraction record out-of-band via an
                // `extNotification`, so a client that tracks message identity (or
                // a future adapter change that tags chunks with uuids) can act on
                // it. `extNotification` is non-invasive: clients that don't handle
                // the method ignore it, so there is no transcript pollution or
                // regression for existing clients.
                const retracted = message.retracted_message_uuids ?? [];
                this.logger.error(
                  `Session ${message.session_id}: model_refusal_fallback ` +
                    `(${message.direction}) ${message.original_model} -> ` +
                    `${message.fallback_model}` +
                    (message.api_refusal_category
                      ? ` [category: ${message.api_refusal_category}]`
                      : "") +
                    `; retracted ${retracted.length} message(s).`,
                );
                await this.client
                  .extNotification(MODEL_REFUSAL_FALLBACK_NOTIFICATION, {
                    sessionId: message.session_id,
                    direction: message.direction,
                    originalModel: message.original_model,
                    fallbackModel: message.fallback_model,
                    apiRefusalCategory: message.api_refusal_category ?? null,
                    apiRefusalExplanation: message.api_refusal_explanation ?? null,
                    retractedMessageUuids: retracted,
                    content: message.content,
                  })
                  .catch((err) => {
                    this.logger.error(
                      "Failed to forward model_refusal_fallback notification:",
                      err,
                    );
                  });
                break;
              }
              case "thinking_tokens": {
                // A running token-count *estimate* the SDK digests from thinking
                // pings (`estimated_tokens`) — not thinking text. On redacted-
                // thinking models (e.g. opus) it is the *only* progress signal:
                // the engine streams a signed thinking block but no plaintext.
                // Forward it as a throttled `_meta` hint (NOT thought text, and
                // coalesced because it fires per stream frame) so clients can
                // render a "thinking… ~N tokens" indicator. Non-redacted thinking
                // text still flows via the stream_event `thinking_delta` path
                // (see toAcpNotifications).
                const estimated = message.estimated_tokens;
                const now = Date.now();
                if (
                  estimated > lastThinkingTokensValue &&
                  (now - lastThinkingTokensAt >= THINKING_TOKENS_THROTTLE_MS ||
                    estimated - lastThinkingTokensValue >= THINKING_TOKENS_MIN_DELTA)
                ) {
                  lastThinkingTokensAt = now;
                  lastThinkingTokensValue = estimated;
                  await this.client.sessionUpdate({
                    sessionId: message.session_id,
                    update: {
                      sessionUpdate: "agent_thought_chunk",
                      content: { type: "text", text: "" },
                      _meta: { claudeCode: { thinkingTokens: estimated } },
                    },
                  });
                }
                break;
              }
              default:
                unreachable(message, this.logger);
                break;
            }
            break;
          case "result": {
            // Accumulate usage from this result
            session.accumulatedUsage.inputTokens += message.usage.input_tokens;
            session.accumulatedUsage.outputTokens += message.usage.output_tokens;
            session.accumulatedUsage.cachedReadTokens += message.usage.cache_read_input_tokens;
            session.accumulatedUsage.cachedWriteTokens += message.usage.cache_creation_input_tokens;

            const matchingModelUsage = lastAssistantModel
              ? getMatchingModelUsage(message.modelUsage, lastAssistantModel)
              : null;
            // Only overwrite when we have an authoritative value — a miss
            // (e.g. a turn with no top-level assistant message) would
            // otherwise discard the window learned on a prior turn and
            // leave the next prompt's mid-stream updates reporting 200k.
            if (matchingModelUsage) {
              session.contextWindowSize = matchingModelUsage.contextWindow;
            }

            // Task-notification followups are autonomous work triggered by a
            // task-notification system message, not by the user's prompt.
            // They should not influence the user-turn lifecycle (stop reason,
            // slash-command output forwarding) but their cost is real.
            const isTaskNotification = message.origin?.kind === "task-notification";

            // Capture the terminal turn reason (CONCEPTION §3, D3) so it can be
            // forwarded to the client on the final usage_update and the
            // PromptResponse. Mirrors the stop reasons the switch below derives.
            // Task-notification followups are autonomous and never set it.
            if (!isTaskNotification) {
              if (message.subtype === "success" || message.subtype === "error_during_execution") {
                lastTurnEndReason =
                  message.stop_reason === "max_tokens"
                    ? "max_tokens"
                    : message.is_error
                      ? "error"
                      : "end_turn";
              } else {
                // error_max_turns | error_max_budget_usd | error_max_structured_output_retries
                lastTurnEndReason = "max_turns";
              }
            }

            // Send usage_update notification. Carry the terminal turn reason in
            // `_meta` so the acpx-ui server reads it off the same `usage_update`
            // stream-tail it already treats as the turn-end marker (§2.2).
            if (lastAssistantTotalUsage !== null) {
              await this.client.sessionUpdate({
                sessionId: params.sessionId,
                update: {
                  sessionUpdate: "usage_update",
                  used: lastAssistantTotalUsage,
                  size: session.contextWindowSize,
                  cost: {
                    amount: message.total_cost_usd,
                    currency: "USD",
                  },
                  ...((message.origin || lastTurnEndReason) && {
                    _meta: {
                      ...(message.origin && { "_claude/origin": message.origin }),
                      ...(lastTurnEndReason && {
                        [LAST_TURN_END_REASON_META_KEY]: lastTurnEndReason,
                      }),
                    },
                  }),
                },
              });
            }

            if (session.cancelled) {
              if (!isTaskNotification) {
                stopReason = "cancelled";
              }
              break;
            }

            switch (message.subtype) {
              case "success": {
                if (message.result.includes("Please run /login")) {
                  throw RequestError.authRequired();
                }
                if (message.stop_reason === "max_tokens") {
                  if (!isTaskNotification) {
                    stopReason = "max_tokens";
                  }
                  break;
                }
                if (message.is_error) {
                  throw RequestError.internalError(
                    errorKindData(lastAssistantError),
                    message.result,
                  );
                }
                // For local-only commands (no model invocation), the result
                // text is the command output — forward it to the client.
                // Task-notification followups never originate from a user
                // slash command, so skip the forwarding for them.
                if (isLocalOnlyCommand && !isTaskNotification) {
                  for (const notification of toAcpNotifications(
                    message.result,
                    "assistant",
                    params.sessionId,
                    this.toolUseCache,
                    this.client,
                    this.logger,
                  )) {
                    await this.client.sessionUpdate(notification);
                  }
                }
                break;
              }
              case "error_during_execution": {
                if (message.stop_reason === "max_tokens") {
                  if (!isTaskNotification) {
                    stopReason = "max_tokens";
                  }
                  break;
                }
                if (message.is_error) {
                  throw RequestError.internalError(
                    errorKindData(lastAssistantError),
                    message.errors.join(", ") || message.subtype,
                  );
                }
                if (!isTaskNotification) {
                  stopReason = "end_turn";
                }
                break;
              }
              case "error_max_budget_usd":
              case "error_max_turns":
              case "error_max_structured_output_retries":
                if (message.is_error) {
                  throw RequestError.internalError(
                    errorKindData(lastAssistantError),
                    message.errors.join(", ") || message.subtype,
                  );
                }
                if (!isTaskNotification) {
                  stopReason = "max_turn_requests";
                }
                break;
              default:
                unreachable(message, this.logger);
                break;
            }
            break;
          }
          case "stream_event": {
            if (
              message.parent_tool_use_id === null &&
              (message.event.type === "message_start" || message.event.type === "message_delta")
            ) {
              if (message.event.type === "message_start") {
                lastAssistantUsage = snapshotFromUsage(message.event.message.usage);
                const model = message.event.message.model;
                if (model && model !== "<synthetic>") {
                  lastAssistantModel = model;
                  // Only upgrade from the default — once a `result` has given
                  // us an authoritative window, trust it over the heuristic.
                  // Model switches invalidate the cached window via
                  // `syncSessionConfigState`, which resets us back to the
                  // default so this branch runs again for the new model.
                  if (session.contextWindowSize === DEFAULT_CONTEXT_WINDOW) {
                    const inferred = inferContextWindowFromModel(model);
                    if (inferred !== null) {
                      session.contextWindowSize = inferred;
                    }
                  }
                }
              } else {
                const usage = message.event.usage;
                const prev: Readonly<UsageSnapshot> = lastAssistantUsage ?? ZERO_USAGE;
                // Per Anthropic API, message_delta usage fields are *cumulative*;
                // nullable fields (input_tokens and the cache fields) fall back
                // to the prior snapshot when the server omits them from this
                // delta. Only output_tokens is guaranteed non-null.
                lastAssistantUsage = {
                  input_tokens: usage.input_tokens ?? prev.input_tokens,
                  output_tokens: usage.output_tokens,
                  cache_read_input_tokens:
                    usage.cache_read_input_tokens ?? prev.cache_read_input_tokens,
                  cache_creation_input_tokens:
                    usage.cache_creation_input_tokens ?? prev.cache_creation_input_tokens,
                };
              }

              const nextUsage = totalTokens(lastAssistantUsage);
              if (nextUsage !== lastAssistantTotalUsage) {
                lastAssistantTotalUsage = nextUsage;
                await this.client.sessionUpdate({
                  sessionId: params.sessionId,
                  update: {
                    sessionUpdate: "usage_update",
                    used: nextUsage,
                    size: session.contextWindowSize,
                  },
                });
              }
            }
            for (const notification of streamEventToAcpNotifications(
              message,
              params.sessionId,
              this.toolUseCache,
              this.client,
              this.logger,
              {
                clientCapabilities: this.clientCapabilities,
                cwd: session.cwd,
                taskState: session.taskState,
                subagentCache: this.subagentCache,
              },
            )) {
              await this.client.sessionUpdate(notification);
            }
            break;
          }
          case "user":
          case "assistant": {
            if (session.cancelled) {
              break;
            }

            // Check for prompt replay
            if (message.type === "user" && "uuid" in message && message.uuid) {
              if (message.uuid === promptUuid) {
                break;
              }

              const pending = session.pendingMessages.get(message.uuid as string);
              if (pending) {
                pending.resolve(false);
                session.pendingMessages.delete(message.uuid as string);
                handedOff = true;
                // the current loop stops with end_turn,
                // the loop of the next prompt continues running
                return { stopReason: "end_turn", usage: sessionUsage(session) };
              }
              if ("isReplay" in message && message.isReplay) {
                // not pending or unrelated replay message
                break;
              }
            }

            // Snapshot the latest top-level assistant usage and model so the
            // next `result` can emit a usage_update tied to the right context
            // window. Subagent messages are excluded to keep the snapshot
            // aligned with what the user's current selection is producing.
            if (message.type === "assistant" && message.parent_tool_use_id === null) {
              lastAssistantUsage = snapshotFromUsage(message.message.usage);
              lastAssistantTotalUsage = totalTokens(lastAssistantUsage);
              if (message.message.model && message.message.model !== "<synthetic>") {
                lastAssistantModel = message.message.model;
              }
              if (message.error) {
                lastAssistantError = message.error;
              }
            }

            // Strip <command-*>/<local-command-stdout> markers and render any
            // remaining prose. Skill bodies and built-in slash commands (e.g.
            // /usage, /status, /model) arrive wrapped in these tags; pure-marker
            // payloads (e.g. /compact's malformed output) strip to null and are
            // skipped. Mirrors the replay path at replaySessionHistory.
            if (
              message.message.role !== "system" &&
              typeof message.message.content === "string" &&
              message.message.content.includes("<local-command-stdout>")
            ) {
              const stripped = stripLocalCommandMetadata(message.message.content);
              if (typeof stripped === "string") {
                for (const notification of toAcpNotifications(
                  stripped,
                  message.message.role,
                  params.sessionId,
                  this.toolUseCache,
                  this.client,
                  this.logger,
                  {
                    clientCapabilities: this.clientCapabilities,
                    parentToolUseId: message.parent_tool_use_id,
                    cwd: session.cwd,
                    taskState: session.taskState,
                  },
                )) {
                  await this.client.sessionUpdate(notification);
                }
              } else {
                this.logger.log(message.message.content);
              }
              break;
            }

            if (
              typeof message.message.content === "string" &&
              message.message.content.includes("<local-command-stderr>")
            ) {
              this.logger.error(message.message.content);
              break;
            }
            // Skip these user messages for now, since they seem to just be messages we don't want in the feed
            if (
              message.type === "user" &&
              (typeof message.message.content === "string" ||
                (Array.isArray(message.message.content) &&
                  message.message.content.length === 1 &&
                  message.message.content[0].type === "text"))
            ) {
              break;
            }
            if (message.message.role === "system") {
              break;
            }

            if (
              message.type === "assistant" &&
              message.message.model === "<synthetic>" &&
              Array.isArray(message.message.content) &&
              message.message.content.length === 1 &&
              message.message.content[0].type === "text" &&
              message.message.content[0].text.includes("Please run /login")
            ) {
              throw RequestError.authRequired();
            }

            const content =
              message.type === "assistant"
                ? // Handled by stream events above
                  message.message.content.filter(
                    (item) => !["text", "thinking"].includes(item.type),
                  )
                : message.message.content;

            for (const notification of toAcpNotifications(
              content,
              message.message.role,
              params.sessionId,
              this.toolUseCache,
              this.client,
              this.logger,
              {
                clientCapabilities: this.clientCapabilities,
                parentToolUseId: message.parent_tool_use_id,
                cwd: session.cwd,
                taskState: session.taskState,
                subagentCache: this.subagentCache,
              },
            )) {
              await this.client.sessionUpdate(notification);
            }
            break;
          }
          case "tool_progress":
          case "tool_use_summary":
          case "auth_status":
          case "prompt_suggestion":
          case "rate_limit_event":
            break;
          default:
            unreachable(message);
            break;
        }
      }
      throw new Error("Session did not end in result");
    } catch (error) {
      errored = true;
      // A failed turn typically leaves a trailing `session_state_changed: idle`
      // (and possibly more) in the query iterator. If we don't drain it here,
      // the next prompt's first `query.next()` consumes that stale idle and
      // short-circuits to end_turn with zero usage
      // Bounded so a misbehaving SDK can't hang the next prompt indefinitely.
      try {
        await session.query.interrupt();
      } catch (drainErr) {
        this.logger.error(
          `Session ${params.sessionId}: failed to drain query after prompt error:`,
          drainErr,
        );
      }

      if (error instanceof RequestError || !(error instanceof Error)) {
        throw error;
      }
      const message = error.message;
      if (
        message.includes("ProcessTransport") ||
        message.includes("terminated process") ||
        message.includes("process exited with") ||
        message.includes("process terminated by signal") ||
        message.includes("Failed to write to process stdin")
      ) {
        this.logger.error(`Session ${params.sessionId}: Claude Agent process died: ${message}`);
        session.settingsManager.dispose();
        session.input.end();
        delete this.sessions[params.sessionId];
        throw RequestError.internalError(
          undefined,
          "The Claude Agent process exited unexpectedly. Please start a new session.",
        );
      }
      throw error;
    } finally {
      // Stop the turn no-activity timer for this prompt.
      clearInterval(noActivityTimer);
      // Always clear the resolve callback so the background loop switches to
      // idle mode (forwarding inter-turn activity) when this prompt exits.
      session.activePromptResolve = null;

      if (!handedOff) {
        if (errored) {
          session.promptRunning = false;
          // The query stream was just drained — handing pending prompts or
          // buffered messages off onto it would let them race with the recovery.
          // Discard the buffer and cancel the pendings so each waiting prompt()
          // returns stopReason: "cancelled" and the client can decide to retry.
          session.pendingSdkMessages.length = 0;
          for (const pending of session.pendingMessages.values()) {
            pending.resolve(true);
          }
          session.pendingMessages.clear();
        } else {
          // Clean exit (idle-status return, or stream end without error). Drain
          // any messages the reader buffered after the loop stopped parking (A1
          // lifecycle rule): a buffered user-replay whose uuid matches a parked
          // prompt hands the stream to that prompt (same as the mid-loop handoff
          // at :1517) and leaves the rest of the buffer for its successor;
          // anything else is a genuine inter-turn update routed to idle handling.
          let drainedHandoff = false;
          while (session.pendingSdkMessages.length > 0) {
            const buffered = session.pendingSdkMessages.shift();
            if (!buffered) continue; // null sentinel — stream end, nothing to route
            if (
              buffered.type === "user" &&
              "uuid" in buffered &&
              buffered.uuid &&
              session.pendingMessages.has(buffered.uuid as string)
            ) {
              const uuid = buffered.uuid as string;
              const pending = session.pendingMessages.get(uuid)!;
              pending.resolve(false);
              session.pendingMessages.delete(uuid);
              handedOff = true;
              drainedHandoff = true;
              break; // successor prompt owns the stream + remaining buffer
            }
            if (
              session.emitRawSDKMessages &&
              shouldEmitRawMessage(session.emitRawSDKMessages, buffered)
            ) {
              await this.client.extNotification("_claude/sdkMessage", {
                sessionId: params.sessionId,
                message: buffered as Record<string, unknown>,
              });
            }
            await this.handleIdleMessage(buffered, params.sessionId);
          }
          if (!drainedHandoff) {
            session.promptRunning = false;
            // Last-resort band-aid retained: if the loop finished without the
            // SDK replaying a still-parked prompt's message, release the oldest
            // so it doesn't get stuck.
            if (session.pendingMessages.size > 0) {
              const next = [...session.pendingMessages.entries()].sort(
                (a, b) => a[1].order - b[1].order,
              )[0];
              if (next) {
                next[1].resolve(false);
                session.pendingMessages.delete(next[0]);
              }
            }
          }
        }
      }
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      return;
    }
    session.cancelled = true;
    for (const [, pending] of session.pendingMessages) {
      pending.resolve(true);
    }
    session.pendingMessages.clear();
    // Discard any buffered mid-await SDK messages (A1 lifecycle rule): cancel
    // throws away in-flight turn state, mirroring the query.interrupt() stale
    // stream-drain below. teardownSession() reaches this via cancel() too.
    session.pendingSdkMessages.length = 0;
    // Cancel hygiene (CONCEPTION §4.6): also unblock the ACTIVE prompt loop's
    // pending `nextMessage()` so it observes `cancelled`, returns
    // `{stopReason:'cancelled'}`, and clears `promptRunning` in its finally.
    // Without this the loop stays parked at `await nextMessage()` even after a
    // cancel, so `promptRunning` never clears and newly-parked prompts keep
    // ACCUMULATING — the wedge grows and stays undetectable. This does NOT
    // un-wedge the SDK child: the background reader loop is still stuck at
    // `query.next()` and `interrupt()` can't force it to yield, so the *next*
    // prompt would re-wedge. The guaranteed cure is an external force-restart.
    if (session.activePromptResolve) {
      const resolve = session.activePromptResolve;
      session.activePromptResolve = null;
      resolve(null);
    }
    await session.query.interrupt();
  }

  /** Cleanly tear down a session: cancel in-flight work, dispose resources,
   *  and remove it from the session map. */
  private async teardownSession(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    await this.cancel({ sessionId });
    session.settingsManager.dispose();
    session.abortController.abort();
    session.query.close();
    delete this.sessions[sessionId];
  }

  /** Tear down all active sessions. Called when the ACP connection closes. */
  async dispose(): Promise<void> {
    await Promise.all(Object.keys(this.sessions).map((id) => this.teardownSession(id)));
  }

  async closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }
    await this.teardownSession(params.sessionId);
    return {};
  }

  async unstable_deleteSession(params: DeleteSessionRequest): Promise<DeleteSessionResponse> {
    // Tear down any active in-memory state first so the on-disk file isn't
    // recreated by an outstanding query writing to it.
    if (this.sessions[params.sessionId]) {
      await this.teardownSession(params.sessionId);
    }
    await deleteSession(params.sessionId);
    return {};
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest,
  ): Promise<SetSessionModelResponse | void> {
    const session = await this.resolveSessionForConfigOp(params.sessionId);
    if (!session) {
      throw new Error("Session not found");
    }
    // Resolve aliases (e.g. "opus", "opus[1m]") to canonical model IDs so
    // downstream lookups in modelInfos succeed and the effort option isn't
    // silently dropped.
    const resolved = resolveModelPreference(session.modelInfos, params.modelId);
    const modelId = resolved?.value ?? params.modelId;
    // Re-attach an explicit `[1m]` context hint the base ModelInfo dropped, so
    // the binary receives e.g. "sonnet[1m]" (strips the suffix + enables the
    // long-context beta) instead of the stripped "sonnet" (which runs at 200k).
    // Storing the `[1m]` form as currentModelId also makes the reported window
    // 1M (inferContextWindowFromModel matches `\b1m\b`) and keeps the alias in
    // the advertised set so acpx's exact-string replay gate accepts it on
    // resume. A plain pick (no hint) is returned untouched.
    const runModelId = effectiveRunModelId(params.modelId, modelId);
    // When the selection is the synthesized box-default entry, the literal
    // string "default" is NOT a real SDK model id — map it to the SDK's
    // documented "use the default" (`setModel(undefined)`), the same reset the
    // box-default path in getAvailableModels uses. A CONCRETE model
    // (sonnet/haiku/opus/fable/...) still applies that exact value. We persist
    // the resume-safe "default" id either way, never `undefined`.
    const isBoxDefault =
      resolved?.value === "default" || params.modelId === "" || params.modelId === "default";
    await session.query.setModel(isBoxDefault ? undefined : runModelId);
    await this.updateConfigOption(params.sessionId, "model", runModelId);
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    if (!(await this.resolveSessionForConfigOp(params.sessionId))) {
      throw new Error("Session not found");
    }

    await this.applySessionMode(params.sessionId, params.modeId);
    await this.updateConfigOption(params.sessionId, "mode", params.modeId);
    return {};
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const session = await this.resolveSessionForConfigOp(params.sessionId);
    if (!session) {
      throw new Error("Session not found");
    }
    if (typeof params.value !== "string") {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    const option = session.configOptions.find((o) => o.id === params.configId);
    if (!option) {
      throw new Error(`Unknown config option: ${params.configId}`);
    }

    const allValues =
      "options" in option && Array.isArray(option.options)
        ? option.options.flatMap((o) => ("options" in o ? o.options : [o]))
        : [];
    let validValue = allValues.find((o) => o.value === params.value);

    // For model options, fall back to resolveModelPreference when the exact
    // value doesn't match.  This lets callers use human-friendly aliases like
    // "opus" or "sonnet" instead of full model IDs like "claude-opus-4-6".
    if (!validValue && params.configId === "model") {
      const modelInfos: ModelInfo[] = allValues.map((o) => ({
        value: o.value,
        displayName: o.name,
        description: o.description ?? "",
      }));
      const resolved = resolveModelPreference(modelInfos, params.value);
      if (resolved) {
        validValue = allValues.find((o) => o.value === resolved.value);
      }
    }

    if (!validValue) {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    // Use the canonical option value so downstream code always receives the
    // model ID rather than the caller-supplied alias.
    const resolvedValue = validValue.value;
    // Preserve an explicit `[1m]` context hint on a model pick — same rationale
    // as unstable_setSessionModel: the resolved option value is the stripped
    // base ("sonnet"), so forwarding it verbatim would run 200k despite a
    // "sonnet[1m]" request. Re-attach the hint for the value applied to the SDK
    // and stored as currentModelId; non-model options are unaffected.
    const appliedValue =
      params.configId === "model"
        ? effectiveRunModelId(params.value, resolvedValue)
        : resolvedValue;

    if (params.configId === "mode") {
      await this.applySessionMode(params.sessionId, appliedValue);
      await this.client.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "current_mode_update",
          currentModeId: appliedValue,
        },
      });
    } else if (params.configId === "model") {
      await this.sessions[params.sessionId].query.setModel(appliedValue);
    }
    // Effort SDK sync is handled inside applyConfigOptionValue so that direct
    // effort changes and effort changes induced by a model switch go through
    // the same path.

    await this.applyConfigOptionValue(params.sessionId, session, params.configId, appliedValue);

    return { configOptions: session.configOptions };
  }

  private async applySessionMode(sessionId: string, modeId: string): Promise<void> {
    switch (modeId) {
      case "auto":
      case "default":
      case "acceptEdits":
      case "bypassPermissions":
      case "dontAsk":
      case "plan":
        break;
      default:
        throw new Error("Invalid Mode");
    }

    const session = this.sessions[sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    if (!session.modes.availableModes.some((mode) => mode.id === modeId)) {
      throw new Error(`Mode ${modeId} is not available in this session`);
    }

    try {
      await session.query.setPermissionMode(modeId);
    } catch (error) {
      if (error instanceof Error) {
        if (!error.message) {
          error.message = "Invalid Mode";
        }
        throw error;
      } else {
        // eslint-disable-next-line preserve-caught-error
        throw new Error("Invalid Mode");
      }
    }
  }

  private async replaySessionHistory(sessionId: string): Promise<void> {
    const toolUseCache: ToolUseCache = {};
    const messages = await getSessionMessages(sessionId);

    for (const message of messages) {
      // @ts-expect-error - untyped in SDK but we handle all of these
      let content: unknown = message.message.content;
      // @ts-expect-error - untyped in SDK but we handle all of these
      if (message.message.role === "user") {
        content = stripLocalCommandMetadata(content);
        if (content === null) continue;
      }

      for (const notification of toAcpNotifications(
        // @ts-expect-error - untyped in SDK but we handle all of these
        content,
        // @ts-expect-error - untyped in SDK but we handle all of these
        message.message.role,
        sessionId,
        toolUseCache,
        this.client,
        this.logger,
        {
          registerHooks: false,
          clientCapabilities: this.clientCapabilities,
          cwd: this.sessions[sessionId]?.cwd,
          taskState: this.sessions[sessionId]?.taskState,
        },
      )) {
        await this.client.sessionUpdate(notification);
      }
    }
  }

  async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    const response = await this.client.readTextFile(params);
    return response;
  }

  async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    const response = await this.client.writeTextFile(params);
    return response;
  }

  canUseTool(sessionId: string): CanUseTool {
    return async (toolName, toolInput, { signal, suggestions, toolUseID }) => {
      const alwaysAllowLabel = describeAlwaysAllow(suggestions, toolName);
      const supportsTerminalOutput = this.clientCapabilities?._meta?.["terminal_output"] === true;
      const session = this.sessions[sessionId];
      if (!session) {
        return {
          behavior: "deny",
          message: "Session not found",
        };
      }

      if (toolName === "ExitPlanMode") {
        const optionsAll: PermissionOption[] = [
          { kind: "allow_always", name: 'Yes, and use "auto" mode', optionId: "auto" },
          {
            kind: "allow_always",
            name: "Yes, and auto-accept edits",
            optionId: "acceptEdits",
          },
          { kind: "allow_once", name: "Yes, and manually approve edits", optionId: "default" },
          { kind: "reject_once", name: "No, keep planning", optionId: "plan" },
        ];
        if (ALLOW_BYPASS) {
          optionsAll.unshift({
            kind: "allow_always",
            name: "Yes, and bypass permissions",
            optionId: "bypassPermissions",
          });
        }
        // Filter against the session's currently-advertised modes so we never
        // present options the active model can't honor (e.g. `auto` on Haiku).
        // `bypassPermissions` is already covered by `availableModes` via
        // `buildAvailableModes`/`ALLOW_BYPASS`. The `plan` option is a
        // "keep planning" reject path; it's always present in `availableModes`.
        const options = optionsAll.filter((o) =>
          session.modes.availableModes.some((m) => m.id === o.optionId),
        );

        const response = await this.client.requestPermission({
          options,
          sessionId,
          toolCall: {
            toolCallId: toolUseID,
            rawInput: toolInput,
            ...toolInfoFromToolUse(
              { name: toolName, input: toolInput, id: toolUseID },
              supportsTerminalOutput,
              session?.cwd,
            ),
          },
        });

        if (signal.aborted || response.outcome?.outcome === "cancelled") {
          throw new Error("Tool use aborted");
        }
        const selectedMode =
          response.outcome?.outcome === "selected" ? response.outcome.optionId : undefined;
        const selectedModeWasOffered = options.some((option) => option.optionId === selectedMode);
        if (
          selectedModeWasOffered &&
          (selectedMode === "default" ||
            selectedMode === "acceptEdits" ||
            selectedMode === "auto" ||
            selectedMode === "bypassPermissions")
        ) {
          await this.client.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "current_mode_update",
              currentModeId: selectedMode,
            },
          });
          await this.updateConfigOption(sessionId, "mode", selectedMode);

          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              { type: "setMode", mode: selectedMode, destination: "session" },
            ],
          };
        } else {
          return {
            behavior: "deny",
            message: "User rejected request to exit plan mode.",
          };
        }
      }

      if (session.modes.currentModeId === "bypassPermissions") {
        return {
          behavior: "allow",
          updatedInput: toolInput,
          updatedPermissions: suggestions ?? [
            { type: "addRules", rules: [{ toolName }], behavior: "allow", destination: "session" },
          ],
        };
      }

      const response = await this.client.requestPermission({
        options: [
          {
            kind: "allow_always",
            name: alwaysAllowLabel,
            optionId: "allow_always",
          },
          { kind: "allow_once", name: "Allow", optionId: "allow" },
          { kind: "reject_once", name: "Reject", optionId: "reject" },
        ],
        sessionId,
        toolCall: {
          toolCallId: toolUseID,
          rawInput: toolInput,
          ...toolInfoFromToolUse(
            { name: toolName, input: toolInput, id: toolUseID },
            supportsTerminalOutput,
            session?.cwd,
          ),
        },
      });
      if (signal.aborted || response.outcome?.outcome === "cancelled") {
        throw new Error("Tool use aborted");
      }
      if (
        response.outcome?.outcome === "selected" &&
        (response.outcome.optionId === "allow" || response.outcome.optionId === "allow_always")
      ) {
        // If Claude Code has suggestions, it will update their settings already
        if (response.outcome.optionId === "allow_always") {
          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              {
                type: "addRules",
                rules: [{ toolName }],
                behavior: "allow",
                destination: "session",
              },
            ],
          };
        }
        return {
          behavior: "allow",
          updatedInput: toolInput,
        };
      } else {
        return {
          behavior: "deny",
          message: "User refused permission to run tool",
        };
      }
    };
  }

  private async sendAvailableCommandsUpdate(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;
    const commands = await session.query.supportedCommands();
    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands: getAvailableSlashCommands(commands),
      },
    });
  }

  private async updateConfigOption(
    sessionId: string,
    configId: string,
    value: string,
  ): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;

    await this.applyConfigOptionValue(sessionId, session, configId, value);

    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions: session.configOptions,
      },
    });
  }

  private async applyConfigOptionValue(
    sessionId: string,
    session: Session,
    configId: string,
    value: string,
  ): Promise<void> {
    if (configId === "mode") {
      session.modes = { ...session.modes, currentModeId: value };
      session.configOptions = session.configOptions.map((o) =>
        o.id === configId && typeof o.currentValue === "string" ? { ...o, currentValue: value } : o,
      );
    } else if (configId === "model") {
      // Resolve the new model's `ModelInfo` once: its `description` feeds the
      // context-window heuristic below, and `supportsAutoMode` the mode clamp.
      // Tolerate a `[1m]` hint on `value` that the base entry lacks (mid-session
      // switch to "sonnet[1m]" when modelInfos holds only the base "sonnet").
      const newModelInfo = findModelInfoById(session.modelInfos, value);
      if (session.models.currentModelId !== value) {
        // The cached context window was learned for the previous model; reset
        // to the new model's heuristic so mid-stream updates between now and
        // the next `result` reflect the user's selection instead of the old
        // model's window. Pass the description so the 1M-context `default`
        // model is told apart from plain `opus` (they share a base model id).
        session.contextWindowSize =
          inferContextWindowFromModel(value, newModelInfo?.description) ?? DEFAULT_CONTEXT_WINDOW;
      }
      session.models = { ...session.models, currentModelId: value };

      // Recompute availableModes for the new model and clamp the current
      // mode if the SDK no longer offers it (today: "auto" on Haiku).
      // `ModelInfo.supportsAutoMode` is the canonical SDK signal.
      const newAvailableModes = buildAvailableModes(newModelInfo);
      // Capture BEFORE mutating session.modes so the log message reflects
      // the invalidated mode rather than "default".
      const previousModeId = session.modes.currentModeId;
      let modeDowngraded = false;
      if (!newAvailableModes.some((m) => m.id === previousModeId)) {
        session.modes = {
          availableModes: newAvailableModes,
          currentModeId: "default",
        };
        try {
          await session.query.setPermissionMode("default");
        } catch (err) {
          // Failing the entire model switch over a bookkeeping sync error is
          // worse UX than logging and continuing; the user explicitly asked
          // to change models. The next setPermissionMode from the user will
          // either succeed or surface a fresh error.
          this.logger.error(
            `Failed to sync permissionMode to "default" after model switch invalidated "${previousModeId}":`,
            err,
          );
        }
        modeDowngraded = true;
      } else {
        session.modes = { ...session.modes, availableModes: newAvailableModes };
      }

      // Rebuild config options since effort levels depend on the selected model
      const effortOpt = session.configOptions.find((o) => o.id === "effort");
      const currentEffort =
        typeof effortOpt?.currentValue === "string" ? effortOpt.currentValue : undefined;
      session.configOptions = buildConfigOptions(
        session.modes,
        session.models,
        session.modelInfos,
        currentEffort,
      );

      // Sync effort with the SDK if it changed after the model switch
      const newEffortOpt = session.configOptions.find((o) => o.id === "effort");
      const newEffort =
        typeof newEffortOpt?.currentValue === "string" ? newEffortOpt.currentValue : undefined;
      if (newEffort !== currentEffort) {
        await session.query.applyFlagSettings({
          effortLevel: toSdkEffortLevel(newEffort),
        });
      }

      // Emit current_mode_update only after session.modes AND
      // session.configOptions have been fully reconciled. This way, a failure
      // in the configOptions/effort rebuild above can't leave the client with
      // a clamped currentModeId but stale configOptions, and the notification
      // still precedes the caller's config_option_update so order-sensitive
      // clients update currentModeId before re-rendering the option list.
      if (modeDowngraded) {
        await this.client.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "current_mode_update",
            currentModeId: "default",
          },
        });
      }
    } else {
      session.configOptions = session.configOptions.map((o) =>
        o.id === configId && typeof o.currentValue === "string" ? { ...o, currentValue: value } : o,
      );
      if (configId === "effort") {
        await session.query.applyFlagSettings({
          effortLevel: toSdkEffortLevel(value),
        });
      }
    }
  }

  /**
   * Resolve the in-memory session for a per-session config op (set_model /
   * set_mode / set_config_option), lazily resuming a forked session that acpx
   * minted out-of-band and never registered here (see `lastForkContext`).
   *
   * acpx's Claude copy path replaces our fork id with its own durable forked
   * transcript id, then calls these ops on that id over the same connection
   * with no preceding `session/resume`. The durable transcript exists on disk,
   * so we resume it here using the originating fork's cwd/mcpServers/_meta.
   *
   * Returns undefined only for a genuinely-unknown id (no `lastForkContext`) —
   * callers surface the usual "Session not found", so that case behaves as
   * before. But when a lazy resume-from-disk actually THROWS, we now re-throw
   * the underlying error instead of collapsing it to undefined. Previously the
   * throw was swallowed and callers reported an opaque "Session not found",
   * hiding the real reason — which made non-default-model fork creation fail
   * with no diagnosable cause (fork brick 29efbe0c). Surfacing the error lets it
   * reach acpx/the UI.
   */
  private async resolveSessionForConfigOp(sessionId: string): Promise<Session | undefined> {
    const existing = this.sessions[sessionId];
    if (existing) {
      return existing;
    }
    const ctx = this.lastForkContext;
    if (!ctx) {
      return undefined;
    }
    try {
      // TODO(fork brick 29efbe0c, staging step-0): `ctx.cwd` is the fork
      // *request* cwd. For a cross-cwd Claude copy that is the SOURCE cwd, while
      // acpx materializes the durable transcript at the DESTINATION cwd — so this
      // resume can look in the wrong project dir (candidate cause "B1"). Not
      // changed here because it is unconfirmed: the staging step-0 adapter log
      // (this catch's error line) must first show whether the throw is B1 (cwd),
      // B2 (config-dir/subscription) or B3 (transcript not resumable). Do not
      // guess the cwd change before that evidence.
      await this.getOrCreateSession({
        sessionId,
        cwd: ctx.cwd,
        mcpServers: ctx.mcpServers,
        additionalDirectories: ctx.additionalDirectories,
        _meta: ctx._meta,
      });
    } catch (error) {
      this.logger.error(`Session ${sessionId}: lazy resume of forked session failed:`, error);
      const detail = error instanceof Error ? error.message : String(error);
      throw RequestError.internalError(
        undefined,
        `Failed to resume forked session ${sessionId} for config op: ${detail}`,
      );
    }
    return this.sessions[sessionId];
  }

  private async getOrCreateSession(params: {
    sessionId: string;
    cwd: string;
    mcpServers?: NewSessionRequest["mcpServers"];
    additionalDirectories?: NewSessionRequest["additionalDirectories"];
    _meta?: NewSessionRequest["_meta"];
  }): Promise<NewSessionResponse> {
    const existingSession = this.sessions[params.sessionId];
    if (existingSession) {
      const fingerprint = computeSessionFingerprint(params);
      if (fingerprint === existingSession.sessionFingerprint) {
        return {
          sessionId: params.sessionId,
          modes: existingSession.modes,
          models: existingSession.models,
          configOptions: existingSession.configOptions,
        };
      }

      // Session-defining params changed (e.g. cwd pointed at a git worktree,
      // or MCP servers reconfigured). Tear down the existing session and
      // recreate it so the underlying Query process picks up the new values.
      await this.teardownSession(params.sessionId);
    }

    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        additionalDirectories: params.additionalDirectories,
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
      },
    );

    return {
      sessionId: response.sessionId,
      modes: response.modes,
      models: response.models,
      configOptions: response.configOptions,
    };
  }

  /**
   * Await an init/resume promise (`q.initializationResult()`) but BOUNDED
   * (CONCEPTION §3, D2):
   *   • emit a `_claude/sessionStatus` heartbeat every RESUME_HEARTBEAT_MS so a
   *     long resume never looks dead and the client stream keeps advancing
   *     (defeating staleness) — without changing the `session/update` stream-tail
   *     kind the server uses to tell `resuming` from `working`; and
   *   • HARD-FAIL at INIT_HARD_MS with a STRUCTURED terminal error instead of
   *     hanging forever, aborting + closing the orphaned query so its child
   *     process doesn't leak (the session isn't registered yet, so
   *     teardownSession can't reach it).
   * This SURFACES + bounds the hang; it is not the cure — a natively-blocked
   * child is only cured by an external process force-restart.
   */
  private async awaitInitializationBounded<T>(
    initPromise: Promise<T>,
    opts: { sessionId: string; isResume: boolean; abort: () => void; close: () => void },
  ): Promise<T> {
    const { sessionId, isResume } = opts;
    const startedAt = Date.now();
    const heartbeat = setInterval(() => {
      const elapsedMs = Date.now() - startedAt;
      const seconds = Math.round(elapsedMs / 1000);
      void this.client
        .extNotification(SESSION_STATUS_NOTIFICATION, {
          sessionId,
          phase: isResume ? "resuming" : "initializing",
          elapsedMs,
          hardLimitMs: INIT_HARD_MS,
          message: `${isResume ? "still resuming" : "still initializing"} — ${seconds}s`,
        })
        .catch(() => {});
    }, RESUME_HEARTBEAT_MS);
    let hardTimer: ReturnType<typeof setTimeout> | undefined;
    let timedOut = false;
    try {
      return await Promise.race([
        initPromise,
        new Promise<never>((_, reject) => {
          hardTimer = setTimeout(() => {
            timedOut = true;
            reject(
              RequestError.internalError(
                {
                  "_claude/timeout": "init",
                  phase: isResume ? "resume" : "init",
                  elapsedMs: INIT_HARD_MS,
                },
                isResume
                  ? "Resume timed out — the session may be too large to rehydrate. " +
                      "Try Hard recover or start a fresh session."
                  : "Initialization timed out. Try Hard recover or start a fresh session.",
              ),
            );
          }, INIT_HARD_MS);
        }),
      ]);
    } catch (error) {
      if (timedOut) {
        try {
          opts.abort();
        } catch {
          /* best-effort */
        }
        try {
          opts.close();
        } catch {
          /* best-effort */
        }
      }
      throw error;
    } finally {
      clearInterval(heartbeat);
      if (hardTimer) clearTimeout(hardTimer);
    }
  }

  private async createSession(
    params: NewSessionRequest,
    creationOpts: { resume?: string; forkSession?: boolean } = {},
  ): Promise<NewSessionResponse> {
    // We want to create a new session id unless it is resume,
    // but not resume + forkSession.
    let sessionId;
    if (creationOpts.forkSession) {
      sessionId = randomUUID();
    } else if (creationOpts.resume) {
      sessionId = creationOpts.resume;
    } else {
      sessionId = randomUUID();
    }

    const input = new Pushable<SDKUserMessage>();

    const settingsManager = new SettingsManager(params.cwd, {
      logger: this.logger,
    });
    await settingsManager.initialize();

    const mcpServers: Record<string, McpServerConfig> = {};
    if (Array.isArray(params.mcpServers)) {
      for (const server of params.mcpServers) {
        if ("type" in server && (server.type === "http" || server.type === "sse")) {
          // HTTP or SSE type MCP server
          mcpServers[server.name] = {
            type: server.type,
            url: server.url,
            headers: server.headers
              ? Object.fromEntries(server.headers.map((e) => [e.name, e.value]))
              : undefined,
          };
        } else if (!("type" in server)) {
          // Stdio type MCP server (with or without explicit type field)
          mcpServers[server.name] = {
            type: "stdio",
            command: server.command,
            args: server.args,
            env: server.env
              ? Object.fromEntries(server.env.map((e) => [e.name, e.value]))
              : undefined,
          };
        }
      }
    }

    let systemPrompt: Options["systemPrompt"] = { type: "preset", preset: "claude_code" };
    if (params._meta?.systemPrompt) {
      const customPrompt = params._meta.systemPrompt;
      if (typeof customPrompt === "string") {
        systemPrompt = customPrompt;
      } else if (
        typeof customPrompt === "object" &&
        customPrompt !== null &&
        !Array.isArray(customPrompt)
      ) {
        // Forward all preset options (append, excludeDynamicSections, and
        // anything the SDK adds later) while locking type/preset.
        systemPrompt = {
          ...(customPrompt as object),
          type: "preset",
          preset: "claude_code",
        } as Options["systemPrompt"];
      }
    }

    const permissionMode = resolvePermissionMode(
      settingsManager.getSettings().permissions?.defaultMode,
      this.logger,
    );

    // Extract options from _meta if provided
    const sessionMeta = params._meta as NewSessionMeta | undefined;
    const userProvidedOptions = sessionMeta?.claudeCode?.options;

    // Configure thinking tokens from environment variable
    const maxThinkingTokens = process.env.MAX_THINKING_TOKENS
      ? parseInt(process.env.MAX_THINKING_TOKENS, 10)
      : undefined;

    // Parse model configuration from environment (e.g. Bedrock model overrides)
    const modelConfig = parseModelConfig(process.env.CLAUDE_MODEL_CONFIG);

    // Disable this for now, not a great way to expose this over ACP at the moment (in progress work so we can revisit)
    const disallowedTools = ["AskUserQuestion"];

    // Resolve which built-in tools to expose.
    // Explicit tools array from _meta.claudeCode.options takes precedence.
    // disableBuiltInTools is a legacy shorthand for tools: [] — kept for
    // backward compatibility but callers should prefer the tools array.
    const tools: Options["tools"] =
      userProvidedOptions?.tools ??
      (params._meta?.disableBuiltInTools === true ? [] : { type: "preset", preset: "claude_code" });

    const abortController = userProvidedOptions?.abortController || new AbortController();

    // Per-session task state. Created here (rather than in the session record
    // below) so the TaskCreated/TaskCompleted hook callbacks can close over
    // the same Map that the streaming message handler will read from.
    const taskState: TaskState = new Map();

    const options: Options = {
      systemPrompt,
      settingSources: ["user", "project", "local"],
      ...(maxThinkingTokens !== undefined && { maxThinkingTokens }),
      ...userProvidedOptions,
      // CLAUDE_MODEL_CONFIG env var is a fallback for model
      // configuration (e.g. Bedrock model ID overrides). When the caller
      // provides settings via _meta, we intentionally ignore the env var —
      // the caller is assumed to have full control over model configuration.
      ...(!userProvidedOptions?.settings &&
        modelConfig && {
          settings: {
            ...(modelConfig.modelOverrides && { modelOverrides: modelConfig.modelOverrides }),
            ...(modelConfig.availableModels && { availableModels: modelConfig.availableModels }),
          },
        }),
      env: {
        ...process.env,
        ...userProvidedOptions?.env,
        ...createEnvForGateway(this.gatewayAuthRequest),
        // Opt-in to session state events like when the agent is idle
        CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS: "1",
      },
      // Override certain fields that must be controlled by ACP
      cwd: params.cwd,
      includePartialMessages: true,
      mcpServers: { ...(userProvidedOptions?.mcpServers || {}), ...mcpServers },
      // If we want bypassPermissions to be an option, we have to allow it here.
      // But it doesn't work in root mode, so we only activate it if it will work.
      allowDangerouslySkipPermissions: ALLOW_BYPASS,
      permissionMode,
      canUseTool: this.canUseTool(sessionId),
      pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE ?? (await claudeCliPath()),
      extraArgs: {
        ...userProvidedOptions?.extraArgs,
        "replay-user-messages": "",
      },
      disallowedTools: [...(userProvidedOptions?.disallowedTools || []), ...disallowedTools],
      tools,
      hooks: {
        ...userProvidedOptions?.hooks,
        PostToolUse: [
          ...(userProvidedOptions?.hooks?.PostToolUse || []),
          {
            hooks: [
              createPostToolUseHook(this.logger, {
                onEnterPlanMode: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "current_mode_update",
                      currentModeId: "plan",
                    },
                  });
                  await this.updateConfigOption(sessionId, "mode", "plan");
                },
              }),
            ],
          },
        ],
        TaskCreated: [
          ...(userProvidedOptions?.hooks?.TaskCreated || []),
          {
            hooks: [
              createTaskHook({
                taskState,
                onChange: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "plan",
                      entries: taskStateToPlanEntries(taskState),
                    },
                  });
                },
              }),
            ],
          },
        ],
        TaskCompleted: [
          ...(userProvidedOptions?.hooks?.TaskCompleted || []),
          {
            hooks: [
              createTaskHook({
                taskState,
                onChange: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "plan",
                      entries: taskStateToPlanEntries(taskState),
                    },
                  });
                },
              }),
            ],
          },
        ],
      },
      ...creationOpts,
      abortController,
    };

    // Prefer the official ACP `additionalDirectories` field. Fall back to the
    // legacy `_meta.additionalRoots` extension for clients that haven't been
    // updated yet. Either source is merged with directories supplied via
    // `_meta.claudeCode.options.additionalDirectories` (SDK pass-through).
    const acpAdditionalDirectories =
      params.additionalDirectories ?? sessionMeta?.additionalRoots ?? [];
    options.additionalDirectories = [
      ...(userProvidedOptions?.additionalDirectories ?? []),
      ...acpAdditionalDirectories,
    ];

    if (creationOpts?.resume === undefined || creationOpts?.forkSession) {
      // Set our own session id if not resuming an existing session.
      options.sessionId = sessionId;
    }

    // Handle abort controller from meta options
    if (abortController?.signal.aborted) {
      throw new Error("Cancelled");
    }

    const q = query({
      prompt: input,
      options,
    });

    // INIT/RESUME bounded timeout + heartbeat (CONCEPTION §3, D2). Never block
    // forever on a heavy resume or a blocked child: emit "still resuming — N s"
    // heartbeats and HARD-FAIL at INIT_HARD_MS with a structured terminal error.
    let initializationResult;
    try {
      initializationResult = await this.awaitInitializationBounded(q.initializationResult(), {
        sessionId,
        isResume: Boolean(creationOpts.resume),
        abort: () => abortController.abort(),
        close: () => q.close(),
      });
    } catch (error) {
      if (
        creationOpts.resume &&
        error instanceof Error &&
        (error.message === "Query closed before response received" ||
          error.message.includes("No conversation found with session ID"))
      ) {
        throw RequestError.resourceNotFound(sessionId);
      }
      throw error;
    }

    if (
      shouldHideClaudeAuth() &&
      initializationResult.account.subscriptionType &&
      !this.gatewayAuthRequest
    ) {
      throw RequestError.authRequired(
        undefined,
        "This integration does not support using claude.ai subscriptions.",
      );
    }

    // Apply user's `availableModels` allowlist from settings.json before any
    // downstream model handling. The SDK only enforces this allowlist in its
    // own UI, not in `initializationResult.models`, so we filter here to keep
    // configOptions, the current-model resolver, and the stored modelInfos
    // consistent with what the user configured.
    const settingsAvailableModels = settingsManager.getSettings().availableModels;
    const allowedModels = Array.isArray(settingsAvailableModels)
      ? applyAvailableModelsAllowlist(initializationResult.models, settingsAvailableModels)
      : initializationResult.models;

    // `getAvailableModels` may re-label the user's pinned model so its
    // advertised id stays stable across new/resume (see its docs). Use the
    // returned `resolvedModelInfos` — not the pre-call `allowedModels` — for
    // every downstream model lookup so the stored `modelInfos`, advertised
    // ids, and `currentModelId` remain mutually consistent.
    const availableModels = await getAvailableModels(
      q,
      allowedModels,
      initializationResult.models,
      settingsManager,
      this.logger,
    );

    // Advertise the Claude "fable" model. The bundled Claude Code binary knows
    // and can RUN fable for our Claude Max accounts, but a server-side launch
    // gate hides it from the advertised model menu — so `initializationResult.
    // models` never contains it, even on the latest SDK. Inject it here, after
    // the SDK/allowlist list is resolved, so a generic ACP client (acpx) can
    // pin `--model fable` and pass the client-side support gate. The injection
    // runs on EVERY init — new AND resume — via the shared `createSession`
    // path, which is exactly what keeps the advertised id the literal `fable`
    // on resume (the SDK would otherwise surface the resolved concrete id
    // `claude-fable-5`, drifting the advertised value and breaking the acpx
    // replay gate — the failure mode `opus[1m]` hit). See `injectFableModel`.
    const __fableInjected = injectFableModel(availableModels.state, availableModels.modelInfos);
    const { state: models, modelInfos: resolvedModelInfos } = injectOpusModel(
      __fableInjected.state,
      __fableInjected.modelInfos,
    );

    // Gate `auto` (and future model-specific modes) on the resolved model's
    // `ModelInfo`. See `buildAvailableModes` for the canonical SDK signal.
    const currentModelInfo = resolvedModelInfos.find((m) => m.value === models.currentModelId);
    const availableModes = buildAvailableModes(currentModelInfo);

    // Clamp `permissionMode` if the resolved session does not offer it. The
    // common case is `permissions.defaultMode: "auto"` resolving to a model
    // that does not support auto mode (e.g. Haiku); without this clamp the
    // SDK would later throw `"auto mode unavailable for this model"` from
    // `setPermissionMode`. Keep `permissionMode` as the resolved user intent
    // (matches what was passed into `options.permissionMode` above) and use
    // `effectiveMode` for the post-clamp value the session actually runs in.
    let effectiveMode: PermissionMode = permissionMode;
    if (!availableModes.some((m) => m.id === effectiveMode)) {
      if (effectiveMode === "auto") {
        this.logger.error(
          `permissions.defaultMode "auto" is not available for model ` +
            `"${models.currentModelId}"; falling back to "default".`,
        );
      } else {
        this.logger.error(
          `permissions.defaultMode "${effectiveMode}" is not available in ` +
            `this session; falling back to "default".`,
        );
      }
      effectiveMode = "default";
      // Sync the SDK so it doesn't keep "auto" cached internally. Wrapped in
      // try/catch since failing here would abort session creation entirely.
      try {
        await q.setPermissionMode("default");
      } catch (err) {
        this.logger.error("Failed to sync clamped permissionMode to SDK:", err);
      }
    }

    const modes = {
      currentModeId: effectiveMode,
      availableModes,
    };

    const configOptions = buildConfigOptions(
      modes,
      models,
      resolvedModelInfos,
      settingsManager.getSettings().effortLevel,
    );

    // Apply the initial effort level to the SDK so it matches the UI default
    const initialEffort = configOptions.find((o) => o.id === "effort");
    if (
      initialEffort &&
      typeof initialEffort.currentValue === "string" &&
      initialEffort.currentValue !== "default"
    ) {
      await q.applyFlagSettings({
        effortLevel: initialEffort.currentValue as Settings["effortLevel"],
      });
    }

    this.sessions[sessionId] = {
      query: q,
      input: input,
      cancelled: false,
      cwd: params.cwd,
      sessionFingerprint: computeSessionFingerprint(params),
      settingsManager,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      modes,
      models,
      modelInfos: resolvedModelInfos,
      configOptions,
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController,
      emitRawSDKMessages: sessionMeta?.claudeCode?.emitRawSDKMessages ?? false,
      activePromptResolve: null,
      pendingSdkMessages: [],
      backgroundLoopError: null,
      contextWindowSize:
        inferContextWindowFromModel(models.currentModelId, currentModelInfo?.description) ??
        DEFAULT_CONTEXT_WINDOW,
      taskState,
    };

    this.startBackgroundReaderLoop(sessionId);

    return {
      sessionId,
      models,
      modes,
      configOptions,
    };
  }

  /**
   * Persistent background reader loop: the sole consumer of session.query.
   * Routes each message to the active prompt handler (if one is running) or
   * processes it as an idle inter-turn update (subagent activity after end_turn).
   */
  private startBackgroundReaderLoop(sessionId: string): void {
    const loop = async () => {
      const session = this.sessions[sessionId];
      if (!session) return;

      try {
        while (true) {
          const { value, done } = await session.query.next();

          if (done || !value) {
            // Session ended — wake any waiting prompt so it can return/throw.
            if (session.activePromptResolve) {
              const resolve = session.activePromptResolve;
              session.activePromptResolve = null;
              resolve(null);
            } else if (session.promptRunning) {
              // Loop owns the stream but is mid-await: queue a null sentinel so
              // its next park observes stream end (A1).
              session.pendingSdkMessages.push(null);
            }
            break;
          }

          if (session.activePromptResolve) {
            // Deliver to the active prompt's nextMessage() call.
            const resolve = session.activePromptResolve;
            session.activePromptResolve = null;
            resolve(value);
          } else if (session.promptRunning) {
            // A1: a prompt owns the stream but its loop is mid-await (no resolver
            // parked). Buffer instead of routing to handleIdleMessage — which
            // silently discarded turn-control messages and withheld the response
            // (RCA §1.2). nextMessage() drains this at the next park. Buffer ALL
            // message types: classification is where bugs live, and a buffered
            // content chunk is simply processed milliseconds later, exactly as a
            // parked-path delivery would be.
            if (!session.backgroundLoopError) {
              if (session.pendingSdkMessages.length >= MAX_PENDING_SDK_MESSAGES) {
                // Overflow ⇒ the loop is wedged and the buffer is masking it.
                // Loud failure: stage the error and queue a null sentinel so the
                // loop terminates the turn through the existing error path.
                session.backgroundLoopError = new Error(
                  `Session ${sessionId}: buffered SDK message count exceeded ` +
                    `${MAX_PENDING_SDK_MESSAGES}; terminating the turn instead of ` +
                    `dropping messages.`,
                );
                session.pendingSdkMessages.push(null);
              } else {
                session.pendingSdkMessages.push(value);
              }
            }
          } else {
            // Genuine inter-turn (idle) message: emit raw if configured, then forward.
            if (
              session.emitRawSDKMessages &&
              shouldEmitRawMessage(session.emitRawSDKMessages, value)
            ) {
              await this.client.extNotification("_claude/sdkMessage", {
                sessionId,
                message: value as Record<string, unknown>,
              });
            }
            await this.handleIdleMessage(value, sessionId);
          }
        }
      } catch (error) {
        // Claude process died — store error so the prompt can re-throw it.
        session.backgroundLoopError = error instanceof Error ? error : new Error(String(error));
        if (session.activePromptResolve) {
          const resolve = session.activePromptResolve;
          session.activePromptResolve = null;
          resolve(null);
        } else if (session.promptRunning) {
          // Loop is mid-await: queue a null sentinel so its next park returns
          // null and re-throws backgroundLoopError (A1).
          session.pendingSdkMessages.push(null);
        }
      }
    };

    loop(); // fire and forget; errors are handled internally
  }

  /**
   * Handle a message that arrives while no prompt is active (idle inter-turn).
   * Forwards stream_event, assistant, and result messages as session/update
   * notifications so ACPX can observe teammate-triggered main-agent activity.
   */
  private async handleIdleMessage(message: SDKMessage, sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;

    switch (message.type) {
      case "stream_event": {
        for (const notification of streamEventToAcpNotifications(
          message,
          sessionId,
          this.toolUseCache,
          this.client,
          this.logger,
          {
            clientCapabilities: this.clientCapabilities,
            cwd: session.cwd,
            taskState: session.taskState,
            subagentCache: this.subagentCache,
          },
        )) {
          await this.client.sessionUpdate(notification);
        }
        break;
      }
      case "assistant": {
        const content = message.message.content.filter(
          (item: any) => !["text", "thinking"].includes(item.type),
        );
        for (const notification of toAcpNotifications(
          content,
          message.message.role,
          sessionId,
          this.toolUseCache,
          this.client,
          this.logger,
          {
            clientCapabilities: this.clientCapabilities,
            parentToolUseId: message.parent_tool_use_id,
            cwd: session.cwd,
            taskState: session.taskState,
            subagentCache: this.subagentCache,
          },
        )) {
          await this.client.sessionUpdate(notification);
        }
        break;
      }
      case "result": {
        // Accumulate usage from idle teammate turns.
        session.accumulatedUsage.inputTokens += message.usage.input_tokens;
        session.accumulatedUsage.outputTokens += message.usage.output_tokens;
        session.accumulatedUsage.cachedReadTokens += message.usage.cache_read_input_tokens;
        session.accumulatedUsage.cachedWriteTokens += message.usage.cache_creation_input_tokens;
        break;
      }
      case "system": {
        if (message.subtype === "task_started" && message.tool_use_id) {
          await this.onTeammateSpawned(
            message.tool_use_id,
            message.task_id,
            message.description,
            sessionId,
          );
        } else if (message.subtype === "task_progress") {
          await this.onTaskProgress(
            message.tool_use_id,
            message.task_id,
            message.last_tool_name,
            sessionId,
          );
        } else if (message.subtype === "task_notification") {
          await this.onTaskNotification(
            message.tool_use_id,
            message.task_id,
            message.status,
            sessionId,
          );
        }
        break;
      }
      default:
        break;
    }
  }

  /**
   * Called when a task_progress system message arrives.
   * Emits a tool_call_update with status 'task_progress' so ACPX can observe subagent activity.
   */
  private async onTaskProgress(
    toolUseId: string | undefined,
    taskId: string,
    lastToolName: string | undefined,
    sessionId: string,
  ): Promise<void> {
    const subagent = toolUseId ? this.subagentCache.get(toolUseId) : undefined;
    await this.client.sessionUpdate({
      sessionId,
      update: {
        _meta: {
          claudeCode: {
            toolName: "Agent",
            status: "task_progress",
            subagentId: subagent?.agentId ?? taskId,
            subagentName: subagent?.name,
            subagentColor: subagent?.color,
            taskLastToolName: lastToolName,
          },
        } satisfies ToolUpdateMeta,
        toolCallId: toolUseId ?? taskId,
        sessionUpdate: "tool_call_update",
      },
    });
  }

  /**
   * Called when a task_notification system message arrives (task completed/failed/stopped).
   * Emits a tool_call_update with status 'task_completed', 'task_failed', or 'task_stopped'.
   */
  private async onTaskNotification(
    toolUseId: string | undefined,
    taskId: string,
    status: "completed" | "failed" | "stopped",
    sessionId: string,
  ): Promise<void> {
    const subagent = toolUseId ? this.subagentCache.get(toolUseId) : undefined;
    await this.client.sessionUpdate({
      sessionId,
      update: {
        _meta: {
          claudeCode: {
            toolName: "Agent",
            status: `task_${status}`,
            subagentId: subagent?.agentId ?? taskId,
            subagentName: subagent?.name,
            subagentColor: subagent?.color,
          },
        } satisfies ToolUpdateMeta,
        toolCallId: toolUseId ?? taskId,
        sessionUpdate: "tool_call_update",
      },
    });
  }

  /**
   * Called when a task_started system message arrives (in active prompt or idle).
   * Populates subagentCache and emits a tool_call_update with status 'teammate_spawned'.
   */
  private async onTeammateSpawned(
    toolUseId: string,
    taskId: string,
    description: string,
    sessionId: string,
  ): Promise<void> {
    const toolUse = this.toolUseCache[toolUseId];
    if (!toolUse || (toolUse.name !== "Agent" && toolUse.name !== "Task")) return;

    const input = toolUse.input as { name?: string; description?: string; color?: string };
    const agentName = (input.name || input.description || description).trim();
    this.subagentCache.set(toolUseId, {
      agentId: taskId,
      name: agentName,
      color: input.color,
    });

    await this.client.sessionUpdate({
      sessionId,
      update: {
        _meta: {
          claudeCode: {
            toolName: toolUse.name,
            status: "teammate_spawned",
            subagentId: taskId,
            subagentName: agentName,
            subagentColor: input.color,
          },
        } satisfies ToolUpdateMeta,
        toolCallId: toolUseId,
        sessionUpdate: "tool_call_update",
      },
    });
  }
}

function shouldEmitRawMessage(
  config: boolean | SDKMessageFilter[],
  message: { type: string; subtype?: string; origin?: SDKMessageOrigin },
): boolean {
  if (config === true) return true;
  if (config === false) return false;
  return config.some(
    (f) =>
      f.type === message.type &&
      (f.subtype === undefined || f.subtype === message.subtype) &&
      (f.origin === undefined || f.origin === message.origin?.kind),
  );
}

function sessionUsage(session: Session) {
  return {
    inputTokens: session.accumulatedUsage.inputTokens,
    outputTokens: session.accumulatedUsage.outputTokens,
    cachedReadTokens: session.accumulatedUsage.cachedReadTokens,
    cachedWriteTokens: session.accumulatedUsage.cachedWriteTokens,
    totalTokens:
      session.accumulatedUsage.inputTokens +
      session.accumulatedUsage.outputTokens +
      session.accumulatedUsage.cachedReadTokens +
      session.accumulatedUsage.cachedWriteTokens,
  };
}

/** Sum all four fields as a proxy for post-turn context occupancy: the current
 *  turn's output becomes next turn's input. Per the Anthropic API, input_tokens
 *  excludes cache tokens — cache_read and cache_creation are reported
 *  separately — so summing all four is not double-counting. */
function totalTokens(usage: UsageSnapshot): number {
  return (
    usage.input_tokens +
    usage.output_tokens +
    usage.cache_read_input_tokens +
    usage.cache_creation_input_tokens
  );
}

/**
 * Build the `data` payload attached to a `RequestError.internalError` when we
 * have a categorical error from the Claude SDK. Returns `undefined` when no
 * categorical error is available, matching the previous behavior of passing
 * `undefined` to `RequestError.internalError`.
 *
 * The `errorKind` field is a convention for ACP clients to dispatch on
 * without having to pattern-match the human-readable message text. Clients
 * that don't understand it fall back to the existing message-based rendering.
 */
function errorKindData(
  errorKind: SDKAssistantMessageError | undefined,
): { errorKind: SDKAssistantMessageError } | undefined {
  return errorKind ? { errorKind } : undefined;
}

/** Project a nullable API usage object into our non-null snapshot shape.
 *  Both SDK message_start and assistant message `usage` have `number | null`
 *  cache fields; we coerce absent values to 0 so `totalTokens` never hits
 *  NaN. `input_tokens`/`output_tokens` are typed `number` by the SDK but
 *  synthetic or third-party-backend stream events have been observed emitting
 *  them as null/undefined — coerce those too so a malformed upstream event
 *  can't leak NaN into the wire `used` field. Delta events have different
 *  semantics (cumulative + prev fallback) and are handled inline. */
function snapshotFromUsage(usage: {
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}): UsageSnapshot {
  return {
    input_tokens: usage.input_tokens ?? 0,
    output_tokens: usage.output_tokens ?? 0,
    cache_read_input_tokens: usage.cache_read_input_tokens ?? 0,
    cache_creation_input_tokens: usage.cache_creation_input_tokens ?? 0,
  };
}

function createEnvForGateway(request?: GatewayAuthRequest) {
  if (!request?._meta) {
    return {};
  }
  const customHeaders = Object.entries(request._meta.gateway.headers)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");

  if (request.methodId === "gateway-bedrock") {
    return {
      CLAUDE_CODE_USE_BEDROCK: "1",
      AWS_BEARER_TOKEN_BEDROCK: " ", // Must be non-empty to bypass pass configuration check
      ANTHROPIC_BEDROCK_BASE_URL: request._meta.gateway.baseUrl,
      ANTHROPIC_CUSTOM_HEADERS: customHeaders,
    };
  }
  return {
    ANTHROPIC_BASE_URL: request._meta.gateway.baseUrl,
    ANTHROPIC_CUSTOM_HEADERS: customHeaders,
    ANTHROPIC_AUTH_TOKEN: " ", // Must be specified to bypass claude login requirement
  };
}

/**
 * Build the list of permission modes the agent will advertise for the given
 * model. `auto` is gated by `ModelInfo.supportsAutoMode === true`, which is
 * the SDK's model-level availability signal. `undefined`/`false` both exclude
 * `auto`. `bypassPermissions` is still gated by `ALLOW_BYPASS`.
 */
function buildAvailableModes(modelInfo: ModelInfo | undefined): SessionModeState["availableModes"] {
  const modes: SessionModeState["availableModes"] = [];

  // Only advertise "auto" when the SDK reports the model supports it.
  if (modelInfo?.supportsAutoMode === true) {
    modes.push({
      id: "auto",
      name: "Auto",
      description: "Use a model classifier to approve/deny permission prompts",
    });
  }

  modes.push(
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
    {
      id: "plan",
      name: "Plan Mode",
      description: "Planning mode, no actual tool execution",
    },
    {
      id: "dontAsk",
      name: "Don't Ask",
      description: "Don't prompt for permissions, deny if not pre-approved",
    },
  );

  if (ALLOW_BYPASS) {
    modes.push({
      id: "bypassPermissions",
      name: "Bypass Permissions",
      description: "Bypass all permission checks",
    });
  }

  return modes;
}

// Translate a UI effort value into the flag-layer payload. The SDK
// shallow-merges `applyFlagSettings`, drops `undefined` during JSON transport,
// and only clears a key when an explicit `null` is sent — see
// `applyFlagSettings` in @anthropic-ai/claude-agent-sdk. Mapping both the
// `"default"` sentinel and `undefined` (effort option absent for the model) to
// `null` ensures any previously-applied flag is actually cleared.
function toSdkEffortLevel(value: string | undefined): Settings["effortLevel"] | null {
  return value === undefined || value === "default" ? null : (value as Settings["effortLevel"]);
}

function buildConfigOptions(
  modes: SessionModeState,
  models: SessionModelState,
  modelInfos: ModelInfo[],
  currentEffortLevel?: string,
): SessionConfigOption[] {
  const options: SessionConfigOption[] = [
    {
      id: "mode",
      name: "Mode",
      description: "Session permission mode",
      category: "mode",
      type: "select",
      currentValue: modes.currentModeId,
      options: modes.availableModes.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      })),
    },
    {
      id: "model",
      name: "Model",
      description: "AI model to use",
      category: "model",
      type: "select",
      currentValue: models.currentModelId,
      options: models.availableModels.map((m) => ({
        value: m.modelId,
        name: m.name,
        description: m.description ?? undefined,
      })),
    },
  ];

  // Add effort level option based on the currently selected model. Tolerate a
  // `[1m]` hint on currentModelId that the base entry lacks (a mid-session
  // switch stores "sonnet[1m]" while modelInfos may hold only "sonnet").
  const currentModelInfo = findModelInfoById(modelInfos, models.currentModelId);
  const supportedLevels = currentModelInfo?.supportsEffort
    ? (currentModelInfo.supportedEffortLevels ?? [])
    : [];

  if (supportedLevels.length > 0) {
    const effortOptions = [
      { value: "default", name: "Default" },
      ...supportedLevels.map((level) => ({
        value: level,
        name: level
          .split(/[_-]/)
          .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
          .join(" "),
      })),
    ];

    const includes = (l: string) => l === "default" || (supportedLevels as string[]).includes(l);
    const validEffort =
      currentEffortLevel && includes(currentEffortLevel) ? currentEffortLevel : "default";

    options.push({
      id: "effort",
      name: "Effort",
      description: "Available effort levels for this model",
      category: "thought_level",
      type: "select",
      currentValue: validEffort,
      options: effortOptions,
    });
  }

  return options;
}

// Claude Code CLI persists display strings like "opus[1m]" in settings,
// but the SDK model list uses IDs like "claude-opus-4-6-1m".
const MODEL_CONTEXT_HINT_PATTERN = /\[(\d+m)\]$/i;

// Captures a model family version such as `4-6` or `4.7` so we can keep
// `claude-opus-4-6` from being copied onto the SDK's `opus` alias when that
// alias currently resolves to a different family version (e.g. Opus 4.7).
const MODEL_FAMILY_VERSION_PATTERN = /\b(\d+)[-.](\d+)\b/;

function extractModelFamilyVersion(s: string): string | null {
  const match = s.match(MODEL_FAMILY_VERSION_PATTERN);
  return match ? `${match[1]}.${match[2]}` : null;
}

function modelVersionsCompatible(preference: string, candidate: ModelInfo): boolean {
  const preferred = extractModelFamilyVersion(preference);
  if (!preferred) return true;
  const candidateVersion =
    extractModelFamilyVersion(candidate.value) ??
    extractModelFamilyVersion(candidate.displayName) ??
    extractModelFamilyVersion(candidate.description);
  if (!candidateVersion) return true;
  return preferred === candidateVersion;
}

function tokenizeModelPreference(model: string): { tokens: string[]; contextHint?: string } {
  const lower = model.trim().toLowerCase();
  const contextHint = lower.match(MODEL_CONTEXT_HINT_PATTERN)?.[1]?.toLowerCase();

  const normalized = lower.replace(MODEL_CONTEXT_HINT_PATTERN, " $1 ");
  const rawTokens = normalized.split(/[^a-z0-9]+/).filter(Boolean);
  const tokens = rawTokens
    .map((token) => {
      if (token === "opusplan") return "opus";
      if (token === "best" || token === "default") return "";
      return token;
    })
    .filter((token) => token && token !== "claude")
    .filter((token) => /[a-z]/.test(token) || token.endsWith("m"));

  return { tokens, contextHint };
}

function scoreModelMatch(model: ModelInfo, tokens: string[], contextHint?: string): number {
  const haystack = `${model.value} ${model.displayName}`.toLowerCase();
  let score = 0;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      score += token === contextHint ? 3 : 1;
    }
  }
  return score;
}

function resolveModelPreference(models: ModelInfo[], preference: string): ModelInfo | null {
  const trimmed = preference.trim();
  if (!trimmed) return null;

  const lower = trimmed.toLowerCase();

  // Exact match on value or display name
  const directMatch = models.find(
    (model) =>
      model.value === trimmed ||
      model.value.toLowerCase() === lower ||
      model.displayName.toLowerCase() === lower,
  );
  if (directMatch) return directMatch;

  // Substring match
  const includesMatch = models.find((model) => {
    if (!modelVersionsCompatible(trimmed, model)) return false;
    const value = model.value.toLowerCase();
    const display = model.displayName.toLowerCase();
    return value.includes(lower) || display.includes(lower) || lower.includes(value);
  });
  if (includesMatch) return includesMatch;

  // Tokenized matching for aliases like "opus[1m]"
  const { tokens, contextHint } = tokenizeModelPreference(trimmed);
  if (tokens.length === 0) return null;

  let bestMatch: ModelInfo | null = null;
  let bestScore = 0;
  for (const model of models) {
    if (!modelVersionsCompatible(trimmed, model)) continue;
    const score = scoreModelMatch(model, tokens, contextHint);
    if (0 < score && (!bestMatch || bestScore < score)) {
      bestMatch = model;
      bestScore = score;
    }
  }

  return bestMatch;
}

/**
 * Preserve an explicit `[1m]` (or other `[<n>m]`) context hint on the model id
 * that is actually handed to the Claude Code binary via `query.setModel(...)`.
 *
 * `resolveModelPreference` matches the base `ModelInfo` for alias lookups and
 * returns its bare `.value` ("sonnet"), dropping the `[1m]` suffix. But the
 * binary is the authority on `[1m]`: it strips `(\[1m\])+$` itself and enables
 * the long-context beta (`anthropic-beta: context-1m-2025-08-07`) for the base
 * model. If we forward the stripped base, the binary runs the model at its 200k
 * default even though the picker/label promised 1M — the exact defect this fixes.
 *
 * So: when the *requested* id carried a context hint, re-attach it to the
 * resolved base (unless the base already carries one). A requested id WITHOUT a
 * hint is returned untouched — we never fabricate long-context for a plain pick.
 */
export function effectiveRunModelId(requested: string, resolvedBase: string): string {
  const hint = requested.match(MODEL_CONTEXT_HINT_PATTERN)?.[1];
  if (!hint) return resolvedBase;
  // The resolved base may ALREADY denote this context tier — either bracketed
  // ("sonnet[1m]") or as a concrete id ("claude-opus-4-6-1m"). Both forms carry
  // the hint as a `\b<hint>\b` token; don't double-append (a doubled suffix
  // would defeat the binary's `(\[1m\])+$` strip and inferContextWindowFromModel
  // already matches either form). `hint` is `\d+m`, so it is regex-safe.
  if (new RegExp(`\\b${hint}\\b`, "i").test(resolvedBase)) return resolvedBase;
  return `${resolvedBase}[${hint}]`;
}

/**
 * Look up the `ModelInfo` for a stored/current model id, tolerating a `[1m]`
 * context hint the advertised base entry does not carry. A mid-session switch
 * stores the `[1m]` run value as `currentModelId` (so the reported window is
 * 1M), but `session.modelInfos` may hold only the base entry (e.g. a session
 * started on "default" then switched to "sonnet[1m]"). Fall back to the alias
 * resolver so effort/mode gating keeps keying on the base model's capabilities.
 */
function findModelInfoById(modelInfos: ModelInfo[], id: string): ModelInfo | undefined {
  return (
    modelInfos.find((m) => m.value === id) ?? resolveModelPreference(modelInfos, id) ?? undefined
  );
}

function resolveSettingsModel(
  models: ModelInfo[],
  settingsModel: unknown,
  logger: Logger,
): ModelInfo | null {
  if (settingsModel === undefined) {
    return null;
  }
  if (typeof settingsModel !== "string") {
    const typeLabel = settingsModel === null ? "null" : typeof settingsModel;
    logger.error(`Ignoring model from settings: expected a string, got ${typeLabel}.`);
    return null;
  }
  return resolveModelPreference(models, settingsModel);
}

/** Advertised id / alias for the Claude "fable" model. The bundled Claude Code
 *  binary resolves this alias to the concrete `claude-fable-5` (CC ≥2.1.172),
 *  so clients pin the stable `fable` while the SDK runs `claude-fable-5`. */
const FABLE_MODEL_ID = "fable";
const OPUS_MODEL_ID = "opus";

/**
 * Additively advertise the Claude "fable" model on top of the resolved
 * SDK/allowlist model set.
 *
 * Why this is needed: the bundled Claude Code binary is entitled to RUN fable
 * for our Claude Max accounts (forcing `--model claude-fable-5`/`fable` runs a
 * turn), but a server-side launch gate omits fable from the advertised model
 * menu — `initializationResult.models` / `supportedModels()` never list it,
 * even on the latest SDK. The acpx client-side support gate
 * (`assertRequestedModelSupported`) rejects any `--model` value the agent did
 * not advertise, so without this injection `--model fable` cannot be selected.
 *
 * Properties (all required by the design):
 * - **Additive** — the full base set (Default + sonnet/sonnet[1m]/haiku/
 *   opus[1m], or whatever the SDK/allowlist resolved) is preserved untouched.
 *   No restrict-list drift: if Anthropic adds or renames a base model it still
 *   flows through.
 * - **Idempotent** — a no-op if `fable` is already advertised. The SDK never
 *   surfaces it today; a future SDK that does would not get a duplicate, and
 *   we never collide with an existing entry (`resolveModelPreference(base,
 *   "fable")` returns `null` against the base set — there is no fuzzy match
 *   against sonnet/haiku/opus).
 * - **Resume-stable** — because the caller runs this on EVERY init (new AND
 *   resume, via the shared `createSession` path), the advertised id stays the
 *   literal `fable` on reconnect. We never rely on the SDK to surface fable, so
 *   unlike the removed `opus[1m]` the advertised value does not drift to the
 *   resolved concrete id (`claude-fable-5`) on resume — which is what kept the
 *   acpx replay gate accepting the persisted `fable` alias.
 */
function injectFableModel(
  state: SessionModelState,
  modelInfos: ModelInfo[],
): { state: SessionModelState; modelInfos: ModelInfo[] } {
  if (modelInfos.some((m) => m.value === FABLE_MODEL_ID)) {
    return { state, modelInfos };
  }
  const fableInfo: ModelInfo = {
    value: FABLE_MODEL_ID,
    displayName: "Fable",
    description: "Fable",
  };
  return {
    state: {
      ...state,
      availableModels: [
        ...state.availableModels,
        {
          modelId: fableInfo.value,
          name: fableInfo.displayName,
          description: fableInfo.description,
        },
      ],
    },
    modelInfos: [...modelInfos, fableInfo],
  };
}

function injectOpusModel(
  state: SessionModelState,
  modelInfos: ModelInfo[],
): { state: SessionModelState; modelInfos: ModelInfo[] } {
  if (modelInfos.some((m) => m.value === OPUS_MODEL_ID)) {
    return { state, modelInfos };
  }
  const opusInfo: ModelInfo = {
    value: OPUS_MODEL_ID,
    displayName: "Opus",
    description: "Opus 4.8",
  };
  return {
    state: {
      ...state,
      availableModels: [
        ...state.availableModels,
        {
          modelId: opusInfo.value,
          name: opusInfo.displayName,
          description: opusInfo.description,
        },
      ],
    },
    modelInfos: [...modelInfos, opusInfo],
  };
}

/**
 * Restrict the SDK's model list to the user's `availableModels` allowlist
 * (already merged-and-deduped across settings sources by `SettingsManager`).
 * The user's exact entries become the model IDs surfaced via configOptions
 * and passed to `setModel`, which prevents Claude Code from silently
 * substituting a date-pinned variant (e.g. `haiku` →
 * `claude-haiku-4-5-20251001`) that the user may not have access to.
 *
 * Display info and capability flags are copied from the closest SDK match so
 * the UI still renders sensible names and effort levels.
 *
 * Semantics from https://code.claude.com/docs/en/model-config#restrict-model-selection:
 * - `undefined` is handled by the caller (no allowlist applied).
 * - The Default option is unaffected by `availableModels` — it always remains
 *   available, even when the allowlist is `[]`.
 */
function applyAvailableModelsAllowlist(sdkModels: ModelInfo[], allowlist: string[]): ModelInfo[] {
  // Default is always preserved per the docs. Synthesize one if the SDK
  // didn't surface it so downstream code (e.g. `getAvailableModels` picking
  // `models[0]` as a fallback) still has something to work with.
  const defaultModel = sdkModels.find((m) => m.value === "default") ?? {
    value: "default",
    displayName: "Default",
    description: "",
  };
  const result: ModelInfo[] = [defaultModel];
  const seen = new Set<string>([defaultModel.value]);

  const sdkModelsWithoutDefault = sdkModels.filter((m) => m.value !== "default");

  for (const entry of allowlist) {
    const trimmed = entry.trim();
    if (!trimmed || seen.has(trimmed)) continue;

    const sdkMatch = resolveModelPreference(sdkModelsWithoutDefault, trimmed);
    if (sdkMatch) {
      result.push({ ...sdkMatch, value: trimmed });
    } else {
      result.push({ value: trimmed, displayName: trimmed, description: "" });
    }
    seen.add(trimmed);
  }

  return result;
}

async function getAvailableModels(
  query: Query,
  models: ModelInfo[],
  sdkModels: ModelInfo[],
  settingsManager: SettingsManager,
  logger: Logger,
): Promise<{ state: SessionModelState; modelInfos: ModelInfo[] }> {
  const settings = settingsManager.getSettings();

  let currentModel = models[0];
  let resolvedFromInput: string | undefined;

  // Model priority (highest to lowest):
  // 1. ANTHROPIC_MODEL environment variable
  // 2. settings.model (user configuration)
  // 3. models[0] (default first model)
  if (process.env.ANTHROPIC_MODEL) {
    const match = resolveModelPreference(models, process.env.ANTHROPIC_MODEL);
    if (match) {
      currentModel = match;
      resolvedFromInput = process.env.ANTHROPIC_MODEL;
    }
  } else if (typeof settings.model === "string") {
    const match = resolveSettingsModel(models, settings.model, logger);
    if (match) {
      currentModel = match;
      resolvedFromInput = settings.model;
    }
  }

  // Skip the setModel round-trip when we can prove the SDK has already landed
  // on the same model. Two cases qualify:
  //  (a) No override applied — currentModel stayed at models[0]; the SDK is on
  //      its own default and we have nothing to sync.
  //  (b) The resolver returned the user's input verbatim AND that value exists
  //      in the SDK's original model list — meaning no fuzzy match or
  //      allowlist rewrite was involved, and the SDK (which reads the same
  //      ANTHROPIC_MODEL / settings.json) will have arrived at the same entry.
  // Anything else (fuzzy match, allowlist-synthesized value, alias) gets a
  // setModel call so we don't drift from the user's intended pin.
  const sdkSawSameValue = sdkModels.some((m) => m.value === currentModel.value);
  const userInputWasFuzzyMatch =
    resolvedFromInput !== undefined && currentModel.value !== resolvedFromInput;
  if (resolvedFromInput === undefined) {
    // Box-default config (no ANTHROPIC_MODEL / settings.model override). ACTIVELY
    // reset the SDK to its own default model instead of skipping setModel. On
    // session/new this is a no-op (the SDK is already on its default); on
    // session/resume it CLEARS a `/model <x>` slash command the SDK persisted and
    // replayed from the transcript, so the live serving model matches the
    // advertised "default" (box default == Opus 4.8 / 1M) rather than a stale pin.
    // The literal string "default" is NOT a real model id; `undefined` is the
    // SDK's documented "use the default" (sdk.d.ts: setModel(model?: string) —
    // "or undefined to use the default").
    await query.setModel(undefined);
  } else {
    // Concrete override resolved from input: skip only when the SDK already landed
    // on the exact same value (no fuzzy/alias rewrite). Unchanged from before.
    const skipSetModel = !userInputWasFuzzyMatch && sdkSawSameValue;
    if (!skipSetModel) {
      // Preserve an explicit `[1m]` context hint from the resolved input so a
      // session created/resumed on a "sonnet[1m]"/"opus[1m]" pin runs the
      // long-context beta from the start. `resolveModelPreference` strips the
      // suffix down to the base ModelInfo; the binary re-derives it. Without
      // this the adapter advertises 1M (via the relabel below) while the SDK
      // silently runs the base model at 200k.
      await query.setModel(effectiveRunModelId(resolvedFromInput, currentModel.value));
    }
  }

  // Keep the advertised model id STABLE for the user's pinned model across
  // session/new and session/resume. On session/new the SDK surfaces the
  // user's configured alias verbatim as a model value (e.g. "opus[1m]"), but
  // on session/resume the on-disk Claude Code session is pinned to the
  // resolved concrete id, so the SDK surfaces that instead (e.g.
  // "claude-opus-4-8[1m]"). A generic ACP client (acpx) persists the alias it
  // first saw and re-asserts it against the advertised set on every reconnect
  // via an exact-string match; if we let the advertised value drift to the
  // resolved id on resume, that gate rejects the still-valid alias and the
  // session can no longer be revived. So when the user's configured input
  // fuzzily resolved to a different concrete id, re-label that one resolved
  // entry under the alias the user/SDK uses on new. `setModel` above already
  // received the resolved `currentModel.value`, so this only affects what we
  // advertise — not what the SDK runs. The relabeled list is returned as
  // `modelInfos` so downstream consumers (configOptions, effort gating,
  // setModel lookups) stay consistent with the advertised ids. Robust across
  // future model bumps: "opus[1m]" keeps working when the resolved id changes.
  let advertisedModels = models;
  let advertisedCurrentId = currentModel.value;
  if (userInputWasFuzzyMatch && resolvedFromInput) {
    const aliasAlreadyAdvertised = models.some((m) => m.value === resolvedFromInput);
    if (!aliasAlreadyAdvertised) {
      advertisedModels = models.map((model) =>
        model.value === currentModel.value ? { ...model, value: resolvedFromInput } : model,
      );
      advertisedCurrentId = resolvedFromInput;
    }
  }

  return {
    state: {
      availableModels: advertisedModels.map((model) => ({
        modelId: model.value,
        name: model.displayName,
        description: model.description,
      })),
      currentModelId: advertisedCurrentId,
    },
    modelInfos: advertisedModels,
  };
}

function getAvailableSlashCommands(commands: SlashCommand[]): AvailableCommand[] {
  const UNSUPPORTED_COMMANDS = [
    "clear",
    "cost",
    "keybindings-help",
    "login",
    "logout",
    "output-style:new",
    "release-notes",
    "todos",
  ];

  return commands
    .map((command) => {
      const input = command.argumentHint
        ? {
            hint: Array.isArray(command.argumentHint)
              ? command.argumentHint.join(" ")
              : command.argumentHint,
          }
        : null;
      let name = command.name;
      if (command.name.endsWith(" (MCP)")) {
        name = `mcp:${name.replace(" (MCP)", "")}`;
      }
      return {
        name,
        description: command.description || "",
        input,
      };
    })
    .filter((command: AvailableCommand) => !UNSUPPORTED_COMMANDS.includes(command.name));
}

function formatUriAsLink(uri: string): string {
  try {
    if (uri.startsWith("file://")) {
      const path = uri.slice(7); // Remove "file://"
      const name = path.split("/").pop() || path;
      return `[@${name}](${uri})`;
    } else if (uri.startsWith("zed://")) {
      const parts = uri.split("/");
      const name = parts[parts.length - 1] || uri;
      return `[@${name}](${uri})`;
    }
    return uri;
  } catch {
    return uri;
  }
}

export function promptToClaude(prompt: PromptRequest): SDKUserMessage {
  const content: any[] = [];
  const context: any[] = [];

  for (const chunk of prompt.prompt) {
    switch (chunk.type) {
      case "text": {
        let text = chunk.text;
        // change /mcp:server:command args -> /server:command (MCP) args
        const mcpMatch = text.match(/^\/mcp:([^:\s]+):(\S+)(?:\s(.*))?$/);
        if (mcpMatch) {
          const [, server, command, args] = mcpMatch;
          text = `/${server}:${command} (MCP)${args ? ` ${args}` : ""}`;
        }
        content.push({ type: "text", text });
        break;
      }
      case "resource_link": {
        const formattedUri = formatUriAsLink(chunk.uri);
        content.push({
          type: "text",
          text: formattedUri,
        });
        break;
      }
      case "resource": {
        if ("text" in chunk.resource) {
          const formattedUri = formatUriAsLink(chunk.resource.uri);
          content.push({
            type: "text",
            text: formattedUri,
          });
          context.push({
            type: "text",
            text: `\n<context ref="${chunk.resource.uri}">\n${chunk.resource.text}\n</context>`,
          });
        }
        // Ignore blob resources (unsupported)
        break;
      }
      case "image":
        if (chunk.data) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              data: chunk.data,
              media_type: chunk.mimeType,
            },
          });
        } else if (chunk.uri && chunk.uri.startsWith("http")) {
          content.push({
            type: "image",
            source: {
              type: "url",
              url: chunk.uri,
            },
          });
        }
        break;
      // Ignore audio and other unsupported types
      default:
        break;
    }
  }

  content.push(...context);

  return {
    type: "user",
    message: {
      role: "user",
      content: content,
    },
    session_id: prompt.sessionId,
    parent_tool_use_id: null,
  };
}

/**
 * Convert an SDKAssistantMessage (Claude) to a SessionNotification (ACP).
 * Only handles text, image, and thinking chunks for now.
 */
export function toAcpNotifications(
  content: string | ContentBlockParam[] | BetaContentBlock[] | BetaRawContentBlockDelta[],
  role: "assistant" | "user",
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    registerHooks?: boolean;
    clientCapabilities?: ClientCapabilities;
    parentToolUseId?: string | null;
    cwd?: string;
    taskState?: TaskState;
    subagentCache?: Map<string, SubagentInfo>;
  },
): SessionNotification[] {
  const taskState = options?.taskState ?? new Map();
  const registerHooks = options?.registerHooks !== false;
  const supportsTerminalOutput = options?.clientCapabilities?._meta?.["terminal_output"] === true;
  if (typeof content === "string") {
    const update: SessionNotification["update"] = {
      sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
      content: {
        type: "text",
        text: content,
      },
    };

    if (options?.parentToolUseId) {
      const subagent = options.subagentCache?.get(options.parentToolUseId);
      update._meta = {
        ...update._meta,
        claudeCode: {
          ...(update._meta?.claudeCode || {}),
          parentToolUseId: options.parentToolUseId,
          ...(subagent
            ? {
                subagentId: subagent.agentId,
                subagentName: subagent.name,
                ...(subagent.color !== undefined ? { subagentColor: subagent.color } : {}),
              }
            : {}),
        },
      };
    }

    return [{ sessionId, update }];
  }

  const output = [];
  // Only handle the first chunk for streaming; extend as needed for batching
  for (const chunk of content) {
    let update: SessionNotification["update"] | null = null;
    switch (chunk.type) {
      case "text":
      case "text_delta":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "text",
            text: chunk.text,
          },
        };
        break;
      case "image":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "image",
            data: chunk.source.type === "base64" ? chunk.source.data : "",
            mimeType: chunk.source.type === "base64" ? chunk.source.media_type : "",
            uri: chunk.source.type === "url" ? chunk.source.url : undefined,
          },
        };
        break;
      case "thinking":
      case "thinking_delta":
        update = {
          sessionUpdate: "agent_thought_chunk",
          content: {
            type: "text",
            text: chunk.thinking,
          },
        };
        break;
      case "tool_use":
      case "server_tool_use":
      case "mcp_tool_use": {
        const alreadyCached = chunk.id in toolUseCache;
        toolUseCache[chunk.id] = chunk;
        if (chunk.name === "TodoWrite") {
          // @ts-expect-error - sometimes input is empty object or undefined
          if (Array.isArray(chunk.input?.todos)) {
            update = {
              sessionUpdate: "plan",
              entries: planEntries(chunk.input as { todos: ClaudePlanEntry[] }),
            };
          }
        } else if (
          chunk.name === "TaskCreate" ||
          chunk.name === "TaskUpdate" ||
          chunk.name === "TaskList" ||
          chunk.name === "TaskGet"
        ) {
          // Task* tool_use is suppressed; the plan update is emitted at
          // tool_result time once we have the task ID (for TaskCreate) and
          // confirmation that the change took effect.
        } else {
          // Only register hooks on first encounter to avoid double-firing
          if (registerHooks && !alreadyCached) {
            registerHookCallback(chunk.id, {
              onPostToolUseHook: async (toolUseId, toolInput, toolResponse) => {
                const toolUse = toolUseCache[toolUseId];
                if (toolUse) {
                  // Both `Edit` and `Write` produce a structuredPatch in their
                  // PostToolUse tool_response. For Edit the diff replaces the
                  // optimistic content built at tool_use time. For Write the
                  // optimistic content (built from `input.content` alone with
                  // `oldText: null`) shows "creation" semantics regardless of
                  // whether the file existed; the structuredPatch from the
                  // hook lets us emit the real diff for `type: "update"`. The
                  // helper returns `{}` if the response shape isn't usable.
                  const editDiff =
                    toolUse.name === "Edit" || toolUse.name === "Write"
                      ? toolUpdateFromDiffToolResponse(toolResponse)
                      : {};
                  const update: SessionNotification["update"] = {
                    _meta: {
                      claudeCode: {
                        toolResponse,
                        toolName: toolUse.name,
                      },
                    } satisfies ToolUpdateMeta,
                    toolCallId: toolUseId,
                    sessionUpdate: "tool_call_update",
                    ...editDiff,
                  };
                  await client.sessionUpdate({
                    sessionId,
                    update,
                  });
                } else {
                  logger.error(
                    `[claude-agent-acp] Got a tool response for tool use that wasn't tracked: ${toolUseId}`,
                  );
                }
              },
            });
          }

          let rawInput;
          try {
            rawInput = JSON.parse(JSON.stringify(chunk.input));
          } catch {
            // ignore if we can't turn it to JSON
          }

          if (alreadyCached) {
            // Second encounter (full assistant message after streaming) —
            // send as tool_call_update to refine the existing tool_call
            // rather than emitting a duplicate tool_call.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call_update",
              rawInput,
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          } else {
            // First encounter (streaming content_block_start or replay) —
            // send as tool_call with terminal_info for Bash tools.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
                ...(chunk.name === "Bash" && supportsTerminalOutput
                  ? { terminal_info: { terminal_id: chunk.id } }
                  : {}),
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call",
              rawInput,
              status: "pending",
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          }
        }
        break;
      }

      case "tool_result":
      case "tool_search_tool_result":
      case "web_fetch_tool_result":
      case "web_search_tool_result":
      case "code_execution_tool_result":
      case "bash_code_execution_tool_result":
      case "text_editor_code_execution_tool_result":
      case "mcp_tool_result": {
        const toolUse = toolUseCache[chunk.tool_use_id];
        if (!toolUse) {
          logger.error(
            `[claude-agent-acp] Got a tool result for tool use that wasn't tracked: ${chunk.tool_use_id}`,
          );
          break;
        }

        if (
          toolUse.name === "TaskCreate" ||
          toolUse.name === "TaskUpdate" ||
          toolUse.name === "TaskList" ||
          toolUse.name === "TaskGet"
        ) {
          // Headless/SDK sessions emit Task* tools instead of TodoWrite.
          // TaskCreate / TaskUpdate mutate the accumulated task list; TaskList
          // and TaskGet are read-only so we just suppress their tool_call /
          // tool_result events. The plan update is emitted as a snapshot of
          // the accumulated state, mirroring the legacy TodoWrite behavior.
          const isError = "is_error" in chunk && chunk.is_error;
          if (!isError) {
            if (toolUse.name === "TaskCreate") {
              applyTaskCreate(
                taskState,
                toolUse.input as Parameters<typeof applyTaskCreate>[1],
                parseTaskCreateOutput(chunk.content),
              );
            } else if (toolUse.name === "TaskUpdate") {
              applyTaskUpdate(taskState, toolUse.input as Parameters<typeof applyTaskUpdate>[1]);
            }
          }
          if (!isError && (toolUse.name === "TaskCreate" || toolUse.name === "TaskUpdate")) {
            update = {
              sessionUpdate: "plan",
              entries: taskStateToPlanEntries(taskState),
            };
          }
        } else if (toolUse.name !== "TodoWrite") {
          const { _meta: toolMeta, ...toolUpdate } = toolUpdateFromToolResult(
            chunk,
            toolUseCache[chunk.tool_use_id],
            supportsTerminalOutput,
          );

          // When terminal output is supported, send terminal_output as a
          // separate notification to match codex-acp's streaming lifecycle:
          //   1. tool_call       → _meta.terminal_info  (already sent above)
          //   2. tool_call_update → _meta.terminal_output (sent here)
          //   3. tool_call_update → _meta.terminal_exit  (sent below with status)
          if (toolMeta?.terminal_output) {
            output.push({
              sessionId,
              update: {
                _meta: {
                  terminal_output: toolMeta.terminal_output,
                  ...(options?.parentToolUseId
                    ? {
                        claudeCode: {
                          parentToolUseId: options.parentToolUseId,
                          ...(options.subagentCache?.get(options.parentToolUseId)
                            ? {
                                subagentId: options.subagentCache.get(options.parentToolUseId)!
                                  .agentId,
                                subagentName: options.subagentCache.get(options.parentToolUseId)!
                                  .name,
                                ...(options.subagentCache.get(options.parentToolUseId)!.color !==
                                undefined
                                  ? {
                                      subagentColor: options.subagentCache.get(
                                        options.parentToolUseId,
                                      )!.color,
                                    }
                                  : {}),
                              }
                            : {}),
                        },
                      }
                    : {}),
                },
                toolCallId: chunk.tool_use_id,
                sessionUpdate: "tool_call_update" as const,
              },
            });
          }

          update = {
            _meta: {
              claudeCode: {
                toolName: toolUse.name,
              },
              ...(toolMeta?.terminal_exit ? { terminal_exit: toolMeta.terminal_exit } : {}),
            } satisfies ToolUpdateMeta,
            toolCallId: chunk.tool_use_id,
            sessionUpdate: "tool_call_update",
            status: "is_error" in chunk && chunk.is_error ? "failed" : "completed",
            rawOutput: chunk.content,
            ...toolUpdate,
          };

          // Signal to the queue owner that this session has scheduled future
          // activity so it can disable its idle TTL and keep the process tree
          // alive until the scheduled wakeup fires.
          const isSchedulingTool =
            toolUse.name === "ScheduleWakeup" ||
            toolUse.name === "CronCreate" ||
            toolUse.name === "RemoteTrigger";
          const isToolError = "is_error" in chunk && chunk.is_error;
          if (isSchedulingTool && !isToolError) {
            output.push({
              sessionId,
              update: {
                sessionUpdate: "tool_call_update" as const,
                toolCallId: chunk.tool_use_id,
                _meta: {
                  claudeCode: {
                    hasScheduledWakeup: true,
                  },
                },
              },
            });
          }
        }
        break;
      }

      case "document":
      case "search_result":
      case "redacted_thinking":
      case "input_json_delta":
      case "citations_delta":
      case "signature_delta":
      case "container_upload":
      case "compaction":
      case "compaction_delta":
      case "advisor_tool_result":
      case "mid_conv_system":
        break;

      default:
        unreachable(chunk, logger);
        break;
    }
    if (update) {
      if (options?.parentToolUseId) {
        const subagent = options.subagentCache?.get(options.parentToolUseId);
        update._meta = {
          ...update._meta,
          claudeCode: {
            ...(update._meta?.claudeCode || {}),
            parentToolUseId: options.parentToolUseId,
            ...(subagent
              ? {
                  subagentId: subagent.agentId,
                  subagentName: subagent.name,
                  ...(subagent.color !== undefined ? { subagentColor: subagent.color } : {}),
                }
              : {}),
          },
        };
      }
      output.push({ sessionId, update });
    }
  }

  return output;
}

export function streamEventToAcpNotifications(
  message: SDKPartialAssistantMessage,
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    clientCapabilities?: ClientCapabilities;
    cwd?: string;
    taskState?: TaskState;
    subagentCache?: Map<string, SubagentInfo>;
  },
): SessionNotification[] {
  const event = message.event;
  switch (event.type) {
    case "content_block_start":
      return toAcpNotifications(
        [event.content_block],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
          taskState: options?.taskState,
          subagentCache: options?.subagentCache,
        },
      );
    case "content_block_delta":
      return toAcpNotifications(
        [event.delta],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
          taskState: options?.taskState,
          subagentCache: options?.subagentCache,
        },
      );
    // No content. `ping` is a Messages-API keep-alive event that the SDK's
    // `BetaRawMessageStreamEvent` union doesn't include even though the
    // wire format emits it; the `as never` cast lets us no-op it here
    // instead of letting it fall through to `unreachable`.
    case "ping" as never:
    case "message_start":
    case "message_delta":
    case "message_stop":
    case "content_block_stop":
      return [];

    default:
      unreachable(event, logger);
      return [];
  }
}

export function runAcp() {
  const input = nodeToWebWritable(process.stdout);
  const output = nodeToWebReadable(process.stdin);

  const stream = ndJsonStream(input, output);
  let agent!: ClaudeAcpAgent;
  const connection = new AgentSideConnection((client) => {
    agent = new ClaudeAcpAgent(client);
    return agent;
  }, stream);
  return { connection, agent };
}

function commonPrefixLength(a: string, b: string) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) {
    i++;
  }
  return i;
}

/** Best-effort first guess of a model's context window, used only until a
 *  `result` message arrives with the authoritative `modelUsage.contextWindow`.
 *
 *  Two pre-result signals, checked in order:
 *  1. A `1m` token in the model ID — Anthropic's explicit 1M-context variants
 *     and the `opus[1m]`/`sonnet[1m]` display aliases encode "1m" as a distinct
 *     token (e.g. "claude-opus-4-6-1m"); `\b1m\b` catches it without also
 *     matching "10m" or an embedded substring.
 *  2. "1M context" in the model's `description`. This is the ONLY pre-result
 *     signal that separates the box-default `default` model (Opus 4.8 *with 1M
 *     context* → 1,000,000) from plain `opus` (Opus 4.8 → 200,000): both
 *     resolve to the same base API model id (`claude-opus-4-8`), so the ID
 *     alone cannot tell them apart. The model menu's own description carries
 *     the distinction — the `default` ModelInfo reads "Opus 4.8 with 1M
 *     context", while `opus`'s reads just "Opus 4.8" and stays at the default.
 *     `description` is the SDK `ModelInfo.description` ("Description of the
 *     model's capabilities"); there is no structured context-window field. */
export function inferContextWindowFromModel(model: string, description?: string): number | null {
  if (/\b1m\b/i.test(model)) return 1_000_000;
  if (description && /\b1m\b[\s_-]*context/i.test(description)) return 1_000_000;
  return null;
}

function parseModelConfig(
  raw: string | undefined,
): { modelOverrides?: Record<string, string>; availableModels?: string[] } | undefined {
  if (!raw) return undefined;
  const parsed = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("CLAUDE_MODEL_CONFIG must be a JSON object");
  }
  const result: { modelOverrides?: Record<string, string>; availableModels?: string[] } = {};
  if (parsed.modelOverrides !== undefined) result.modelOverrides = parsed.modelOverrides;
  if (parsed.availableModels !== undefined) result.availableModels = parsed.availableModels;
  return Object.keys(result).length > 0 ? result : undefined;
}

function getMatchingModelUsage(modelUsage: Record<string, ModelUsage>, currentModel: string) {
  let bestKey: string | null = null;
  let bestLen = 0;

  for (const key of Object.keys(modelUsage)) {
    const len = commonPrefixLength(key, currentModel);
    if (len > bestLen) {
      bestLen = len;
      bestKey = key;
    }
  }

  if (bestKey) {
    return modelUsage[bestKey];
  }
}
