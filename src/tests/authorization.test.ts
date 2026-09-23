import { describe, expect, it, Mock, vi, afterEach, beforeEach } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";
import { AgentSideConnection } from "@agentclientprotocol/sdk";

const mockQuery = vi.hoisted(() =>
  vi.fn(() => ({
    initializationResult: vi.fn().mockResolvedValue({
      models: [
        { value: "id", displayName: "name", description: "description", supportsAutoMode: true },
      ],
    }),
    setModel: vi.fn(),
    setPermissionMode: vi.fn(),
    supportedCommands: vi.fn().mockResolvedValue([]),
  })),
);

vi.mock("@anthropic-ai/claude-agent-sdk", async () => ({
  ...(await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  )),
  query: mockQuery,
}));

describe("authorization", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    //await all pending events like
    vi.runAllTimers();
    vi.useRealTimers();

    vi.unstubAllGlobals();
    vi.resetAllMocks();
  });

  async function createAgentMock(): Promise<[ClaudeAcpAgent, Mock]> {
    const connectionMock = {
      sessionUpdate: async (_: any) => {},
    } as AgentSideConnection;

    const agent = new ClaudeAcpAgent(connectionMock);

    return [agent, mockQuery];
  }

  // `initialize` picks its auth methods from an `isRemote` check over AMBIENT env
  // vars (`isRemote` in ../acp-agent.ts): NO_BROWSER, SSH_CONNECTION, SSH_CLIENT,
  // SSH_TTY or CLAUDE_CODE_REMOTE — any one of them collapses the list to the single
  // legacy `claude-login` method. Our own test transport sets SSH_CONNECTION and
  // SSH_CLIENT, because `workbench-exec` reaches the workbench over SSH. So a case
  // asserting the NON-remote methods has to establish that precondition itself
  // instead of inheriting the environment it happens to run in — otherwise it passes
  // on a developer laptop and fails on the box we actually gate on.
  //
  // ⚠️ DO NOT "improve" this into a delete-list of the five names above. It looks
  // tidier and it is the bug: an empty env is hermetic BY CONSTRUCTION, so a sixth
  // marker added to `isRemote` is covered the day it lands, whereas a name list goes
  // stale in silence and these cases start failing again only on hosts that set the
  // new var. Returns a fresh object per call so a mutation cannot leak between cases.
  //
  // The opposite branch stays executable in the two `... falls back to single legacy
  // login method` cases below, which SET a marker and assert `claude-login`.
  function nonRemoteEnv(): Record<string, string | undefined> {
    return {};
  }

  it("gateway auth not offered without capability", async () => {
    const [agent] = await createAgentMock();

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
    });
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "gateway" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "gateway-bedrock" }),
    );
  });

  it("gateway auth offered when client advertises auth._meta.gateway capability", async () => {
    const [agent] = await createAgentMock();

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        auth: { _meta: { gateway: true } },
      } as any,
    });
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({
        id: "gateway",
        _meta: { gateway: { protocol: "anthropic" } },
      }),
    );
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({
        id: "gateway-bedrock",
        _meta: { gateway: { protocol: "bedrock" } },
      }),
    );
  });

  it("uses gateway env after anthropic gateway auth", async () => {
    const [agent, mockQuery] = await createAgentMock();

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        auth: { terminal: true, _meta: { gateway: true } },
      } as any,
    });
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "gateway" }),
    );

    await agent.authenticate({
      methodId: "gateway",
      _meta: { gateway: { baseUrl: "https://gateway.example", headers: { "x-api-key": "test" } } },
    });

    await agent.newSession({
      cwd: "testRoot",
      mcpServers: [],
      _meta: {
        claudeCode: {
          options: {
            env: {
              userEnv: "userEnv",
            },
          },
        },
      },
    });

    expect(mockQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        options: expect.objectContaining({
          env: expect.objectContaining({
            ANTHROPIC_AUTH_TOKEN: " ",
            ANTHROPIC_BASE_URL: "https://gateway.example",
            ANTHROPIC_CUSTOM_HEADERS: "x-api-key: test",
            userEnv: "userEnv",
          }),
        }),
      }),
    );
  });

  it("uses gateway env after gateway auth", async () => {
    const [agent, mockQuery] = await createAgentMock();

    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        auth: { terminal: true, _meta: { gateway: true } },
      } as any,
    });

    await agent.authenticate({
      methodId: "gateway-bedrock",
      _meta: {
        gateway: { baseUrl: "https://gateway.example", headers: { "custom-header": "test" } },
      },
    });

    await agent.newSession({
      cwd: "testRoot",
      mcpServers: [],
    });

    expect(mockQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        options: expect.objectContaining({
          env: expect.objectContaining({
            CLAUDE_CODE_USE_BEDROCK: "1",
            AWS_BEARER_TOKEN_BEDROCK: " ",
            ANTHROPIC_BEDROCK_BASE_URL: "https://gateway.example",
            ANTHROPIC_CUSTOM_HEADERS: "custom-header: test",
          }),
        }),
      }),
    );
  });

  it("hide claude authentication without terminal-auth", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, argv: ["--hide-claude-auth"] });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        auth: { _meta: { gateway: true } },
      } as any,
    });
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "gateway" }),
    );
  });

  it("hide claude auth but still show console login when terminal-auth is set", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, argv: ["--hide-claude-auth"], env: nonRemoteEnv() });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        _meta: { "terminal-auth": true },
      },
    });
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
  });

  it("hide claude auth but still show console login with terminal capability", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, argv: ["--hide-claude-auth"], env: nonRemoteEnv() });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        auth: { terminal: true },
      },
    });
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
  });

  it("SSH session falls back to single legacy login method", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, env: { ...process.env, SSH_TTY: "/dev/pts/0" } });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: { auth: { terminal: true } },
    });

    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "claude-login" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
  });

  it("CLAUDE_CODE_REMOTE falls back to single legacy login method", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, env: { ...process.env, CLAUDE_CODE_REMOTE: "1" } });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: { auth: { terminal: true } },
    });

    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "claude-login" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
  });

  it("remote environment respects hide-claude-auth", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", {
      ...process,
      argv: ["--hide-claude-auth"],
      env: { ...process.env, SSH_CONNECTION: "192.168.1.1 12345 192.168.1.2 22" },
    });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: { auth: { terminal: true } },
    });

    expect(initializeResponse.authMethods).not.toContainEqual(
      expect.objectContaining({ id: "claude-login" }),
    );
  });

  it("show claude authentication", async () => {
    const [agent] = await createAgentMock();
    vi.stubGlobal("process", { ...process, env: nonRemoteEnv() });

    const initializeResponse = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: { auth: { terminal: true } },
    });

    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "claude-ai-login" }),
    );
    expect(initializeResponse.authMethods).toContainEqual(
      expect.objectContaining({ id: "console-login" }),
    );
  });
});
