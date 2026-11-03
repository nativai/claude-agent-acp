# ACP adapter for the Claude Agent SDK

[![npm](https://img.shields.io/npm/v/%40agentclientprotocol%2Fclaude-agent-acp)](https://www.npmjs.com/package/@agentclientprotocol/claude-agent-acp)

Use [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview#branding-guidelines) from [ACP-compatible](https://agentclientprotocol.com) clients!

This tool implements an ACP agent by using the official [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview), supporting:

- Context @-mentions
- Images
- Tool calls (with permission requests)
- Following
- Edit review
- TODO lists
- Interactive (and background) terminals
- Custom [Slash commands](https://docs.anthropic.com/en/docs/claude-code/slash-commands)
- Client MCP servers

Learn more about the [Agent Client Protocol](https://agentclientprotocol.com/).

## Nativai control pods

On a control pod (`NV_DEV_SERVER_ROLE=control` or `/etc/nativai/image-role`), both
the SDK and the adapter settings resolver
load user and managed settings only. Repository auth helpers, environment values,
hooks and project/local settings cannot configure the process. The control image
also supplies a managed MCP policy; explicit SDK callbacks remain available.

Ancestor/worktree `CLAUDE.md`, `.claude/CLAUDE.md` and `CLAUDE.local.md` files are
appended as plain startup instructions, preserving the existing platform primer.
`@` imports remain literal and are not expanded. Subdirectory memory is not
automatically loaded on later tool reads. Symlinks/redirected paths, non-regular
files and files over 1 MiB are skipped with a warning. User memory remains the
SDK's responsibility. These restrictions do not apply outside control pods.

## Contribution Policy

This project does not require a Contributor License Agreement (CLA). Instead, contributions are accepted under the following terms:

> By contributing to this project, you agree that your contributions will be licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). You affirm that you have the legal right to submit your work, that you are not including code you do not have rights to, and that you understand contributions are made without requiring a Contributor License Agreement (CLA).
