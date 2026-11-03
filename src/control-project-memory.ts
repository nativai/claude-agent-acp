import { constants } from "node:fs";
import { open, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { Options } from "@anthropic-ai/claude-agent-sdk";

type SystemPrompt = Options["systemPrompt"];

/** Preserve startup project memory as data when executable settings are disabled.
 * @-imports stay literal; neither imported files nor scripts are evaluated here.
 */
export async function appendControlProjectMemory(
  cwd: string,
  systemPrompt: SystemPrompt,
): Promise<SystemPrompt> {
  const directories: string[] = [];
  let current = await realpath(cwd);
  for (;;) {
    directories.unshift(current);
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  const sections: string[] = [];
  for (const directory of directories) {
    for (const name of ["CLAUDE.md", ".claude/CLAUDE.md", "CLAUDE.local.md"]) {
      const file = join(directory, name);
      let handle;
      try {
        // A workbench must not turn a memory path into a control-secret read.
        // Check the opened descriptor as well as refusing a final symlink.
        handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
        const info = await handle.stat();
        if (!info.isFile() || (await realpath(`/proc/self/fd/${handle.fd}`)) !== file) {
          console.error(
            `claude-agent-acp: ignored non-regular or redirected project memory ${JSON.stringify(file)}`,
          );
          continue;
        }
        if (info.size > 1024 * 1024) {
          console.error(
            `claude-agent-acp: ignored project memory over 1 MiB ${JSON.stringify(file)}`,
          );
          continue;
        }
        const content = await handle.readFile("utf8");
        if (content.trim())
          sections.push(`Project instructions from ${JSON.stringify(file)}:\n${content}`);
      } catch (error) {
        const code = (error as { code?: string }).code;
        if (code !== "ENOENT" && code !== "ENOTDIR") {
          console.error(
            `claude-agent-acp: could not safely read project memory ${JSON.stringify(file)} (${code ?? "unknown"})`,
          );
        }
      } finally {
        await handle?.close();
      }
    }
  }
  if (sections.length === 0) return systemPrompt;
  const memory = sections.join("\n\n");
  if (systemPrompt === undefined) {
    return { type: "preset", preset: "claude_code", append: memory };
  }
  if (typeof systemPrompt === "string") return `${systemPrompt}\n\n${memory}`;
  if (Array.isArray(systemPrompt)) return [...systemPrompt, memory];
  if (systemPrompt.type === "custom") {
    return {
      ...systemPrompt,
      prompt:
        typeof systemPrompt.prompt === "string"
          ? `${systemPrompt.prompt}\n\n${memory}`
          : [...systemPrompt.prompt, memory],
    };
  }
  return { ...systemPrompt, append: [systemPrompt.append, memory].filter(Boolean).join("\n\n") };
}
