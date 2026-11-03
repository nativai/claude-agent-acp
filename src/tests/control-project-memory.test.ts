import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendControlProjectMemory } from "../control-project-memory.js";

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

describe("control project memory is plain text", () => {
  it("keeps ancestor, project, local and .claude instructions and leaves @-imports literal", async () => {
    const root = await mkdtemp(join(tmpdir(), "claude-project-memory-"));
    roots.push(root);
    const cwd = join(root, "project");
    await mkdir(join(cwd, ".claude"), { recursive: true });
    await writeFile(join(root, "CLAUDE.md"), "ancestor instructions");
    await writeFile(join(cwd, "CLAUDE.md"), "project instructions\n@./imported.md");
    await writeFile(join(cwd, "CLAUDE.local.md"), "local instructions");
    await writeFile(join(cwd, ".claude/CLAUDE.md"), "dot-claude instructions");
    await writeFile(join(cwd, "imported.md"), "IMPORTED_SECRET_MUST_NOT_APPEAR");
    const result = await appendControlProjectMemory(cwd, {
      type: "preset",
      preset: "claude_code",
      append: "platform primer",
    });
    expect(result).toMatchObject({ type: "preset", preset: "claude_code" });
    const text = JSON.stringify(result);
    for (const phrase of [
      "platform primer",
      "ancestor instructions",
      "project instructions",
      "local instructions",
      "dot-claude instructions",
      "@./imported.md",
    ])
      expect(text).toContain(phrase);
    expect(text).not.toContain("IMPORTED_SECRET_MUST_NOT_APPEAR");
    expect(text!.indexOf("ancestor instructions")).toBeLessThan(
      text!.indexOf("project instructions"),
    );
    expect(await appendControlProjectMemory(cwd, "custom system prompt")).toContain(
      "custom system prompt",
    );
    expect(await appendControlProjectMemory(cwd, ["array prompt"])).toEqual(
      expect.arrayContaining(["array prompt", expect.stringContaining("project instructions")]),
    );
    expect(
      await appendControlProjectMemory(cwd, {
        type: "custom",
        prompt: ["custom block"],
        snapshot: true,
      }),
    ).toMatchObject({
      type: "custom",
      snapshot: true,
      prompt: expect.arrayContaining([
        "custom block",
        expect.stringContaining("project instructions"),
      ]),
    });
  });

  it("does not follow a project memory symlink into control-only files", async () => {
    const root = await mkdtemp(join(tmpdir(), "claude-memory-link-"));
    roots.push(root);
    const cwd = join(root, "project");
    await mkdir(cwd);
    await writeFile(join(root, "private.txt"), "PRIVATE_MUST_NOT_APPEAR");
    await symlink(join(root, "private.txt"), join(cwd, "CLAUDE.md"));
    expect(await appendControlProjectMemory(cwd, "platform only")).toBe("platform only");
  });
});
