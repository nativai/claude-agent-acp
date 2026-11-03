import { readFileSync } from "node:fs";

const controlImage = (() => {
  try {
    return readFileSync("/etc/nativai/image-role", "utf8").trim() === "control";
  } catch {
    return false;
  }
})();

export function isControlPod(): boolean {
  return controlImage || process.env.NV_DEV_SERVER_ROLE === "control";
}
