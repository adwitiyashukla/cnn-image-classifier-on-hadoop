import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../space/doodle.js", import.meta.url));
const doodle = await import("data:text/javascript;base64," + source.toString("base64"));
const jobs = JSON.parse(readFileSync(0, "utf8"));
const results = jobs.map(([name, ...args]) => {
  const value = doodle[name](...args);
  return ArrayBuffer.isView(value) ? Array.from(value) : value;
});
process.stdout.write(JSON.stringify(results));
