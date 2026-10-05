import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import { computeExtensionsFingerprint } from "../src/core/extensions/fingerprint.ts";
import type { Extension, ExtensionAPI } from "../src/core/extensions/types.ts";
import { createHarness } from "./suite/harness.ts";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function fixture() {
	const root = mkdtempSync(join(tmpdir(), "pi-fingerprint-"));
	roots.push(root);
	const entry = join(root, "index.ts");
	writeFileSync(entry, "export default () => 1;");
	const extensions = [{ path: entry, resolvedPath: entry }] as Extension[];
	return { root, entry, extensions };
}
describe("loaded extension fingerprint", () => {
	it("detects same-size code edits even with preserved timestamps", () => {
		const f = fixture();
		utimesSync(f.entry, 1, 1);
		const before = computeExtensionsFingerprint(f.extensions);
		writeFileSync(f.entry, "export default () => 2;");
		utimesSync(f.entry, 1, 1);
		expect(computeExtensionsFingerprint(f.extensions).fingerprint).not.toBe(before.fingerprint);
	});
	it("ignores logs, caches, tests and virtual environments", () => {
		const f = fixture();
		const before = computeExtensionsFingerprint(f.extensions);
		writeFileSync(join(f.root, "runtime.log"), "changed");
		for (const dir of [".venv", "tests", "__pycache__"]) {
			mkdirSync(join(f.root, dir));
			writeFileSync(join(f.root, dir, "code.py"), "data");
		}
		expect(computeExtensionsFingerprint(f.extensions)).toEqual(before);
	});
	it("detects a deleted loaded entry without blocking the reload", () => {
		const f = fixture();
		const before = computeExtensionsFingerprint(f.extensions);
		rmSync(f.entry);
		expect(computeExtensionsFingerprint(f.extensions).fingerprint).not.toBe(before.fingerprint);
	});
	it("tracks the real source behind a symlink entry", () => {
		const f = fixture();
		const link = join(f.root, "alias.ts");
		symlinkSync(f.entry, link);
		const extensions = [{ path: link, resolvedPath: link }] as Extension[];
		const before = computeExtensionsFingerprint(extensions);
		writeFileSync(f.entry, "export default () => 2;");
		expect(computeExtensionsFingerprint(extensions).fingerprint).not.toBe(before.fingerprint);
	});
	it("automatically reloads changed code before the next idle prompt, not unchanged files", async () => {
		const f = fixture();
		let reloads = 0;
		const prompts: string[] = [];
		const hashes: string[] = [];
		const factory = (pi: ExtensionAPI) => {
			const loadedCode = readFileSync(f.entry, "utf8");
			pi.on("session_start", (e) => {
				hashes.push(e.extensionsFingerprint!);
			});
			pi.on("before_agent_start", () => {
				prompts.push(loadedCode);
			});
		};
		let loaded = await createTestExtensionsResult([{ factory, path: f.entry }]);
		const loader = createTestResourceLoader({ extensionsResult: loaded });
		loader.getExtensions = () => loaded;
		loader.reload = async () => {
			reloads++;
			loaded = await createTestExtensionsResult([{ factory, path: f.entry }]);
		};
		const h = await createHarness({ resourceLoader: loader });
		try {
			await h.session.bindExtensions({});
			h.setResponses([fauxAssistantMessage("ok"), fauxAssistantMessage("ok"), fauxAssistantMessage("ok")]);
			await h.session.prompt("one");
			expect(reloads).toBe(0);
			writeFileSync(f.entry, "export default () => 2;");
			await h.session.prompt("two");
			expect(reloads).toBe(1);
			await h.session.prompt("three");
			expect(reloads).toBe(1);
			expect(prompts).toEqual(["export default () => 1;", "export default () => 2;", "export default () => 2;"]);
			expect(hashes).toHaveLength(2);
			expect(hashes[1]).not.toBe(hashes[0]);
		} finally {
			h.cleanup();
		}
	});
	it("delivers a fingerprint on startup and emits the change after session_start", async () => {
		const events: string[] = [];
		let fingerprint: string | undefined;
		const h = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("session_start", (event) => {
						events.push(event.type);
						fingerprint = event.extensionsFingerprint;
					});
					pi.on("extensions_changed", (event) => {
						events.push(event.type);
						expect(event.current).toBe(fingerprint);
					});
				},
			],
		});
		try {
			await h.session.bindExtensions({});
			expect(fingerprint).toMatch(/^[a-f0-9]{64}$/);
			expect(events).toEqual(["session_start", "extensions_changed"]);
		} finally {
			h.cleanup();
		}
	});
});
