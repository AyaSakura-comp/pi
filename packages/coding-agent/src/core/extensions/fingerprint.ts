/**
 * Fingerprint of the loaded extension set, so extensions can react when any of
 * them was installed, removed or updated since the last pi start (e.g. to rebuild
 * caches derived from the system prompt and tool definitions they contribute).
 *
 * Hash executable/config contents, not mtimes: touching files and runtime logs
 * must not trigger an update, and same-size code replacements must be detected.
 */

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import type { Extension } from "./types.ts";

export interface ExtensionsFingerprint {
	fingerprint: string;
	entries: Array<{ path: string; hash: string }>;
}

export interface ExtensionsChange {
	previous?: string;
	current: string;
	/** Extension entry paths that were added, removed or changed. */
	changed: string[];
}

const SKIP_DIRS = new Set([
	"node_modules",
	".git",
	".venv",
	"venv",
	"__pycache__",
	"test",
	"tests",
	"docs",
	"coverage",
	".cache",
]);
const MAX_FILES = 5000;

function packageRoot(entry: string): string {
	let dir = path.dirname(entry);
	for (let i = 0; i < 3; i++) {
		if (fs.existsSync(path.join(dir, "package.json"))) return dir;
		const parent = path.dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return path.dirname(entry);
}

function hashTree(root: string): string {
	const hash = createHash("sha256");
	const files: string[] = [];
	const walk = (dir: string) => {
		let names: fs.Dirent[];
		try {
			names = fs.readdirSync(dir, { withFileTypes: true });
		} catch {
			return;
		}
		for (const entry of names.sort((a, b) => a.name.localeCompare(b.name))) {
			if (files.length >= MAX_FILES) throw new Error(`Extension fingerprint exceeds ${MAX_FILES} files: ${root}`);
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				if (!SKIP_DIRS.has(entry.name)) walk(full);
			} else if (
				entry.isFile() &&
				/\.(?:ts|js|mjs|cjs|json|py|sh)$/.test(entry.name) &&
				!/\.(?:test|spec)\./.test(entry.name)
			) {
				files.push(full);
			}
		}
	};
	walk(root);
	for (const file of files.sort()) {
		try {
			hash.update(`${path.relative(root, file)}\0`);
			hash.update(fs.readFileSync(file));
			hash.update("\0");
		} catch {
			// Vanished while walking; leave it out.
		}
	}
	return hash.digest("hex");
}

export function computeExtensionsFingerprint(extensions: Extension[]): ExtensionsFingerprint {
	const entries = extensions
		.filter((extension) => !extension.path.startsWith("<"))
		.map((extension) => {
			try {
				const entry = fs.realpathSync(extension.resolvedPath);
				return { path: entry, hash: hashTree(packageRoot(entry)) };
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
				// A removed entry is a change to reload, not a permanent prompt blocker.
				return { path: extension.resolvedPath, hash: "missing" };
			}
		})
		.sort((a, b) => a.path.localeCompare(b.path));
	const fingerprint = createHash("sha256")
		.update(entries.map((entry) => `${entry.path}\0${entry.hash}`).join("\n"))
		.digest("hex");
	return { fingerprint, entries };
}

/**
 * Compares against the fingerprint recorded at `statePath` and records the current one.
 * Returns the change, or undefined when the extension set is unchanged.
 */
export function recordExtensionsFingerprint(
	current: ExtensionsFingerprint,
	statePath: string,
): ExtensionsChange | undefined {
	let previous: ExtensionsFingerprint | undefined;
	try {
		previous = JSON.parse(fs.readFileSync(statePath, "utf-8")) as ExtensionsFingerprint;
	} catch {
		previous = undefined;
	}
	if (previous?.fingerprint === current.fingerprint) return undefined;

	const before = new Map((previous?.entries ?? []).map((entry) => [entry.path, entry.hash]));
	const after = new Map(current.entries.map((entry) => [entry.path, entry.hash]));
	const changed = [...new Set([...before.keys(), ...after.keys()])]
		.filter((entryPath) => before.get(entryPath) !== after.get(entryPath))
		.sort();
	try {
		fs.mkdirSync(path.dirname(statePath), { recursive: true });
		const tmp = `${statePath}.${process.pid}.tmp`;
		fs.writeFileSync(tmp, JSON.stringify(current, null, 2));
		fs.renameSync(tmp, statePath);
	} catch {
		// Unwritable state: report the change again next start rather than fail startup.
	}
	return { previous: previous?.fingerprint, current: current.fingerprint, changed };
}
