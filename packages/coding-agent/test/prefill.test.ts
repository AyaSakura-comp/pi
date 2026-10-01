import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentEvent, AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import {
	PrefillController,
	type PrefillControllerOptions,
	type PrefillSettings,
	SlotLeases,
} from "../src/core/prefill.ts";

const model = {
	id: "qwen",
	provider: "local-llama",
	api: "openai-completions",
	baseUrl: "http://127.0.0.1:8001/v1",
} as Model<"openai-completions">;

function assistant(ids: string[]): AssistantMessage {
	return {
		role: "assistant",
		content: ids.map((id) => ({ type: "toolCall" as const, id, name: "research", arguments: { q: "x" } })),
		api: "openai-completions",
		provider: "local-llama",
		model: "qwen",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: 0,
	};
}

type Deferred = { promise: Promise<void>; resolve: () => void };
function deferred(): Deferred {
	let resolve = () => {};
	const promise = new Promise<void>((r) => {
		resolve = r;
	});
	return { promise, resolve };
}

// Lease operations do real file I/O, so let the event loop run for a while.
const flush = () => new Promise((resolve) => setTimeout(resolve, 40));

interface Harness {
	controller: PrefillController;
	completions: Array<{ prompt: string; id_slot: number }>;
	/** Gate for each /completion call, released in order by the test. */
	gates: Deferred[];
	aborted: number;
	messages: AgentMessage[];
	settings: PrefillSettings;
	sessionId: string;
	previewSlot?: number;
}

let leaseDir: string;
beforeEach(() => {
	leaseDir = mkdtempSync(join(tmpdir(), "pi-prefill-test-"));
});
afterEach(() => {
	rmSync(leaseDir, { recursive: true, force: true });
});

function harness(overrides: Partial<Harness> = {}, options: Partial<PrefillControllerOptions> = {}): Harness {
	const h: Harness = {
		completions: [],
		gates: [],
		aborted: 0,
		messages: [{ role: "user", content: "question", timestamp: 0 }, assistant(["call-1"])],
		settings: { enabled: true, slots: [1] },
		sessionId: "session-a",
		...overrides,
	} as Harness;
	const fetchFn = (async (url: string, init: RequestInit) => {
		const body = JSON.parse(String(init.body));
		if (url.endsWith("/apply-template")) {
			const last = body.messages.at(-1);
			return Response.json({ prompt: `<head>${last.content}<tail>` });
		}
		const gate = deferred();
		h.gates.push(gate);
		h.completions.push({ prompt: body.prompt, id_slot: body.id_slot });
		await new Promise<void>((resolve, reject) => {
			init.signal?.addEventListener("abort", () => {
				h.aborted++;
				reject(new Error("aborted"));
			});
			void gate.promise.then(resolve);
		});
		return Response.json({ timings: { prompt_n: 10, prompt_ms: 5 } });
	}) as typeof globalThis.fetch;
	h.controller = new PrefillController({
		getSettings: () => h.settings,
		getSessionId: () => h.sessionId,
		getMessages: () => h.messages,
		getModel: () => model,
		preview: async (extra) => {
			const marker = extra[0];
			const text = marker.role === "toolResult" && marker.content[0].type === "text" ? marker.content[0].text : "";
			return {
				model,
				payload: {
					model: "qwen",
					messages: [
						{ role: "user", content: "question" },
						{ role: "assistant", content: null, tool_calls: [{ id: "call-1" }] },
						{ role: "tool", tool_call_id: "call-1", content: text },
					],
					tools: [{ name: "research" }],
					...(h.previewSlot === undefined ? {} : { id_slot: h.previewSlot }),
				},
			};
		},
		leaseDir,
		fetch: fetchFn,
		...options,
	});
	return h;
}

function finalPayload(content: string, extra: Record<string, unknown> = {}) {
	return {
		model: "qwen",
		messages: [
			{ role: "user", content: "question" },
			{ role: "assistant", content: null, tool_calls: [{ id: "call-1" }] },
			{ role: "tool", tool_call_id: "call-1", content },
		],
		tools: [{ name: "research" }],
		stream: true,
		...extra,
	};
}

const assistantEnd = { type: "message_end", message: assistant([]) } as AgentEvent;

describe("PrefillController", () => {
	test("rejects begin unless enabled and bound to the only tool call of the last assistant message", () => {
		expect(harness({ settings: { enabled: false, slots: [1] } }).controller.begin("call-1")).toBeUndefined();
		expect(harness().controller.begin("other")).toBeUndefined();
		const two = harness({ messages: [assistant(["call-1", "call-2"])] });
		expect(two.controller.begin("call-1")).toBeUndefined();
		const provider = harness({ settings: { enabled: true, slots: [1], providers: ["other"] } });
		expect(provider.controller.begin("call-1")).toBeUndefined();
	});

	test("coalesces appends into one in-flight warm, pins the final request, releases after the response", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		expect(h.completions.map((c) => c.prompt)).toEqual(["<head>A"]);
		handle.append("B");
		handle.append("C");
		await flush();
		// Still one request: newer text waits for the in-flight warm.
		expect(h.completions).toHaveLength(1);
		h.gates[0].resolve();
		await flush();
		expect(h.completions.map((c) => c.prompt)).toEqual(["<head>A", "<head>ABC"]);
		h.gates[1].resolve();
		await flush();
		handle.end();
		expect(handle.append("D")).toBe(false);

		const pinned = (await h.controller.pin(finalPayload("ABC and the rest"), model)) as Record<string, unknown>;
		expect(pinned.id_slot).toBe(1);
		expect(pinned.cache_prompt).toBe(true);
		expect(handle.stats()).toMatchObject({ state: "pinned", warms: 2, warmedChars: 3, warmPromptTokens: 20 });
		expect(readdirSync(leaseDir)).toHaveLength(1);
		// A pinned handle pins once.
		expect(await h.controller.pin(finalPayload("ABC"), model)).not.toHaveProperty("id_slot");
		h.controller.onAgentEvent(assistantEnd);
		await flush();
		expect(readdirSync(leaseDir)).toHaveLength(0);
	});

	test("text committed before end() is still warmed after end(), while Pi handles the tool result", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		handle.append("B"); // arrives while the first warm is in flight
		handle.end();
		h.gates[0].resolve();
		await flush();
		expect(h.completions.map((c) => c.prompt)).toEqual(["<head>A", "<head>AB"]);
		const pinning = h.controller.pin(finalPayload("AB tail"), model);
		await flush();
		h.gates[1].resolve();
		const pinned = (await pinning) as Record<string, unknown>;
		expect(pinned.id_slot).toBe(1);
		expect(handle.stats()).toMatchObject({ warmedChars: 2, warms: 2 });
	});

	test("requests that do not carry the tool result yet are passed through and keep the handle", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		const unrelated = { model: "qwen", messages: [{ role: "user", content: "planner" }] };
		expect(await h.controller.pin(unrelated, model)).toBe(unrelated);
		expect(handle.stats().state).toBe("ready");
	});

	test("a final result that does not start with the warmed text is sent unchanged and the lease is released", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		h.gates[0].resolve();
		await flush();
		handle.end();
		const payload = finalPayload("different text");
		expect(await h.controller.pin(payload, model)).toBe(payload);
		expect(handle.stats()).toMatchObject({ state: "cancelled", reason: "result_not_prefix" });
		await flush();
		expect(readdirSync(leaseDir)).toHaveLength(0);
	});

	test("changed history, template options or model are never pinned", async () => {
		for (const [payload, reason] of [
			[finalPayload("A", { tools: [] }), "option_tools"],
			[
				{
					...finalPayload("A"),
					messages: [{ role: "user", content: "edited" }, ...finalPayload("A").messages.slice(1)],
				},
				"history_changed",
			],
			[finalPayload("A", { model: "other" }), "payload_model"],
		] as const) {
			const h = harness();
			const handle = h.controller.begin("call-1")!;
			handle.append("A");
			await flush();
			h.gates[0].resolve();
			await flush();
			handle.end();
			expect(await h.controller.pin(payload, model)).toBe(payload);
			expect(handle.stats().reason).toBe(reason);
			await flush();
		}
	});

	test("a session change between begin and the final request is never pinned", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		h.gates[0].resolve();
		await flush();
		h.sessionId = "session-b";
		const payload = finalPayload("A");
		expect(await h.controller.pin(payload, model)).toBe(payload);
		expect(handle.stats().reason).toBe("session_changed");
	});

	test("two sessions never share a slot; the slot is free again after release", async () => {
		const a = harness();
		const b = harness({ sessionId: "session-b" });
		const ha = a.controller.begin("call-1")!;
		await flush();
		const hb = b.controller.begin("call-1")!;
		await flush();
		expect(ha.stats()).toMatchObject({ state: "ready", slot: 1 });
		expect(hb.stats()).toMatchObject({ state: "failed", reason: "no_free_slot" });
		ha.cancel();
		await flush();
		const c = harness({ sessionId: "session-c" });
		const hc = c.controller.begin("call-1")!;
		await flush();
		expect(hc.stats()).toMatchObject({ state: "ready", slot: 1 });
	});

	test("concurrent begins from many sessions: exactly one wins each slot", async () => {
		const sessions = Array.from({ length: 8 }, (_, i) =>
			harness({ sessionId: `s${i}`, settings: { enabled: true, slots: [1, 2] } }),
		);
		const handles = sessions.map((h) => h.controller.begin("call-1")!);
		await flush();
		await flush();
		const ready = handles.filter((h) => h.stats().state === "ready").map((h) => h.stats().slot);
		expect(ready.sort()).toEqual([1, 2]);
	});

	test("a second begin in the same session supersedes the first and aborts its warm", async () => {
		const h = harness();
		const first = h.controller.begin("call-1")!;
		first.append("A");
		await flush();
		expect(h.completions).toHaveLength(1);
		h.messages = [...h.messages.slice(0, -1), assistant(["call-2"])];
		const second = h.controller.begin("call-2")!;
		await flush();
		expect(first.stats()).toMatchObject({ state: "cancelled", reason: "superseded" });
		expect(h.aborted).toBe(1);
		expect(second.stats().state).toBe("ready");
		expect(first.append("B")).toBe(false);
	});

	test("agent end and abort cancel the handle and release the lease", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		h.controller.onAgentEvent({ type: "agent_end", messages: [] } as AgentEvent);
		await flush();
		expect(handle.stats()).toMatchObject({ state: "cancelled", reason: "agent_end" });
		expect(h.aborted).toBe(1);
		expect(readdirSync(leaseDir)).toHaveLength(0);
	});

	test("a stale lease from a dead process is broken; a live one is respected", async () => {
		const leases = new SlotLeases(leaseDir, Date.now, (pid) => pid === 4242);
		const root = "http://127.0.0.1:8001";
		const record = { pid: 999999, sessionId: "dead", handleId: "old", expiresAt: Date.now() + 60_000 };
		writeFileSync(leases.path(root, 1), JSON.stringify(record));
		expect(await leases.acquire(root, 1, { ...record, pid: 4242, handleId: "new", sessionId: "s" })).toBe(true);
		expect(await leases.holds(root, 1, "new")).toBe(true);
		expect(await leases.acquire(root, 1, { ...record, pid: 4242, handleId: "third", sessionId: "t" })).toBe(false);
		await leases.release(root, 1, "third");
		expect(await leases.holds(root, 1, "new")).toBe(true);
		await leases.release(root, 1, "new");
		expect(existsSync(leases.path(root, 1))).toBe(false);
	});

	test("a lease taken over by another owner stops further warms", async () => {
		const h = harness();
		const handle = h.controller.begin("call-1")!;
		await flush();
		const file = readdirSync(leaseDir)[0];
		writeFileSync(
			join(leaseDir, file),
			JSON.stringify({ pid: process.pid, sessionId: "x", handleId: "thief", expiresAt: Date.now() + 60_000 }),
		);
		handle.append("A");
		await flush();
		expect(h.completions).toHaveLength(0);
		expect(handle.stats()).toMatchObject({ state: "failed", reason: "lease_lost" });
		// The foreign lease is left alone.
		expect(readdirSync(leaseDir)).toHaveLength(1);
	});

	test("uses the slot chosen by the payload pipeline and refuses to pin if it moved", async () => {
		const h = harness({ settings: { enabled: true }, previewSlot: 0 });
		const handle = h.controller.begin("call-1")!;
		handle.append("A");
		await flush();
		expect(h.completions[0].id_slot).toBe(0);
		h.gates[0].resolve();
		await flush();
		const moved = finalPayload("A", { id_slot: 1 });
		expect(await h.controller.pin(moved, model)).toBe(moved);
		expect(handle.stats()).toMatchObject({ slotSource: "session", reason: "session_slot_moved" });
	});
});
