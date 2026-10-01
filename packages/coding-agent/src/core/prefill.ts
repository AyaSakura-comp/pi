/**
 * Speculative prompt prefill for llama.cpp servers while a tool is still running.
 *
 * Problem: a slow tool (for example web research) produces its result piece by
 * piece, but the model only starts reading it after the tool returns, so the
 * whole result is prefilled on the critical path.
 *
 * Solution: an extension tool calls `ctx.prefill.begin(toolCallId)`, then
 * `append(text)` for every stable piece of its future result, then `end()`.
 * The controller renders the exact prompt the next request will use (via the
 * agent's real payload pipeline, cut at a marker placed in the tool result),
 * and sends `head + appended text` to the llama.cpp `/completion` endpoint with
 * `n_predict: 0` on one slot. The next real request is pinned to that slot, so
 * llama.cpp reuses the warmed KV cache and only prefills the remaining suffix.
 *
 * Warms are pure textual prefixes of the final prompt. Hybrid/recurrent models
 * cannot roll their cache back to arbitrary positions, so anything else would
 * not be reused.
 *
 * Isolation and race rules:
 * - One controller per session. A handle is bound to the session id and to the
 *   tool call id of the current assistant message, which must contain exactly
 *   one tool call. A second `begin` in the same session cancels the first.
 * - A slot is only warmed while this handle holds a lease file for it. Leases
 *   are exclusive across sessions and processes, carry pid and expiry, and are
 *   re-checked before every warm. Stale leases (dead pid or expired) are broken.
 * - At most one warm request per handle is in flight; newer appends coalesce.
 * - The final payload is pinned only if it matches the preview exactly (all
 *   earlier messages, template options, model) and its tool result starts with
 *   the warmed text. Otherwise it is sent unchanged. A handle pins at most once.
 * - Abort, agent end, timeout, model change and dispose cancel the handle,
 *   abort its in-flight request and release the lease.
 *
 * Correctness never depends on the warm: llama.cpp only reuses cached tokens
 * that match the final prompt exactly. A lost race costs time, not accuracy.
 */

import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { readFile, rename, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { AgentEvent, AgentMessage } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";

export interface PrefillSettings {
	/** Master switch. Default: false. */
	enabled?: boolean;
	/** Providers allowed to prefill. Default: every provider using the openai-completions API. */
	providers?: string[];
	/** Slots to lease when the payload does not already name one via `id_slot`. */
	slots?: number[];
	/** Handle lifetime in milliseconds. Default: 15 minutes. */
	ttlMs?: number;
}

export type PrefillState = "starting" | "ready" | "ended" | "pinned" | "cancelled" | "failed";

export interface PrefillStats {
	state: PrefillState;
	sessionId: string;
	toolCallId: string;
	slot?: number;
	/** "session": slot chosen by the payload pipeline (id_slot); "lease": slot from settings. */
	slotSource?: "session" | "lease";
	appendedChars: number;
	warmedChars: number;
	warms: number;
	/** Prompt tokens llama.cpp actually processed across warms. */
	warmPromptTokens: number;
	warmPromptMs: number;
	reason?: string;
}

export interface PrefillHandle {
	readonly toolCallId: string;
	/** Append text to the future tool result. Returns false once the handle no longer accepts text. */
	append(text: string): boolean;
	/** No more text will be appended. The next request may be pinned. */
	end(): void;
	/** Drop the prefill and release its slot. */
	cancel(reason?: string): void;
	stats(): PrefillStats;
}

export interface PrefillApi {
	/**
	 * Start a prefill for the tool call `toolCallId`, which must be the only tool
	 * call of the latest assistant message. Returns undefined when prefill is
	 * disabled or not supported for the current model.
	 *
	 * Contract: the tool's final text result must start with everything passed
	 * to `append`, in order. If it does not, the prefill is silently dropped.
	 */
	begin(toolCallId: string): PrefillHandle | undefined;
}

export interface PrefillPreview {
	model: Model<any>;
	payload: unknown;
}

export interface PrefillControllerOptions {
	getSettings: () => PrefillSettings;
	getSessionId: () => string;
	getMessages: () => AgentMessage[];
	getModel: () => Model<any> | undefined;
	/** Build the next provider payload (extension payload hooks applied, no pin) for extra messages. */
	preview: (extraMessages: AgentMessage[], signal: AbortSignal) => Promise<PrefillPreview>;
	/** Directory holding slot lease files, shared by every pi process on the host. */
	leaseDir: string;
	fetch?: typeof globalThis.fetch;
	now?: () => number;
	/** Test hook: whether a pid is alive. */
	isPidAlive?: (pid: number) => boolean;
}

type JsonObject = Record<string, unknown>;

const MARK = "⁣PI_PREFILL_MARK⁣";
const DEFAULT_TTL_MS = 15 * 60 * 1000;
const REQUEST_TIMEOUT_MS = 60_000;
const PIN_WAIT_MS = 30_000;
/** Payload keys that change the rendered prompt. Sampling and output keys may differ. */
const TEMPLATE_KEYS = [
	"tools",
	"tool_choice",
	"parallel_tool_calls",
	"chat_template_kwargs",
	"chat_template_args",
	"enable_thinking",
	"reasoning_effort",
	"response_format",
] as const;

function isObject(value: unknown): value is JsonObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stableJson(value: unknown): string {
	return JSON.stringify(value, (_key, inner) =>
		isObject(inner)
			? Object.fromEntries(
					Object.keys(inner)
						.sort()
						.map((key) => [key, inner[key]]),
				)
			: inner,
	);
}

function sameJson(a: unknown, b: unknown): boolean {
	return stableJson(a) === stableJson(b);
}

function defaultIsPidAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return (error as NodeJS.ErrnoException).code === "EPERM";
	}
}

/** Server root for llama.cpp native endpoints (`/completion`, `/apply-template`). */
export function llamaCppRoot(baseUrl: string): string {
	return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

interface LeaseRecord {
	pid: number;
	sessionId: string;
	handleId: string;
	expiresAt: number;
}

/**
 * Exclusive slot leases as files created with O_EXCL. A lease names its owner
 * handle; owners re-check the file before each use, so a lease broken by
 * another process is noticed before the slot is touched again.
 */
export class SlotLeases {
	private readonly dir: string;
	private readonly now: () => number;
	private readonly isPidAlive: (pid: number) => boolean;

	constructor(dir: string, now: () => number, isPidAlive: (pid: number) => boolean) {
		this.dir = dir;
		this.now = now;
		this.isPidAlive = isPidAlive;
	}

	path(root: string, slot: number): string {
		return join(this.dir, `${root.replace(/[^a-zA-Z0-9]+/g, "_")}__slot${slot}.json`);
	}

	private async read(file: string): Promise<LeaseRecord | undefined> {
		try {
			const parsed: unknown = JSON.parse(await readFile(file, "utf8"));
			if (
				isObject(parsed) &&
				typeof parsed.pid === "number" &&
				typeof parsed.sessionId === "string" &&
				typeof parsed.handleId === "string" &&
				typeof parsed.expiresAt === "number"
			) {
				return parsed as unknown as LeaseRecord;
			}
			return { pid: 0, sessionId: "", handleId: "", expiresAt: 0 };
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
			// Unreadable or half-written file: treat as stale.
			return { pid: 0, sessionId: "", handleId: "", expiresAt: 0 };
		}
	}

	async acquire(root: string, slot: number, record: LeaseRecord): Promise<boolean> {
		mkdirSync(this.dir, { recursive: true });
		const file = this.path(root, slot);
		for (let attempt = 0; attempt < 2; attempt++) {
			try {
				await writeFile(file, JSON.stringify(record), { flag: "wx" });
				return await this.holds(root, slot, record.handleId);
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
			}
			const current = await this.read(file);
			if (!current) continue;
			if (current.expiresAt > this.now() && current.pid > 0 && this.isPidAlive(current.pid)) return false;
			// Break a stale lease: move it aside atomically, then verify we moved the
			// stale record and not a fresh lease written in between.
			const aside = `${file}.stale-${randomUUID()}`;
			try {
				await rename(file, aside);
			} catch {
				continue;
			}
			const moved = await this.read(aside);
			await unlink(aside).catch(() => {});
			if (moved && moved.handleId !== current.handleId) return false;
		}
		return false;
	}

	async holds(root: string, slot: number, handleId: string): Promise<boolean> {
		const current = await this.read(this.path(root, slot));
		return current?.handleId === handleId && current.expiresAt > this.now();
	}

	async release(root: string, slot: number, handleId: string): Promise<void> {
		const file = this.path(root, slot);
		const current = await this.read(file);
		if (current?.handleId !== handleId) return;
		const aside = `${file}.release-${randomUUID()}`;
		try {
			await rename(file, aside);
		} catch {
			return;
		}
		const moved = await this.read(aside);
		if (moved?.handleId === handleId) {
			await unlink(aside).catch(() => {});
		} else {
			// Another owner took the slot between our read and rename: put it back.
			await rename(aside, file).catch(() => {});
		}
	}
}

class Handle implements PrefillHandle {
	readonly id = randomUUID();
	readonly toolCallId: string;
	readonly sessionId: string;
	readonly model: Model<any>;
	readonly abort = new AbortController();
	private readonly controller: PrefillController;
	state: PrefillState = "starting";
	reason?: string;
	/** Everything appended so far: the committed prefix of the tool result. */
	text = "";
	/** Longest prefix of `text` that a completed warm put into the slot. */
	warmed = "";
	head?: string;
	preview?: JsonObject;
	root?: string;
	slot?: number;
	slotSource?: "session" | "lease";
	leased = false;
	inFlight?: Promise<void>;
	warms = 0;
	warmPromptTokens = 0;
	warmPromptMs = 0;
	timer?: ReturnType<typeof setTimeout>;

	constructor(controller: PrefillController, toolCallId: string, sessionId: string, model: Model<any>) {
		this.controller = controller;
		this.toolCallId = toolCallId;
		this.sessionId = sessionId;
		this.model = model;
	}

	get accepting(): boolean {
		return this.state === "starting" || this.state === "ready";
	}

	append(text: string): boolean {
		if (!this.accepting || typeof text !== "string") return false;
		if (!text) return true;
		this.text += text;
		this.controller.schedule(this);
		return true;
	}

	end(): void {
		if (!this.accepting) return;
		this.state = "ended";
		this.controller.schedule(this);
	}

	cancel(reason = "cancelled"): void {
		this.controller.finish(this, "cancelled", reason);
	}

	stats(): PrefillStats {
		return {
			state: this.state,
			sessionId: this.sessionId,
			toolCallId: this.toolCallId,
			slot: this.slot,
			slotSource: this.slotSource,
			appendedChars: this.text.length,
			warmedChars: this.warmed.length,
			warms: this.warms,
			warmPromptTokens: this.warmPromptTokens,
			warmPromptMs: this.warmPromptMs,
			reason: this.reason,
		};
	}
}

export class PrefillController implements PrefillApi {
	private readonly options: PrefillControllerOptions;
	private readonly leases: SlotLeases;
	private readonly fetchFn: typeof globalThis.fetch;
	private readonly now: () => number;
	private active?: Handle;
	/** Handle whose final request was pinned; its lease is held until that response ends. */
	private pinned?: Handle;
	/** Release of this session's previous lease; a new handle waits for it so it can reuse the slot. */
	private releasing: Promise<void> = Promise.resolve();
	private disposed = false;

	constructor(options: PrefillControllerOptions) {
		this.options = options;
		this.now = options.now ?? Date.now;
		this.fetchFn = options.fetch ?? globalThis.fetch;
		this.leases = new SlotLeases(options.leaseDir, this.now, options.isPidAlive ?? defaultIsPidAlive);
	}

	begin(toolCallId: string): PrefillHandle | undefined {
		if (this.disposed || typeof toolCallId !== "string" || !toolCallId) return undefined;
		const settings = this.options.getSettings();
		const model = this.options.getModel();
		if (!settings.enabled || !model || model.api !== "openai-completions") return undefined;
		if (settings.providers && !settings.providers.includes(model.provider)) return undefined;
		const last = this.options.getMessages().at(-1);
		if (last?.role !== "assistant" || last.stopReason !== "toolUse") return undefined;
		const calls = last.content.filter((part) => part.type === "toolCall");
		if (calls.length !== 1 || calls[0].id !== toolCallId) return undefined;
		if (this.active?.toolCallId === toolCallId && this.active.accepting) return undefined;
		if (this.active) this.finish(this.active, "cancelled", "superseded");
		const handle = new Handle(this, toolCallId, this.options.getSessionId(), model);
		this.active = handle;
		handle.timer = setTimeout(() => this.finish(handle, "cancelled", "ttl"), settings.ttlMs ?? DEFAULT_TTL_MS);
		handle.timer.unref?.();
		handle.inFlight = this.start(handle, settings).finally(() => {
			handle.inFlight = undefined;
			this.schedule(handle);
		});
		return handle;
	}

	private async start(handle: Handle, settings: PrefillSettings): Promise<void> {
		try {
			const call = this.options
				.getMessages()
				.flatMap((message) => (message.role === "assistant" ? message.content : []))
				.find((part) => part.type === "toolCall" && part.id === handle.toolCallId);
			const marker: AgentMessage = {
				role: "toolResult",
				toolCallId: handle.toolCallId,
				toolName: call?.type === "toolCall" ? call.name : "",
				content: [{ type: "text", text: MARK }],
				isError: false,
				timestamp: this.now(),
			};
			const { model, payload } = await this.options.preview([marker], handle.abort.signal);
			if (!handle.accepting) return;
			if (model.provider !== handle.model.provider || model.id !== handle.model.id) throw new Error("model_changed");
			if (!isObject(payload) || !Array.isArray(payload.messages)) throw new Error("payload_not_chat");
			const lastMessage: unknown = payload.messages.at(-1);
			if (!isObject(lastMessage) || lastMessage.role !== "tool" || lastMessage.content !== MARK) {
				throw new Error("marker_not_last_tool_message");
			}
			handle.preview = payload;
			handle.root = llamaCppRoot(model.baseUrl);
			const requested = payload.id_slot;
			const candidates =
				typeof requested === "number" && Number.isInteger(requested) && requested >= 0
					? { source: "session" as const, slots: [requested] }
					: { source: "lease" as const, slots: settings.slots ?? [] };
			const record = {
				pid: process.pid,
				sessionId: handle.sessionId,
				handleId: handle.id,
				expiresAt: this.now() + (settings.ttlMs ?? DEFAULT_TTL_MS),
			};
			await this.releasing;
			for (const slot of candidates.slots) {
				if (!Number.isInteger(slot) || slot < 0) continue;
				if (await this.leases.acquire(handle.root, slot, record)) {
					handle.slot = slot;
					handle.slotSource = candidates.source;
					handle.leased = true;
					break;
				}
			}
			if (handle.slot === undefined) throw new Error("no_free_slot");
			if (!handle.accepting && handle.state !== "ended") {
				await this.release(handle);
				return;
			}
			const body: JsonObject = { messages: payload.messages, add_generation_prompt: true };
			for (const key of TEMPLATE_KEYS) if (key in payload) body[key] = payload[key];
			const rendered = await this.post(handle, "/apply-template", body);
			const prompt = isObject(rendered) ? rendered.prompt : undefined;
			if (typeof prompt !== "string") throw new Error("apply_template_unavailable");
			const at = prompt.indexOf(MARK);
			if (at < 0 || prompt.indexOf(MARK, at + 1) >= 0) throw new Error("marker_not_rendered_once");
			handle.head = prompt.slice(0, at);
			if (handle.state === "starting") handle.state = "ready";
		} catch (error) {
			this.finish(handle, "failed", error instanceof Error ? error.message : String(error));
		}
	}

	/** Start a warm for the latest appended text unless one is already in flight. */
	schedule(handle: Handle): void {
		// "ended" still warms: end() only stops new appends. The text already
		// committed is prefilled while Pi handles the tool result (~1 s gap).
		if (handle.inFlight || (handle.state !== "ready" && handle.state !== "ended") || handle.head === undefined)
			return;
		if (handle.text === handle.warmed) return;
		const text = handle.text;
		handle.inFlight = this.warm(handle, text).finally(() => {
			handle.inFlight = undefined;
			this.schedule(handle);
		});
	}

	private async warm(handle: Handle, text: string): Promise<void> {
		try {
			if (!(await this.leases.holds(handle.root!, handle.slot!, handle.id))) throw new Error("lease_lost");
			if (handle.state !== "ready" && handle.state !== "ended") return;
			const result = await this.post(handle, "/completion", {
				prompt: handle.head + text,
				n_predict: 0,
				id_slot: handle.slot,
				cache_prompt: true,
			});
			const timings = isObject(result) && isObject(result.timings) ? result.timings : undefined;
			handle.warms++;
			if (typeof timings?.prompt_n === "number") handle.warmPromptTokens += timings.prompt_n;
			if (typeof timings?.prompt_ms === "number") handle.warmPromptMs += timings.prompt_ms;
			if (text.startsWith(handle.warmed)) handle.warmed = text;
		} catch (error) {
			if (handle.state === "ready" || handle.state === "ended") {
				this.finish(handle, "failed", error instanceof Error ? error.message : String(error));
			}
		}
	}

	private async post(handle: Handle, path: string, body: JsonObject): Promise<unknown> {
		const signal = AbortSignal.any([handle.abort.signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]);
		const response = await this.fetchFn(`${handle.root}${path}`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
			signal,
		});
		if (!response.ok) throw new Error(`prefill_http_${response.status}${path}`);
		return response.json();
	}

	/**
	 * Called with the final provider payload after extension hooks. Returns the
	 * payload pinned to the warmed slot when it provably extends the warm, and
	 * the unchanged payload otherwise.
	 */
	async pin(payload: unknown, model: Model<any>): Promise<unknown> {
		const handle = this.active;
		if (!handle || handle.state === "pinned" || !isObject(payload) || !Array.isArray(payload.messages))
			return payload;
		const messages = payload.messages;
		// Requests that do not yet carry this tool's result are not the final request.
		const toolIndex = messages.findIndex(
			(message: unknown) =>
				isObject(message) && message.role === "tool" && message.tool_call_id === handle.toolCallId,
		);
		if (toolIndex < 0) return payload;
		handle.end();
		// Wait for the warm chain to catch up with the committed text: those
		// tokens would otherwise be prefilled by this request anyway.
		const deadline = Date.now() + PIN_WAIT_MS;
		while (handle.inFlight && Date.now() < deadline) {
			await Promise.race([
				handle.inFlight,
				new Promise((resolve) => setTimeout(resolve, deadline - Date.now()).unref?.()),
			]);
		}
		const reason = this.mismatch(handle, payload, model);
		if (reason) {
			this.finish(handle, "cancelled", reason);
			return payload;
		}
		if (!(await this.leases.holds(handle.root!, handle.slot!, handle.id))) {
			this.finish(handle, "cancelled", "lease_lost");
			return payload;
		}
		handle.state = "pinned";
		this.active = undefined;
		this.pinned = handle;
		return { ...payload, id_slot: handle.slot, cache_prompt: true };
	}

	private mismatch(handle: Handle, payload: JsonObject, model: Model<any>): string | undefined {
		if (handle.state !== "ended") return `state_${handle.state}`;
		const preview = handle.preview;
		if (!preview || !handle.warmed) return "nothing_warmed";
		if (handle.sessionId !== this.options.getSessionId()) return "session_changed";
		if (model.provider !== handle.model.provider || model.id !== handle.model.id) return "model_changed";
		const messages = payload.messages as unknown[];
		const expected = preview.messages as unknown[];
		if (messages.length !== expected.length) return "message_count";
		if (!sameJson(messages.slice(0, -1), expected.slice(0, -1))) return "history_changed";
		const last = messages.at(-1);
		if (!isObject(last) || last.role !== "tool" || last.tool_call_id !== handle.toolCallId) return "tool_not_last";
		if (!sameJson({ ...last, content: MARK }, expected.at(-1))) return "tool_message_shape";
		if (typeof last.content !== "string" || !last.content.startsWith(handle.warmed)) return "result_not_prefix";
		if (payload.model !== preview.model) return "payload_model";
		for (const key of TEMPLATE_KEYS) if (!sameJson(payload[key], preview[key])) return `option_${key}`;
		if (handle.slotSource === "session" && payload.id_slot !== handle.slot) return "session_slot_moved";
		return undefined;
	}

	/** Agent lifecycle: release the pinned lease after its response, drop everything at run end. */
	onAgentEvent(event: AgentEvent): void {
		if (event.type === "message_end" && event.message.role === "assistant" && this.pinned) {
			const handle = this.pinned;
			this.pinned = undefined;
			this.releasing = this.release(handle);
		} else if (event.type === "agent_end") {
			this.cancelAll("agent_end");
		}
	}

	cancelAll(reason: string): void {
		if (this.active) this.finish(this.active, "cancelled", reason);
		if (this.pinned) {
			const handle = this.pinned;
			this.pinned = undefined;
			this.releasing = this.release(handle);
		}
	}

	dispose(): void {
		this.cancelAll("disposed");
		this.disposed = true;
	}

	finish(handle: Handle, state: "cancelled" | "failed", reason: string): void {
		if (handle.state === "cancelled" || handle.state === "failed" || handle.state === "pinned") return;
		handle.state = state;
		handle.reason = reason;
		handle.abort.abort();
		if (this.active === handle) this.active = undefined;
		this.releasing = this.release(handle);
	}

	private async release(handle: Handle): Promise<void> {
		if (handle.timer) clearTimeout(handle.timer);
		if (!handle.leased) return;
		handle.leased = false;
		await this.leases.release(handle.root!, handle.slot!, handle.id).catch(() => {});
	}
}
