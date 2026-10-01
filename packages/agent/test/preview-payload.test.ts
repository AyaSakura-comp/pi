import { type AssistantMessage, type Model, streamSimple } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Agent, type AgentMessage } from "../src/index.ts";

const model: Model<"openai-completions"> = {
	id: "local-model",
	name: "local",
	api: "openai-completions",
	provider: "local-llama",
	baseUrl: "http://127.0.0.1:9/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 32000,
	maxTokens: 4096,
};

const toolCall: AssistantMessage = {
	role: "assistant",
	content: [{ type: "toolCall", id: "call-1", name: "research", arguments: { q: "x" } }],
	api: "openai-completions",
	provider: "local-llama",
	model: "local-model",
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

afterEach(() => {
	vi.unstubAllGlobals();
});

describe("Agent.previewProviderPayload", () => {
	it("builds the next request payload through the real provider path without sending it", async () => {
		const fetchSpy = vi.fn(() => {
			throw new Error("network must not be used");
		});
		vi.stubGlobal("fetch", fetchSpy);
		const agent = new Agent({
			initialState: { systemPrompt: "system", model, tools: [] },
			streamFn: (m, context, options) => streamSimple(m, context, { ...options, apiKey: "none", fetch: fetchSpy }),
		});
		agent.state.messages = [{ role: "user", content: "question", timestamp: 0 }, toolCall];
		const result: AgentMessage = {
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "research",
			content: [{ type: "text", text: "MARK" }],
			isError: false,
			timestamp: 0,
		};
		const { model: used, payload } = await agent.previewProviderPayload([result], {
			onPayload: (p) => ({ ...(p as object), id_slot: 3 }),
		});
		expect(used).toBe(model);
		expect(fetchSpy).not.toHaveBeenCalled();
		const body = payload as { messages: Array<{ role: string; content: unknown }>; id_slot: number };
		expect(body.id_slot).toBe(3);
		expect(body.messages.map((m) => m.role)).toEqual(["system", "user", "assistant", "tool"]);
		expect(body.messages.at(-1)?.content).toBe("MARK");
		// The transcript itself is untouched.
		expect(agent.state.messages).toHaveLength(2);
	});
});
