import { describe, expect, it } from "vitest";
import { convertResponsesMessages } from "../src/api/openai-responses-shared.ts";
import { transformMessages } from "../src/api/transform-messages.ts";
import type { AssistantMessage, Message, Model, ToolResultMessage } from "../src/types.ts";

const model: Model<"openai-codex-responses"> = {
	id: "gpt-5",
	name: "test",
	api: "openai-codex-responses",
	provider: "openai-codex",
	baseUrl: "",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 1000,
};
function call(id: string, stopReason: AssistantMessage["stopReason"]): AssistantMessage {
	return {
		role: "assistant",
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason,
		timestamp: 0,
		content: [{ type: "toolCall", id, name: "edit", arguments: {} }],
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
}
function output(id: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "edit",
		content: [{ type: "text", text: "interrupted" }],
		isError: true,
		timestamp: 0,
	};
}
const user: Message = { role: "user", content: "hello?", timestamp: 0 };

describe("interrupted tool result replay", () => {
	it.each(["aborted", "error"] as const)(
		"omits results belonging to a %s assistant without mutating history",
		(reason) => {
			const messages: Message[] = [
				call("good|fc_good", "toolUse"),
				output("good|fc_good"),
				call("bad|fc_bad", reason),
				output("bad|fc_bad"),
				user,
			];
			const before = structuredClone(messages);
			expect(transformMessages(messages, model)).toEqual([messages[0], messages[1], user]);
			expect(messages).toEqual(before);
		},
	);
	it("omits normalized results after switching models", () => {
		const messages: Message[] = [call("bad|fc_bad", "aborted"), output("bad|fc_bad"), user];
		expect(transformMessages(messages, { ...model, id: "other" }, (id) => id.replaceAll("|", "_"))).toEqual([user]);
	});
	it("does not synthesize results for aborted calls with no result", () => {
		expect(transformMessages([call("bad", "aborted"), user], model)).toEqual([user]);
	});
	it("keeps results when a valid retry reuses the call id", () => {
		const valid = call("retry", "toolUse");
		const result = output("retry");
		expect(transformMessages([call("retry", "aborted"), output("retry"), user, valid, result], model)).toEqual([
			user,
			valid,
			result,
		]);
	});
	it("produces a Codex payload with no orphan output even after repeated errors", () => {
		const messages: Message[] = [
			call("bad|fc_bad", "aborted"),
			output("bad|fc_bad"),
			user,
			{ ...call("unused", "error"), content: [] },
			user,
		];
		const payload = convertResponsesMessages(model, { messages }, new Set([model.provider]));
		expect(payload.some((item) => item.type === "function_call_output")).toBe(false);
		expect(payload).toHaveLength(2);
	});
});
