import { describe, expect, it } from "vitest";
import { adjustMaxTokensForThinking } from "../src/api/simple-options.ts";

describe("adjustMaxTokensForThinking", () => {
	it("uses a distinct configured xhigh budget", () => {
		const adjusted = adjustMaxTokensForThinking(undefined, 16384, "xhigh", {
			high: 4096,
			xhigh: 12000,
		});

		expect(adjusted).toEqual({ maxTokens: 16384, thinkingBudget: 12000 });
	});
});
