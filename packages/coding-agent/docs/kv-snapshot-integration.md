# KV snapshot integration

This fork supplies the Pi-side dependencies used by `pi-kv-cache-manager` with a patched llama-server. Installing the extension alone does not add these core features. Build this core and start new workers to use them; updating files does not replace the JavaScript already loaded by an old core process.

## Loaded-extension changes

`session_start.extensionsFingerprint` hashes the loaded filesystem extension set. `extensions_changed` follows `session_start` when that set differs from the previously loaded runtime in the same process. The initial event has no `previous` value. Consumers must compare the current value against their own persisted cache metadata; an initial event is not proof of a software upgrade.

At the next idle user-prompt boundary, `AgentSession` checks loaded extension source/config contents and runs the normal reload lifecycle before handling the prompt if they changed. It does not reload during streaming, compaction, running Bash or an active tool loop. Newly discovered extension entries still require startup or resource reload. Inline factories are not filesystem-watchable. Tests, logs, dependency directories, caches and virtual environments are excluded; JSON manifests/lockfiles are included.

Fingerprinting is implemented in `src/core/extensions/fingerprint.ts`. Extension events and types are documented in [Extensions](extensions.md#session_start).

A cache consumer should wait until `before_provider_request` receives the complete serialized system prompt and tools before refreshing a derived prompt cache. Earlier lifecycle events do not contain the final request prefix.

## Speculative prefill ownership

The KV extension assigns both `id_slot` and `slot_snapshot` to a request. The latter names the session snapshot that owns the live slot context. Patched llama-server can save dirty context before another owner replaces it.

When speculative prefill uses that session slot, `PrefillController` retains the preview's string-valued `slot_snapshot` on every native `/completion` warm request. Otherwise a warm for the same session would appear to be a different owner and trigger unnecessary eviction/save work. Before pinning a final request, Pi checks both the slot and snapshot owner; a changed or removed owner cancels the prefill handle with `session_snapshot_changed` rather than pinning stale state.

`slot_snapshot` is an extension/server-specific field, not a standard OpenAI API feature. It does not provide a cross-process slot lease, and restore plus generation are not atomic.

## Verification

From `packages/coding-agent`:

```sh
node ../../node_modules/vitest/dist/cli.js --run test/extensions-fingerprint.test.ts test/prefill.test.ts
```

The targeted suite uses temporary files, the faux provider and mocked prefill endpoints; no live provider credentials or llama-server are required. From the repository root, run `npm run check` for the full static/dependency/type gates.

For server commits, extension versions and the bounded four-session live-test results, see `pi-kv-cache-manager/docs/snapshot-release-20261005.md` in the companion repository. The live test used deployed binaries and is not proof of a clean-source rebuild or concurrent disk writing.
