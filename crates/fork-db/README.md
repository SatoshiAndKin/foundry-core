# foundry-fork-db

Fork database used by [Foundry](https://github.com/foundry-rs/foundry).

> **Note:** This crate was previously maintained at [foundry-rs/foundry-fork-db](https://github.com/foundry-rs/foundry-fork-db).

Provides a shared, caching database layer backed by a remote RPC provider, allowing EVM execution against forked chain state. Built on top of [revm](https://github.com/bluealloy/revm).

## Features

- `zstd`: Enables [zstd](https://github.com/gysber/zstd-rs) compression support.

## Remote account loading

For RPCs such as Tempo, where `eth_getBalance` returns a placeholder, use
`BlockchainDbMeta::with_account_fetch_policy(AccountFetchPolicy::RequireAccountInfo)`
to require authoritative account data without fallback. The policy is part of cache
identity, so incompatible caches are discarded even in offline-start mode.

## Fork cache format

The current format uses `cache_format_version = 2`. It stores each code body once
in `codes`; account records use an optional `code_ref` index. Each account keeps
its own balance, nonce, and declared code hash. Absent and empty code remain
separate, as do legacy code and EIP-7702 code with the same original bytes.

The loader accepts older inline-code caches. A successful flush writes version 2
in plain JSON or Zstd, depending on the build. Keep a separate copy of an older
cache if it is shared with binaries that require the inline layout.
