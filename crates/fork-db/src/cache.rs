//! Cache related abstraction

use alloy_chains::Chain;
use alloy_primitives::{Address, B256, U256};
use imbl::HashMap as PersistentMap;
use parking_lot::RwLock;
use revm::{
    DatabaseCommit,
    context::BlockEnv,
    primitives::{
        KECCAK_EMPTY,
        map::{AddressHashMap, HashMap},
    },
    state::{Account, AccountInfo, AccountStatus, Bytecode},
};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned, ser::SerializeMap,
};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use url::Url;

#[cfg(feature = "zstd")]
use zstd::{Encoder, stream::read::Decoder};

pub type StorageInfo = PersistentMap<U256, U256>;

/// Zstd frame magic number: `0x28B52FFD` (little-endian).
const ZSTD_FRAME_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Policy for loading remote account balances, nonces, and code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountFetchPolicy {
    /// Race `eth_getAccountInfo` against separate balance, nonce, and code requests.
    #[default]
    Auto,
    /// Require `eth_getAccountInfo`, without falling back to separate requests.
    ///
    /// Use this when `eth_getBalance` does not expose the balance used by execution,
    /// such as Tempo RPCs that return a placeholder native balance.
    RequireAccountInfo,
}

/// A shareable Block database
#[derive(Clone, Debug)]
pub struct BlockchainDb<B = BlockEnv> {
    /// Contains all the data
    db: Arc<MemDb>,
    /// metadata of the current config
    meta: Arc<RwLock<BlockchainDbMeta<B>>>,
    /// the cache that can be flushed
    cache: Arc<JsonBlockCacheDB<B>>,
}

impl<B: ForkBlockEnv> BlockchainDb<B> {
    /// Creates a new instance of the [BlockchainDb].
    ///
    /// If a `cache_path` is provided it attempts to load a previously stored [JsonBlockCacheData]
    /// and will try to use the cached entries it holds.
    ///
    /// This will return a new and empty [MemDb] if
    ///   - `cache_path` is `None`
    ///   - the file the `cache_path` points to, does not exist
    ///   - the file contains malformed data, or if it couldn't be read
    ///   - the provided `meta` differs from [BlockchainDbMeta] that's stored on disk
    pub fn new(meta: BlockchainDbMeta<B>, cache_path: Option<PathBuf>) -> Self {
        Self::new_db(meta, cache_path, false)
    }

    /// Creates a new instance of the [BlockchainDb] without comparing block metadata.
    /// Account-fetch policies must still match, preventing reuse of incompatible account data.
    /// This is useful for offline-start mode when we don't want to fetch metadata of `block`.
    ///
    /// if a `cache_path` is provided it attempts to load a previously stored [JsonBlockCacheData]
    /// and will try to use the cached entries it holds.
    ///
    /// This will return a new and empty [MemDb] if
    ///   - `cache_path` is `None`
    ///   - the file the `cache_path` points to, does not exist
    ///   - the file contains malformed data, or if it couldn't be read
    ///   - the provided `meta` differs from [BlockchainDbMeta] that's stored on disk
    pub fn new_skip_check(meta: BlockchainDbMeta<B>, cache_path: Option<PathBuf>) -> Self {
        Self::new_db(meta, cache_path, true)
    }

    fn new_db(meta: BlockchainDbMeta<B>, cache_path: Option<PathBuf>, skip_check: bool) -> Self {
        trace!(target: "forge::cache", cache=?cache_path, "initialising blockchain db");
        // read cache and check if metadata matches
        let cache = cache_path
            .as_ref()
            .and_then(|p| {
                JsonBlockCacheDB::load(p).ok().filter(|cache| {
                    // Account-loading semantics must match even when block checks are skipped.
                    if cache.meta().read().account_fetch_policy != meta.account_fetch_policy {
                        return false;
                    }
                    if skip_check {
                        return true;
                    }
                    let mut existing = cache.meta().write();
                    existing.hosts.extend(meta.hosts.clone());
                    if meta == *existing {
                        true
                    } else {
                        warn!(target: "cache", "non-matching block metadata");
                        false
                    }
                })
            })
            .unwrap_or_else(|| JsonBlockCacheDB::new(Arc::new(RwLock::new(meta)), cache_path));

        Self { db: Arc::clone(cache.db()), meta: Arc::clone(cache.meta()), cache: Arc::new(cache) }
    }

    /// Returns the map that holds the account related info
    pub fn accounts(&self) -> &RwLock<PersistentMap<Address, AccountInfo>> {
        &self.db.accounts
    }

    /// Returns the map that holds the storage related info
    pub fn storage(&self) -> &RwLock<PersistentMap<Address, StorageInfo>> {
        &self.db.storage
    }

    /// Returns the map that holds all the block hashes
    pub fn block_hashes(&self) -> &RwLock<PersistentMap<U256, B256>> {
        &self.db.block_hashes
    }

    /// Returns the Env related metadata
    pub const fn meta(&self) -> &Arc<RwLock<BlockchainDbMeta<B>>> {
        &self.meta
    }

    /// Returns the inner cache
    pub const fn cache(&self) -> &Arc<JsonBlockCacheDB<B>> {
        &self.cache
    }

    /// Returns the underlying storage
    pub const fn db(&self) -> &Arc<MemDb> {
        &self.db
    }
}

/// Marker trait for block environment types that can be used with the forking backend.
///
/// Automatically implemented for any `B: Serialize + DeserializeOwned + Default + Clone +
/// PartialEq + Send + Sync + 'static`.
pub trait ForkBlockEnv:
    Serialize + DeserializeOwned + Default + Clone + PartialEq + Send + Sync + 'static
{
}

impl<B: Serialize + DeserializeOwned + Default + Clone + PartialEq + Send + Sync + 'static>
    ForkBlockEnv for B
{
}

/// relevant identifying markers in the context of [BlockchainDb]
#[derive(Clone, Debug, Default, Eq)]
pub struct BlockchainDbMeta<B> {
    /// The chain of the blockchain of the block environment
    pub chain: Option<Chain>,
    /// The block environment
    pub block_env: B,
    /// All the hosts used to connect to
    pub hosts: BTreeSet<String>,
    /// Hash of the exact block anchoring the cached state.
    pub fork_hash: Option<B256>,
    /// Opaque identity of the RPC source and its authentication context.
    pub source_id: Option<B256>,
    /// Account-loading semantics used to populate this cache.
    pub account_fetch_policy: AccountFetchPolicy,
}

impl<B> BlockchainDbMeta<B> {
    /// Creates a new instance
    pub fn new(block_env: B, url: String) -> Self {
        let host = Url::parse(&url)
            .ok()
            .and_then(|url| url.host().map(|host| host.to_string()))
            .unwrap_or(url);

        Self {
            chain: None,
            block_env,
            hosts: BTreeSet::from([host]),
            fork_hash: None,
            source_id: None,
            account_fetch_policy: AccountFetchPolicy::Auto,
        }
    }

    /// Sets the remote account-loading policy and binds cached data to that policy.
    pub const fn with_account_fetch_policy(mut self, policy: AccountFetchPolicy) -> Self {
        self.account_fetch_policy = policy;
        self
    }

    /// Infers the host from the provided url and adds it to the set of hosts
    pub fn with_url(mut self, url: &str) -> Self {
        let host = Url::parse(url)
            .ok()
            .and_then(|url| url.host().map(|host| host.to_string()))
            .unwrap_or(url.to_string());
        self.hosts.insert(host);
        self
    }

    /// Sets the [Chain] of this instance
    pub const fn set_chain(mut self, chain: Chain) -> Self {
        self.chain = Some(chain);
        self
    }

    /// Sets the block environment of this instance
    pub fn set_block_env(mut self, block_env: B) -> Self {
        self.block_env = block_env;
        self
    }

    /// Binds cached data to an exact fork block and opaque RPC source identity.
    pub const fn with_fork_identity(mut self, fork_hash: B256, source_id: B256) -> Self {
        self.fork_hash = Some(fork_hash);
        self.source_id = Some(source_id);
        self
    }
}

// ignore hosts to not invalidate the cache when different endpoints are used, as it's commonly the
// case for http vs ws endpoints
impl<B: PartialEq> PartialEq for BlockchainDbMeta<B> {
    fn eq(&self, other: &Self) -> bool {
        self.block_env == other.block_env
            && self.fork_hash == other.fork_hash
            && self.source_id == other.source_id
            && self.account_fetch_policy == other.account_fetch_policy
    }
}

impl<B: Serialize> Serialize for BlockchainDbMeta<B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let field_count = 3
            + usize::from(self.chain.is_some())
            + usize::from(self.fork_hash.is_some())
            + usize::from(self.source_id.is_some());
        let mut s = serializer.serialize_struct("BlockchainDbMeta", field_count)?;
        if let Some(chain) = &self.chain {
            s.serialize_field("chain", chain)?;
        }
        s.serialize_field("block_env", &self.block_env)?;
        s.serialize_field("hosts", &self.hosts)?;
        s.serialize_field("account_fetch_policy", &self.account_fetch_policy)?;
        if let Some(fork_hash) = self.fork_hash {
            s.serialize_field("fork_hash", &fork_hash)?;
        }
        if let Some(source_id) = self.source_id {
            s.serialize_field("source_id", &source_id)?;
        }
        s.end()
    }
}

/// A backwards compatible representation of a block environment type `B`.
///
/// This prevents deserialization errors of cache files caused by breaking changes to the
/// block environment, for example enabling an optional feature that adds new fields.
/// By filling in missing fields from `B::default()` we can prevent cache file issues.
struct BlockEnvBackwardsCompat<B> {
    inner: B,
}

impl<'de, B: DeserializeOwned + Default + Serialize> Deserialize<'de>
    for BlockEnvBackwardsCompat<B>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = serde_json::Value::deserialize(deserializer)?;

        // we check for any missing fields here
        if let Some(obj) = value.as_object_mut() {
            let default_value = serde_json::to_value(B::default()).unwrap();
            for (key, value) in default_value.as_object().unwrap() {
                if !obj.contains_key(key) {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }

        let inner: B = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self { inner })
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Hosts {
    Multi(BTreeSet<String>),
    Single(String),
}

impl From<Hosts> for BTreeSet<String> {
    fn from(hosts: Hosts) -> Self {
        match hosts {
            Hosts::Multi(hosts) => hosts,
            Hosts::Single(host) => Self::from([host]),
        }
    }
}

impl<'de, B: DeserializeOwned + Default + Serialize> Deserialize<'de> for BlockchainDbMeta<B> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(bound = "B: DeserializeOwned + Default + Serialize")]
        struct Meta<B> {
            chain: Option<Chain>,
            block_env: BlockEnvBackwardsCompat<B>,
            #[serde(alias = "host")]
            hosts: Hosts,
            #[serde(default)]
            fork_hash: Option<B256>,
            #[serde(default)]
            source_id: Option<B256>,
            #[serde(default)]
            account_fetch_policy: AccountFetchPolicy,
        }

        let Meta { chain, block_env, hosts, fork_hash, source_id, account_fetch_policy } =
            Meta::deserialize(deserializer)?;
        Ok(Self {
            chain,
            block_env: block_env.inner,
            hosts: hosts.into(),
            fork_hash,
            source_id,
            account_fetch_policy,
        })
    }
}

/// Share immutable code across accounts without sharing their balances or nonces.
#[derive(Debug, Default)]
pub(crate) struct BytecodeInterner {
    codes: HashMap<B256, Bytecode>,
}

impl BytecodeInterner {
    pub(crate) fn intern(&mut self, code: Bytecode) -> Bytecode {
        self.codes.entry(code.hash_slow()).or_insert(code).clone()
    }

    pub(crate) fn share_account_code(&mut self, account: &mut AccountInfo) {
        if let Some(code) = account.code.take() {
            account.code = Some(self.intern(code));
        }
    }
}

/// In Memory cache containing all fetched accounts and storage slots
/// and their values from RPC.
///
/// Persistent maps share unchanged entries with historical snapshots. Mutations preserve
/// the values retained by earlier snapshots.
#[derive(Debug, Default)]
pub struct MemDb {
    /// Account related data
    pub accounts: RwLock<PersistentMap<Address, AccountInfo>>,
    /// Storage related data
    pub storage: RwLock<PersistentMap<Address, StorageInfo>>,
    /// All retrieved block hashes
    pub block_hashes: RwLock<PersistentMap<U256, B256>>,
}

impl MemDb {
    /// Clears all data stored in this db
    pub fn clear(&self) {
        self.accounts.write().clear();
        self.storage.write().clear();
        self.block_hashes.write().clear();
    }

    // Inserts the account, replacing it if it exists already
    pub fn do_insert_account(&self, address: Address, account: AccountInfo) {
        self.accounts.write().insert(address, account);
    }

    /// The implementation of [DatabaseCommit::commit()]
    pub fn do_commit(&self, changes: AddressHashMap<Account>) {
        let mut storage = self.storage.write();
        let mut accounts = self.accounts.write();
        for (add, mut acc) in changes {
            if acc.is_empty() || acc.is_selfdestructed() {
                accounts.remove(&add);
                storage.remove(&add);
            } else {
                // insert account
                if let Some(code_hash) = acc
                    .info
                    .code
                    .as_ref()
                    .filter(|code| !code.is_empty())
                    .map(|code| code.hash_slow())
                {
                    acc.info.code_hash = code_hash;
                } else if acc.info.code_hash.is_zero() {
                    acc.info.code_hash = KECCAK_EMPTY;
                }
                accounts.insert(add, acc.info);

                let acc_storage = storage.entry(add).or_default();
                if acc.status.contains(AccountStatus::Created) {
                    acc_storage.clear();
                }
                for (index, value) in acc.storage {
                    if value.present_value().is_zero() {
                        acc_storage.remove(&index);
                    } else {
                        acc_storage.insert(index, value.present_value());
                    }
                }
                if acc_storage.is_empty() {
                    storage.remove(&add);
                }
            }
        }
    }
}

impl Clone for MemDb {
    fn clone(&self) -> Self {
        Self {
            storage: RwLock::new(self.storage.read().clone()),
            accounts: RwLock::new(self.accounts.read().clone()),
            block_hashes: RwLock::new(self.block_hashes.read().clone()),
        }
    }
}

impl DatabaseCommit for MemDb {
    fn commit(&mut self, changes: AddressHashMap<Account>) {
        self.do_commit(changes)
    }
}

/// A DB that stores the cached content in a json file
#[derive(Debug)]
pub struct JsonBlockCacheDB<B> {
    /// Where this cache file is stored.
    ///
    /// If this is a [None] then caching is disabled
    cache_path: Option<PathBuf>,
    /// Object that's stored in a json file
    data: JsonBlockCacheData<B>,
    /// Number of database requests that were served from the cache.
    cache_hits: AtomicU64,
    /// Number of cache lookups that scheduled a provider request.
    cache_misses: AtomicU64,
}

impl<B> JsonBlockCacheDB<B> {
    /// Creates a new instance.
    fn new(meta: Arc<RwLock<BlockchainDbMeta<B>>>, cache_path: Option<PathBuf>) -> Self {
        Self {
            cache_path,
            data: JsonBlockCacheData { meta, data: Arc::new(Default::default()) },
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Returns the [MemDb] it holds access to
    pub const fn db(&self) -> &Arc<MemDb> {
        &self.data.data
    }

    /// Metadata stored alongside the data
    pub const fn meta(&self) -> &Arc<RwLock<BlockchainDbMeta<B>>> {
        &self.data.meta
    }

    /// Returns `true` if this is a transient cache and nothing will be flushed
    pub const fn is_transient(&self) -> bool {
        self.cache_path.is_none()
    }

    /// Returns the cache path.
    pub fn cache_path(&self) -> Option<&Path> {
        self.cache_path.as_deref()
    }

    pub(crate) fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    pub(crate) fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }
}

impl<B: ForkBlockEnv> JsonBlockCacheDB<B> {
    /// Loads the contents of the diskmap file and returns the read object.
    ///
    /// When the `zstd` feature is enabled, this will transparently detect and decompress
    /// zstd-compressed cache files (by checking for the zstd magic bytes `0x28B52FFD`).
    /// Plain JSON files are always supported regardless of feature flags.
    ///
    /// # Errors
    /// This will fail if
    ///   - the `path` does not exist
    ///   - the format does not match [JsonBlockCacheData]
    pub fn load(path: impl Into<PathBuf>) -> eyre::Result<Self> {
        let path = path.into();
        trace!(target: "cache", ?path, "reading json cache");
        let file = fs::File::open(&path).inspect_err(|err| {
            warn!(?err, ?path, "Failed to read cache file");
        })?;
        let data = read_cache_data(file, &path).inspect_err(|err| {
            warn!(target: "cache", ?err, ?path, "Failed to deserialize cache data");
        })?;
        trace!(target: "cache", ?path, "read json cache");
        Ok(Self {
            cache_path: Some(path),
            data,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        })
    }
}

/// Detect compression from a bounded prefix, then deserialize without retaining the input.
fn read_cache_data<B: ForkBlockEnv>(
    mut reader: impl Read,
    path: &Path,
) -> eyre::Result<JsonBlockCacheData<B>> {
    let mut prefix = Vec::with_capacity(ZSTD_FRAME_MAGIC.len());
    reader.by_ref().take(ZSTD_FRAME_MAGIC.len() as u64).read_to_end(&mut prefix)?;
    let compressed = prefix.starts_with(&ZSTD_FRAME_MAGIC);
    let reader = Cursor::new(prefix).chain(reader);
    if !compressed {
        return Ok(serde_json::from_reader(BufReader::new(reader))?);
    }

    #[cfg(feature = "zstd")]
    {
        trace!(target: "cache", ?path, "decompressing zstd cache");
        Ok(serde_json::from_reader(BufReader::new(Decoder::new(reader)?))?)
    }

    #[cfg(not(feature = "zstd"))]
    {
        eyre::bail!(
            "cache file at `{}` is zstd-compressed but the `zstd` feature is not enabled",
            path.display()
        );
    }
}

impl<B: Serialize + Clone> JsonBlockCacheDB<B> {
    /// Flushes the DB to disk if caching is enabled.
    #[instrument(level = "warn", skip_all, fields(path = ?self.cache_path))]
    pub fn flush(&self) {
        let Some(path) = &self.cache_path else { return };
        self.flush_to(path.as_path());
    }

    /// Flushes the DB to a specific file.
    ///
    /// When the `zstd` feature is enabled, the cache is written as zstd-compressed JSON.
    /// Otherwise, plain JSON is written.
    pub fn flush_to(&self, cache_path: &Path) {
        let path: &Path = cache_path;

        trace!(target: "cache", "saving json cache");

        if let Err(e) = self.write_cache_atomically(path) {
            return warn!(target: "cache", %e, "Failed to publish json cache");
        }

        trace!(target: "cache", "saved json cache");
    }

    /// Publish a complete cache without truncating the previous generation on failure.
    fn write_cache_atomically(&self, path: &Path) -> eyre::Result<()> {
        let parent =
            path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;

        #[cfg(feature = "zstd")]
        {
            // Finish the frame before publishing, so interrupted writes keep the old cache.
            let mut encoder = Encoder::new(file.as_file_mut(), 3)?;
            // Overlap bounded compression work with JSON serialization.
            encoder.multithread(2)?;
            write_cache_json(&mut encoder, &self.data)?;
            encoder.finish()?;
        }

        #[cfg(not(feature = "zstd"))]
        {
            write_cache_json(file.as_file_mut(), &self.data)?;
        }

        file.as_file().sync_all()?;
        file.persist(path)?;
        Ok(())
    }
}

fn write_cache_json(writer: impl Write, data: &impl Serialize) -> eyre::Result<()> {
    // Hex formatting emits tiny fragments; batch them before compression or file I/O.
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer(&mut writer, data)?;
    writer.flush()?;
    Ok(())
}

/// The Data the [JsonBlockCacheDB] can read and flush
///
/// This will be deserialized in a JSON object with the keys:
/// `["cache_format_version", "meta", "codes", "accounts", "storage", "block_hashes"]`
#[derive(Debug)]
pub struct JsonBlockCacheData<B> {
    pub meta: Arc<RwLock<BlockchainDbMeta<B>>>,
    pub data: Arc<MemDb>,
}

/// Version 2 stores code once and preserves exact block hashes and account state.
const CACHE_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedAccount {
    balance: U256,
    nonce: u64,
    code_hash: B256,
    code_ref: Option<usize>,
}

struct AccountsWithCodeRefs(Vec<(Address, CachedAccount)>);

impl Serialize for AccountsWithCodeRefs {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (address, account) in &self.0 {
            map.serialize_entry(address, account)?;
        }
        map.end()
    }
}

impl<B: Serialize + Clone> Serialize for JsonBlockCacheData<B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Both tables must describe one account snapshot, including during concurrent inserts.
        let accounts = self.data.accounts.read().clone();
        let meta = self.meta.read().clone();
        let storage = self.data.storage.read().clone();
        let block_hashes = self.data.block_hashes.read().clone();
        let mut ordered_accounts = accounts.iter().collect::<Vec<_>>();
        ordered_accounts.sort_unstable_by_key(|(address, _)| **address);
        let mut codes = Vec::<&Bytecode>::new();
        let mut code_indices = HashMap::<_, usize>::default();
        let mut records = Vec::with_capacity(ordered_accounts.len());
        for (address, account) in ordered_accounts {
            let code_ref = if let Some(code) = &account.code {
                let key = (code.kind(), code.hash_slow());
                let index = *code_indices.entry(key).or_insert_with(|| {
                    let index = codes.len();
                    codes.push(code);
                    index
                });
                if codes[index].original_byte_slice() != code.original_byte_slice() {
                    return Err(serde::ser::Error::custom("conflicting code table identity"));
                }
                Some(index)
            } else {
                None
            };
            records.push((
                *address,
                CachedAccount {
                    balance: account.balance,
                    nonce: account.nonce,
                    code_hash: account.code_hash,
                    code_ref,
                },
            ));
        }
        let mut map = serializer.serialize_map(Some(6))?;
        map.serialize_entry("cache_format_version", &CACHE_FORMAT_VERSION)?;
        map.serialize_entry("meta", &meta)?;
        map.serialize_entry("codes", &codes)?;
        map.serialize_entry("accounts", &AccountsWithCodeRefs(records))?;
        map.serialize_entry("storage", &storage)?;
        map.serialize_entry("block_hashes", &block_hashes)?;
        map.end()
    }
}

impl<'de, B: DeserializeOwned + Default + Serialize> Deserialize<'de> for JsonBlockCacheData<B> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Data<B> {
            cache_format_version: u32,
            meta: B,
            codes: Vec<Bytecode>,
            accounts: PersistentMap<Address, CachedAccount>,
            storage: PersistentMap<Address, StorageInfo>,
            block_hashes: PersistentMap<U256, B256>,
        }
        let Data { cache_format_version, meta, codes, accounts, storage, block_hashes } =
            Data::<BlockchainDbMeta<B>>::deserialize(deserializer)?;
        if cache_format_version != CACHE_FORMAT_VERSION {
            return Err(serde::de::Error::custom("unsupported fork cache format version"));
        }
        let mut recovered = PersistentMap::new();
        for (address, account) in accounts {
            let code = account
                .code_ref
                .map(|index| {
                    codes.get(index).cloned().ok_or_else(|| {
                        <D::Error as serde::de::Error>::custom(format!(
                            "invalid fork cache code reference {index}"
                        ))
                    })
                })
                .transpose()?;
            recovered.insert(
                address,
                AccountInfo {
                    balance: account.balance,
                    nonce: account.nonce,
                    code_hash: account.code_hash,
                    code,
                    ..Default::default()
                },
            );
        }
        Ok(Self {
            meta: Arc::new(RwLock::new(meta)),
            data: Arc::new(MemDb {
                accounts: RwLock::new(recovered),
                storage: RwLock::new(storage),
                block_hashes: RwLock::new(block_hashes),
            }),
        })
    }
}

/// A type that flushes a `JsonBlockCacheDB` on drop
///
/// This type intentionally does not implement `Clone` since it's intended that there's only once
/// instance that will flush the cache.
#[derive(Debug)]
pub struct FlushJsonBlockCacheDB<B: Serialize + Clone>(pub Arc<JsonBlockCacheDB<B>>);

impl<B: Serialize + Clone> Drop for FlushJsonBlockCacheDB<B> {
    fn drop(&mut self) {
        trace!(target: "fork::cache", "flushing cache");
        self.0.flush();
        trace!(target: "fork::cache", "flushed cache");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Bytes, keccak256};

    #[derive(Clone)]
    struct InterruptedSerialization;

    impl Serialize for InterruptedSerialization {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("written_before_failure", &"partial cache")?;
            Err(serde::ser::Error::custom("injected serialization failure"))
        }
    }

    #[test]
    fn failed_flush_preserves_published_cache() {
        for existing_cache in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("cache.json");
            let previous = existing_cache.then(|| {
                let db = JsonBlockCacheDB::new(
                    Arc::new(RwLock::new(BlockchainDbMeta::new(
                        BlockEnv::default(),
                        String::new(),
                    ))),
                    Some(path.clone()),
                );
                db.flush();
                fs::read(&path).unwrap()
            });
            let failing = JsonBlockCacheDB::new(
                Arc::new(RwLock::new(BlockchainDbMeta::new(
                    InterruptedSerialization,
                    String::new(),
                ))),
                Some(path.clone()),
            );
            failing.flush();
            assert_eq!(fs::read(&path).ok(), previous);
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), usize::from(existing_cache));
            if existing_cache {
                JsonBlockCacheDB::<BlockEnv>::load(path).unwrap();
            }
        }
    }

    #[test]
    fn can_deserialize_cache() {
        let s = r#"{
    "cache_format_version": 2,
    "codes": [
        {
            "LegacyAnalyzed": {
                "bytecode": "0x00",
                "original_len": 0,
                "jump_table": {
                    "order": "bitvec::order::Lsb0",
                    "head": {
                        "width": 8,
                        "index": 0
                    },
                    "bits": 1,
                    "data": [
                        0
                    ]
                }
            }
        }
    ],
    "meta": {
        "cfg_env": {
            "chain_id": 1337,
            "perf_analyse_created_bytecodes": "Analyse",
            "limit_contract_code_size": 18446744073709551615,
            "memory_limit": 4294967295,
            "disable_block_gas_limit": false,
            "disable_eip3607": false,
            "disable_base_fee": false
        },
        "block_env": {
            "number": 15547871,
            "coinbase": "0x0000000000000000000000000000000000000000",
            "timestamp": 1663351871,
            "difficulty": "0x0",
            "basefee": 12448539171,
            "gas_limit": 30000000,
            "prevrandao": "0x0000000000000000000000000000000000000000000000000000000000000000"
        },
        "hosts": [
            "eth-mainnet.alchemyapi.io"
        ]
    },
    "accounts": {
        "0xb8ffc3cd6e7cf5a098a1c92f48009765b24088dc": {
            "balance": "0x0",
            "nonce": 10,
            "code_hash": "0x3ac64c95eedf82e5d821696a12daac0e1b22c8ee18a9fd688b00cfaf14550aad",
            "code_ref": 0
        }
    },
    "storage": {
        "0xa354f35829ae975e850e23e9615b11da1b3dc4de": {
            "0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e564": "0x5553444320795661756c74000000000000000000000000000000000000000000",
            "0x10": "0x37fd60ff8346",
            "0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563": "0xb",
            "0x6": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "0x5": "0x36ff5b93162e",
            "0x14": "0x29d635a8e000",
            "0x11": "0x63224c73",
            "0x2": "0x6"
        }
    },
    "block_hashes": {}
}"#;

        let cache: JsonBlockCacheData<BlockEnv> = serde_json::from_str(s).unwrap();
        assert_eq!(cache.data.accounts.read().len(), 1);
        assert_eq!(cache.data.storage.read().len(), 1);
        assert!(cache.data.block_hashes.read().is_empty());

        let _s = serde_json::to_string(&cache).unwrap();
    }

    #[test]
    fn can_deserialize_cache_post_4844() {
        let s = r#"{
    "cache_format_version": 2,
    "codes": [
        {
            "LegacyAnalyzed": {
                "bytecode": "0x00",
                "original_len": 0,
                "jump_table": {
                    "order": "bitvec::order::Lsb0",
                    "head": {
                        "width": 8,
                        "index": 0
                    },
                    "bits": 1,
                    "data": [
                        0
                    ]
                }
            }
        }
    ],
    "meta": {
        "cfg_env": {
            "chain_id": 1,
            "kzg_settings": "Default",
            "perf_analyse_created_bytecodes": "Analyse",
            "limit_contract_code_size": 18446744073709551615,
            "memory_limit": 134217728,
            "disable_block_gas_limit": false,
            "disable_eip3607": true,
            "disable_base_fee": false,
            "optimism": false
        },
        "block_env": {
            "number": 18651580,
            "coinbase": "0x4838b106fce9647bdf1e7877bf73ce8b0bad5f97",
            "timestamp": 1700950019,
            "gas_limit": 30000000,
            "basefee": 26886078239,
            "difficulty": "0xc6b1a299886016dea3865689f8393b9bf4d8f4fe8c0ad25f0058b3569297c057",
            "prevrandao": "0xc6b1a299886016dea3865689f8393b9bf4d8f4fe8c0ad25f0058b3569297c057",
            "blob_excess_gas_and_price": {
                "excess_blob_gas": 0,
                "blob_gasprice": 1
            }
        },
        "hosts": [
            "eth-mainnet.alchemyapi.io"
        ]
    },
    "accounts": {
        "0x4838b106fce9647bdf1e7877bf73ce8b0bad5f97": {
            "balance": "0x8e0c373cfcdfd0eb",
            "nonce": 128912,
            "code_hash": "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
            "code_ref": 0
        }
    },
    "storage": {},
    "block_hashes": {}
}"#;

        let cache: JsonBlockCacheData<BlockEnv> = serde_json::from_str(s).unwrap();
        assert_eq!(cache.data.accounts.read().len(), 1);

        let _s = serde_json::to_string(&cache).unwrap();
    }

    #[test]
    fn account_fetch_policy_invalidates_incompatible_disk_caches() {
        for skip_check in [false, true] {
            for (stored, requested) in [
                (AccountFetchPolicy::Auto, AccountFetchPolicy::RequireAccountInfo),
                (AccountFetchPolicy::RequireAccountInfo, AccountFetchPolicy::Auto),
            ] {
                let temp = tempfile::tempdir().unwrap();
                let path = temp.path().join("cache.json");
                let meta = BlockchainDbMeta::new(BlockEnv::default(), "http://localhost".into())
                    .with_account_fetch_policy(stored);
                let db = BlockchainDb::new(meta.clone(), Some(path.clone()));
                let address = Address::with_last_byte(1);
                db.accounts().write().insert(address, AccountInfo::from_balance(U256::from(42)));
                db.cache().flush();

                // Matching semantics preserve the actual balance, including nonzero values.
                let compatible = BlockchainDb::new_db(meta.clone(), Some(path.clone()), skip_check);
                assert_eq!(compatible.accounts().read()[&address].balance, U256::from(42));
                let incompatible = BlockchainDb::new_db(
                    meta.with_account_fetch_policy(requested),
                    Some(path),
                    skip_check,
                );
                assert!(incompatible.accounts().read().is_empty());
                assert_eq!(incompatible.meta().read().account_fetch_policy, requested);
            }
        }
    }

    #[test]
    fn legacy_account_cache_defaults_to_auto_policy() {
        let mut value = serde_json::to_value(BlockchainDbMeta::<BlockEnv>::default()).unwrap();
        value.as_object_mut().unwrap().remove("account_fetch_policy");
        let legacy = serde_json::from_value::<BlockchainDbMeta<BlockEnv>>(value).unwrap();
        assert_eq!(legacy.account_fetch_policy, AccountFetchPolicy::Auto);
        assert_ne!(
            legacy,
            legacy.clone().with_account_fetch_policy(AccountFetchPolicy::RequireAccountInfo)
        );
    }

    #[test]
    fn roundtrip_meta_block_env() {
        let meta = BlockchainDbMeta {
            chain: Some(Chain::mainnet()),
            block_env: BlockEnv { number: U256::from(1u64), ..Default::default() },
            hosts: BTreeSet::from(["eth-mainnet.alchemyapi.io".to_string()]),
            fork_hash: Some(B256::with_last_byte(1)),
            source_id: Some(B256::with_last_byte(2)),
            account_fetch_policy: AccountFetchPolicy::RequireAccountInfo,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let recovered: BlockchainDbMeta<BlockEnv> = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, recovered);
    }

    #[test]
    fn exact_fork_identity_invalidates_cache_metadata() {
        let meta = BlockchainDbMeta::<BlockEnv>::default()
            .with_fork_identity(B256::with_last_byte(1), B256::with_last_byte(2));
        assert_ne!(
            meta,
            BlockchainDbMeta::<BlockEnv>::default()
                .with_fork_identity(B256::with_last_byte(3), B256::with_last_byte(2))
        );
        assert_ne!(
            meta,
            BlockchainDbMeta::<BlockEnv>::default()
                .with_fork_identity(B256::with_last_byte(1), B256::with_last_byte(4))
        );
    }

    #[test]
    fn can_return_cache_path_if_set() {
        // set
        let cache_db = JsonBlockCacheDB::<BlockEnv>::new(
            Arc::new(RwLock::new(BlockchainDbMeta::default())),
            Some(PathBuf::from("/tmp/foo")),
        );
        assert_eq!(Some(Path::new("/tmp/foo")), cache_db.cache_path());

        // unset
        let cache_db = JsonBlockCacheDB::<BlockEnv>::new(
            Arc::new(RwLock::new(BlockchainDbMeta::default())),
            None,
        );
        assert_eq!(None, cache_db.cache_path());
    }

    #[test]
    fn is_zstd_detection() {
        // zstd magic bytes
        assert!([0x28, 0xB5, 0x2F, 0xFD, 0x00].starts_with(&ZSTD_FRAME_MAGIC));
        // plain JSON
        assert!(!b"{\"meta\":{}}".starts_with(&ZSTD_FRAME_MAGIC));
        // too short
        assert!(![0x28, 0xB5].starts_with(&ZSTD_FRAME_MAGIC));
        // empty
        assert!(![].starts_with(&ZSTD_FRAME_MAGIC));
    }

    #[test]
    fn cache_load_shares_equal_code_without_sharing_account_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared-code.json");
        let db = BlockchainDb::new(BlockchainDbMeta::<BlockEnv>::default(), Some(path.clone()));
        let first = Address::from_word(keccak256("first shared-code account"));
        let second = Address::from_word(keccak256("second shared-code account"));
        let distinct = Address::from_word(keccak256("different-code account"));
        for (address, balance, nonce, immediate) in
            [(first, 11u64, 7u64, 1u8), (second, 17, 13, 1), (distinct, 23, 19, 2)]
        {
            // Start with independent allocations, as separate RPC reads do.
            let code = Bytecode::new_raw(Bytes::from(vec![0x60, immediate, 0x00]));
            db.accounts().write().insert(
                address,
                AccountInfo {
                    balance: U256::from(balance),
                    nonce,
                    code_hash: code.hash_slow(),
                    code: Some(code),
                    ..Default::default()
                },
            );
        }
        db.cache().flush();

        let loaded = JsonBlockCacheDB::<BlockEnv>::load(path).unwrap();
        let accounts = loaded.db().accounts.read();
        for (address, balance, nonce, immediate) in
            [(first, 11u64, 7u64, 1u8), (second, 17, 13, 1), (distinct, 23, 19, 2)]
        {
            let account = &accounts[&address];
            assert_eq!(account.balance, U256::from(balance));
            assert_eq!(account.nonce, nonce);
            assert_eq!(
                account.code.as_ref().unwrap().original_byte_slice(),
                &[0x60, immediate, 0x00]
            );
            assert_eq!(account.code_hash, keccak256([0x60, immediate, 0x00]));
        }
        let first_code = accounts[&first].code.as_ref().unwrap().original_byte_slice();
        let second_code = accounts[&second].code.as_ref().unwrap().original_byte_slice();
        let distinct_code = accounts[&distinct].code.as_ref().unwrap().original_byte_slice();
        assert!(!std::ptr::eq(first_code.as_ptr(), distinct_code.as_ptr()));
        assert!(
            std::ptr::eq(first_code.as_ptr(), second_code.as_ptr()),
            "equal cached bytecode must reuse its backing allocation"
        );
    }

    #[test]
    fn flush_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");

        let meta = BlockchainDbMeta::<BlockEnv>::default();
        let db = BlockchainDb::new(meta, Some(path.clone()));

        // Insert some data
        db.accounts().write().insert(
            Address::ZERO,
            AccountInfo { balance: U256::from(42u64), nonce: 1, ..Default::default() },
        );
        db.block_hashes().write().insert(U256::from(1), B256::with_last_byte(1));

        db.cache().flush();
        assert!(path.exists());

        let raw = fs::read(&path).unwrap();
        #[cfg(feature = "zstd")]
        assert!(
            raw.starts_with(&ZSTD_FRAME_MAGIC),
            "should be zstd-compressed with the zstd feature"
        );
        #[cfg(not(feature = "zstd"))]
        assert!(!raw.starts_with(&ZSTD_FRAME_MAGIC), "should be plain JSON without zstd feature");

        // Reload
        let loaded = JsonBlockCacheDB::<BlockEnv>::load(&path).unwrap();
        assert_eq!(loaded.db().accounts.read().len(), 1);
        assert_eq!(loaded.db().accounts.read()[&Address::ZERO].balance, U256::from(42u64));
        assert_eq!(loaded.db().block_hashes.read().len(), 1);
    }

    #[cfg(not(feature = "zstd"))]
    #[test]
    fn load_errors_on_zstd_data_without_feature() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json.zst");

        fs::write(&path, [0x28, 0xB5, 0x2F, 0xFD, 0x00]).unwrap();

        let err = JsonBlockCacheDB::<BlockEnv>::load(&path).unwrap_err();
        assert!(err.to_string().contains("zstd-compressed but the `zstd` feature is not enabled"));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_flush_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json.zst");

        let meta = BlockchainDbMeta::<BlockEnv>::default();
        let db = BlockchainDb::new(meta, Some(path.clone()));

        db.accounts().write().insert(
            Address::ZERO,
            AccountInfo { balance: U256::from(100u64), nonce: 5, ..Default::default() },
        );
        db.storage().write().insert(Address::ZERO, Default::default());

        db.cache().flush();
        assert!(path.exists());

        // Verify on-disk bytes have zstd magic
        let raw = fs::read(&path).unwrap();
        assert!(
            raw.starts_with(&ZSTD_FRAME_MAGIC),
            "should be zstd-compressed with the zstd feature"
        );

        // Reload and verify data integrity
        let loaded = JsonBlockCacheDB::<BlockEnv>::load(&path).unwrap();
        assert_eq!(loaded.db().accounts.read().len(), 1);
        assert_eq!(loaded.db().accounts.read()[&Address::ZERO].nonce, 5);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_can_load_plain_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");

        // Write plain JSON manually
        let meta = BlockchainDbMeta::<BlockEnv>::default();
        let data = JsonBlockCacheData::<BlockEnv> {
            meta: Arc::new(RwLock::new(meta)),
            data: Arc::new(MemDb::default()),
        };
        let json = serde_json::to_vec(&data).unwrap();
        fs::write(&path, &json).unwrap();

        // Should load fine even with zstd feature enabled
        let loaded = JsonBlockCacheDB::<BlockEnv>::load(&path).unwrap();
        assert_eq!(loaded.db().accounts.read().len(), 0);
    }

    struct SmallReads {
        data: Vec<u8>,
        position: usize,
    }

    impl std::io::Read for SmallReads {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            let count = output.len().min(3).min(self.data.len() - self.position);
            output[..count].copy_from_slice(&self.data[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }
    }

    #[test]
    fn stream_load_handles_short_plain_reads() {
        let cache = JsonBlockCacheData::<BlockEnv> {
            meta: Arc::new(RwLock::new(BlockchainDbMeta::default())),
            data: Arc::new(MemDb::default()),
        };
        let json = serde_json::to_vec(&cache).unwrap();
        let mut reader = SmallReads { data: json, position: 0 };
        let loaded: JsonBlockCacheData<BlockEnv> =
            read_cache_data(&mut reader, Path::new("short-reads.json")).unwrap();
        assert_eq!(*loaded.meta.read(), *cache.meta.read());
        assert!(loaded.data.accounts.read().is_empty());
        assert_eq!(reader.position, reader.data.len());
    }

    #[test]
    fn stream_load_rejects_invalid_json_without_buffering_entire_input() {
        let mut reader = SmallReads { data: vec![b'!'; 1024 * 1024], position: 0 };
        let result: eyre::Result<JsonBlockCacheData<BlockEnv>> =
            read_cache_data(&mut reader, Path::new("invalid.json"));
        assert!(result.is_err());
        assert!(
            reader.position < reader.data.len(),
            "invalid input must fail before full buffering"
        );
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn stream_load_handles_short_compressed_reads_and_rejects_truncation() {
        let cache = JsonBlockCacheData::<BlockEnv> {
            meta: Arc::new(RwLock::new(BlockchainDbMeta::default())),
            data: Arc::new(MemDb::default()),
        };
        let json = serde_json::to_vec(&cache).unwrap();
        let compressed = zstd::encode_all(json.as_slice(), 3).unwrap();
        let mut reader = SmallReads { data: compressed.clone(), position: 0 };
        let loaded: JsonBlockCacheData<BlockEnv> =
            read_cache_data(&mut reader, Path::new("short-reads.json.zst")).unwrap();
        assert_eq!(*loaded.meta.read(), *cache.meta.read());
        assert!(loaded.data.accounts.read().is_empty());
        assert_eq!(reader.position, reader.data.len());

        for length in [compressed.len() / 2, compressed.len() - 1] {
            let mut reader = SmallReads { data: compressed[..length].to_vec(), position: 0 };
            let result: eyre::Result<JsonBlockCacheData<BlockEnv>> =
                read_cache_data(&mut reader, Path::new("truncated.json.zst"));
            assert!(result.is_err(), "a truncated compressed frame must fail at length {length}");
        }
    }

    #[derive(Default)]
    struct CacheJsonWriter {
        bytes: Vec<u8>,
        writes: usize,
        flushes: usize,
        fail_flush: bool,
    }

    impl Write for CacheJsonWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if !bytes.is_empty() {
                self.writes += 1;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flushes += 1;
            if self.fail_flush {
                Err(std::io::Error::other("injected cache flush failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn cache_json_batches_small_writes_without_changing_bytes() {
        let data = JsonBlockCacheData {
            meta: Arc::new(RwLock::new(BlockchainDbMeta::<BlockEnv>::default())),
            data: Arc::new(MemDb::default()),
        };
        for index in 0..128u64 {
            data.data.accounts.write().insert(
                Address::from_word(keccak256(index.to_be_bytes())),
                AccountInfo::from_balance(U256::from(index)),
            );
        }
        let expected = serde_json::to_vec(&data).unwrap();
        let mut writer = CacheJsonWriter::default();
        write_cache_json(&mut writer, &data).unwrap();
        assert_eq!(writer.bytes, expected);
        assert!(
            writer.writes <= expected.len().div_ceil(1024),
            "{} writes for {} bytes must reach the compressor in batches",
            writer.writes,
            expected.len()
        );
        assert_eq!(writer.flushes, 1);
    }

    #[test]
    fn cache_json_propagates_flush_failure() {
        let data = JsonBlockCacheData {
            meta: Arc::new(RwLock::new(BlockchainDbMeta::<BlockEnv>::default())),
            data: Arc::new(MemDb::default()),
        };
        let mut writer = CacheJsonWriter { fail_flush: true, ..Default::default() };
        let error = write_cache_json(&mut writer, &data).unwrap_err();
        assert_eq!(error.to_string(), "injected cache flush failure");
    }

    #[test]
    fn large_cache_flush_finishes_frame_before_publication() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large-cache.json");
        let db = BlockchainDb::new(BlockchainDbMeta::<BlockEnv>::default(), Some(path.clone()));
        let code = Bytecode::new_raw(Bytes::from(vec![0x5b; 24 * 1024]));
        for index in 0..192u64 {
            db.accounts().write().insert(
                Address::from_word(keccak256(format!("large-cache-account-{index}"))),
                AccountInfo {
                    balance: U256::MAX - U256::from(index),
                    nonce: index,
                    code_hash: code.hash_slow(),
                    code: Some(code.clone()),
                    ..Default::default()
                },
            );
        }
        let expected = serde_json::to_value(&db.cache().data).unwrap();
        db.cache().write_cache_atomically(&path).unwrap();
        let loaded = JsonBlockCacheDB::<BlockEnv>::load(&path).unwrap();
        assert_eq!(serde_json::to_value(&loaded.data).unwrap(), expected);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[derive(Clone)]
    struct InterruptedLargeSerialization;

    impl Serialize for InterruptedLargeSerialization {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("written_before_failure", &"x".repeat(8 * 1024 * 1024))?;
            Err(serde::ser::Error::custom("injected large serialization failure"))
        }
    }

    #[test]
    fn failed_large_flush_preserves_published_cache() {
        for existing_cache in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("cache.json");
            let previous = existing_cache.then(|| {
                let db = JsonBlockCacheDB::new(
                    Arc::new(RwLock::new(BlockchainDbMeta::new(
                        BlockEnv::default(),
                        String::new(),
                    ))),
                    Some(path.clone()),
                );
                db.write_cache_atomically(&path).unwrap();
                fs::read(&path).unwrap()
            });
            let failing = JsonBlockCacheDB::new(
                Arc::new(RwLock::new(BlockchainDbMeta::new(
                    InterruptedLargeSerialization,
                    String::new(),
                ))),
                Some(path.clone()),
            );
            let error = failing.write_cache_atomically(&path).unwrap_err();
            assert_eq!(error.to_string(), "injected large serialization failure");
            assert_eq!(fs::read(&path).ok(), previous);
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), usize::from(existing_cache));
            if existing_cache {
                JsonBlockCacheDB::<BlockEnv>::load(path).unwrap();
            }
        }
    }

    fn shared_code_fixture() -> JsonBlockCacheData<BlockEnv> {
        let cache = JsonBlockCacheData {
            meta: Arc::new(RwLock::new(
                BlockchainDbMeta::<BlockEnv>::default().with_fork_identity(
                    keccak256("shared-code-block"),
                    keccak256("shared-code-source"),
                ),
            )),
            data: Arc::new(MemDb::default()),
        };
        let delegated =
            Bytecode::new_eip7702(Address::from_word(keccak256("delegated-code-target")));
        let legacy = Bytecode::new_legacy(delegated.original_bytes());
        for (index, code) in
            [Some(legacy.clone()), Some(legacy), Some(delegated), None, Some(Bytecode::default())]
                .into_iter()
                .enumerate()
        {
            let address = Address::from_word(keccak256(format!("shared-code-fixture-{index}")));
            cache.data.accounts.write().insert(
                address,
                AccountInfo {
                    balance: U256::MAX - U256::from(index),
                    nonce: u64::MAX - index as u64,
                    // Preserve the declared hash independently from actual code identity.
                    code_hash: keccak256(format!("declared-code-hash-{index}")),
                    code,
                    ..Default::default()
                },
            );
            cache
                .data
                .storage
                .write()
                .insert(address, PersistentMap::from(vec![(U256::MAX, U256::from(index))]));
        }
        cache.data.block_hashes.write().insert(U256::MAX, keccak256("shared-code-block-hash"));
        cache
    }

    fn assert_shared_code_state(
        expected: &JsonBlockCacheData<BlockEnv>,
        actual: &JsonBlockCacheData<BlockEnv>,
    ) {
        assert_eq!(*expected.meta.read(), *actual.meta.read());
        assert_eq!(*expected.data.storage.read(), *actual.data.storage.read());
        assert_eq!(*expected.data.block_hashes.read(), *actual.data.block_hashes.read());
        let expected = expected.data.accounts.read();
        let actual = actual.data.accounts.read();
        assert_eq!(expected.len(), actual.len());
        for (address, account) in expected.iter() {
            let recovered = &actual[address];
            assert_eq!(
                (account.balance, account.nonce, account.code_hash),
                (recovered.balance, recovered.nonce, recovered.code_hash)
            );
            // AccountInfo equality ignores code, and Bytecode equality ignores kind.
            assert_eq!(
                serde_json::to_value(&account.code).unwrap(),
                serde_json::to_value(&recovered.code).unwrap()
            );
        }
    }

    #[test]
    fn shared_code_cache_stores_one_body_per_kind_and_preserves_full_state() {
        let cache = shared_code_fixture();
        let encoded = serde_json::to_value(&cache).unwrap();
        assert_eq!(encoded["cache_format_version"], 2);
        assert_eq!(encoded["codes"].as_array().unwrap().len(), 3);
        let first = Address::from_word(keccak256("shared-code-fixture-0")).to_string();
        let second = Address::from_word(keccak256("shared-code-fixture-1")).to_string();
        assert_eq!(
            encoded["accounts"][&first]["code_ref"],
            encoded["accounts"][&second]["code_ref"]
        );
        for account in encoded["accounts"].as_object().unwrap().values() {
            assert!(account.get("code").is_none());
        }
        let recovered: JsonBlockCacheData<BlockEnv> = serde_json::from_value(encoded).unwrap();
        assert_shared_code_state(&cache, &recovered);
        let accounts = recovered.data.accounts.read();
        let first = accounts[&Address::from_word(keccak256("shared-code-fixture-0"))]
            .code
            .as_ref()
            .unwrap();
        let second = accounts[&Address::from_word(keccak256("shared-code-fixture-1"))]
            .code
            .as_ref()
            .unwrap();
        assert!(std::ptr::eq(
            first.original_byte_slice().as_ptr(),
            second.original_byte_slice().as_ptr()
        ));
    }

    #[test]
    fn shared_code_cache_rejects_invalid_versions_and_references() {
        let valid = serde_json::to_value(shared_code_fixture()).unwrap();
        let first = Address::from_word(keccak256("shared-code-fixture-0")).to_string();
        let mut cases = Vec::new();
        let mut value = valid.clone();
        value["cache_format_version"] = 999.into();
        cases.push(value);
        let mut value = valid.clone();
        value.as_object_mut().unwrap().remove("cache_format_version");
        cases.push(value);
        let mut value = valid.clone();
        value.as_object_mut().unwrap().remove("codes");
        cases.push(value);
        let mut value = valid.clone();
        value["accounts"][&first]["code_ref"] = u64::MAX.into();
        cases.push(value);
        let mut value = valid.clone();
        value["accounts"][&first]["code_ref"] = (-1).into();
        cases.push(value);
        let mut value = valid.clone();
        value["accounts"][&first]["code"] = serde_json::to_value(Bytecode::default()).unwrap();
        cases.push(value);
        let mut value = valid;
        value["codes"] = serde_json::json!([]);
        cases.push(value);
        for (index, value) in cases.into_iter().enumerate() {
            assert!(
                serde_json::from_value::<JsonBlockCacheData<BlockEnv>>(value).is_err(),
                "invalid case {index} was accepted"
            );
        }
    }

    #[test]
    fn shared_code_cache_rejects_superseded_inline_layout() {
        let cache = shared_code_fixture();
        let value = serde_json::json!({"meta": &*cache.meta.read(), "accounts": &*cache.data.accounts.read(), "storage": &*cache.data.storage.read(), "block_hashes": &*cache.data.block_hashes.read(), "block_hash_cache_version": 1});
        assert!(serde_json::from_value::<JsonBlockCacheData<BlockEnv>>(value).is_err());
    }

    #[test]
    fn shared_code_cache_accepts_reordered_fields_and_rejects_truncated_data() {
        let cache = shared_code_fixture();
        let value = serde_json::to_value(&cache).unwrap();
        let mut reordered = serde_json::Map::new();
        for name in ["accounts", "block_hashes", "storage", "meta", "codes", "cache_format_version"]
        {
            reordered.insert(name.to_owned(), value[name].clone());
        }
        let bytes = serde_json::to_vec(&reordered).unwrap();
        let recovered: JsonBlockCacheData<BlockEnv> = serde_json::from_slice(&bytes).unwrap();
        assert_shared_code_state(&cache, &recovered);
        for length in [bytes.len() / 2, bytes.len() - 1] {
            assert!(
                serde_json::from_slice::<JsonBlockCacheData<BlockEnv>>(&bytes[..length]).is_err()
            );
        }
    }

    struct UpdatingCacheWriter<'a> {
        cache: &'a JsonBlockCacheData<BlockEnv>,
        output: Vec<u8>,
        updated: bool,
    }

    impl Write for UpdatingCacheWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if !self.updated {
                let code = Bytecode::new_raw(Bytes::from(vec![0x60, 0x42]));
                self.cache.data.accounts.write().insert(
                    Address::from_word(keccak256("concurrent-cache-addition")),
                    AccountInfo {
                        code_hash: code.hash_slow(),
                        code: Some(code),
                        ..Default::default()
                    },
                );
                self.updated = true;
            }
            self.output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn shared_code_cache_uses_one_account_snapshot_for_codes_and_references() {
        let cache = shared_code_fixture();
        let mut writer = UpdatingCacheWriter { cache: &cache, output: Vec::new(), updated: false };
        serde_json::to_writer(&mut writer, &cache).unwrap();
        assert!(writer.updated);
        assert_eq!(cache.data.accounts.read().len(), 6);
        let recovered: JsonBlockCacheData<BlockEnv> =
            serde_json::from_slice(&writer.output).unwrap();
        assert_eq!(recovered.data.accounts.read().len(), 5);
        assert!(
            !recovered
                .data
                .accounts
                .read()
                .contains_key(&Address::from_word(keccak256("concurrent-cache-addition")))
        );
    }
}
