//! Terminal adapters for prepared-core acquisition.
use crate::cli::HubOptions;
#[cfg(unix)]
pub use flare::hub::{available_bytes, transport_root};
pub use flare::hub::{is_repository, repository};
use std::path::{Path, PathBuf};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[cfg(unix)]
pub fn token(options: &HubOptions) -> Result<Option<String>> {
    flare::hub::token(&options.into())
}
pub fn run_download(args: crate::cli::Download) -> Result<()> {
    if !args.gpus.is_empty() {
        crate::environment::configure(&args.gpus)?;
    }
    let startup = flare_helios::startup::StartupProgress::stderr_rows();
    let entry = download(
        &args.repository,
        &args.hub,
        args.gpus.len(),
        &args.publisher_keys,
        args.trust_store.as_deref(),
        &[],
        &startup,
    )?;
    println!("{}", entry.directory().display());
    Ok(())
}

pub fn download(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    profiles: &[flare_helios::runtime_core::Publisher],
    startup: &flare_helios::startup::StartupProgress,
) -> Result<flare_helios::core::cache::Entry> {
    let mut resolved = flare::options::HubOptions::from(options);
    if resolved.revision.is_none() && repository(source)?.1.is_none() {
        resolved.revision = Some(crate::hardware::detect_isolated(gpu_count.max(1))?);
    }
    let cache = flare::cache_support::model_root(options.core_cache.as_deref())
        .map_err(|error| error.to_string())?;
    let progress = flare_helios::core::progress::AcquisitionProgress::new();
    let acquire = || {
        let display = flare_helios::core::progress::AcquisitionDisplay::new(startup, &progress);
        display.stage(
            "Acquire core",
            flare_helios::core::progress::Counter::Network,
            None,
        );
        let result = flare::hub::acquire_watched(
            source,
            &resolved,
            gpu_count,
            publisher_keys,
            trust_store,
            profiles,
            progress.clone(),
        );
        if result.is_ok()
            || result
                .as_ref()
                .err()
                .is_some_and(|error| error.is::<flare::hub::PublisherKeyRequired>())
        {
            display.finish();
        }
        result
    };
    let entry = match acquire() {
        Ok(entry) => entry,
        Err(error) => {
            let Some(required) = error.downcast_ref::<flare::hub::PublisherKeyRequired>() else {
                return Err(error);
            };
            crate::trust::import_publisher_key(
                required,
                &flare::trust::ReviewOptions {
                    publisher_keys: publisher_keys.to_vec(),
                    profiles: profiles.to_vec(),
                    hub: resolved.clone(),
                    trust_store: trust_store.map(Path::to_owned),
                    ..Default::default()
                },
            )?;
            acquire()?
        }
    };
    flare_helios::core::cache::Root::open(cache)?.publish_entry(&entry)?;
    Ok(entry)
}

#[cfg(unix)]
pub fn download_watched(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    token: Option<String>,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    profiles: &[flare_helios::runtime_core::Publisher],
    progress: Option<flare_helios::core::progress::AcquisitionProgress>,
) -> Result<flare_helios::core::cache::Entry> {
    flare::hub::download_watched(
        source,
        &options.into(),
        gpu_count,
        token,
        publisher_keys,
        trust_store,
        profiles,
        progress,
    )
}
pub fn refresh_model(
    store: &flare_helios::core::cache::Root,
    entry: flare_helios::core::cache::Entry,
    pinned: bool,
    args: &crate::cli::Run,
) -> Result<(flare_helios::core::cache::Entry, Option<String>)> {
    let model = entry.model().to_owned();
    let version = entry.id().to_owned();
    let refresh = |entry| {
        flare::hub::refresh_model(
            store,
            entry,
            pinned,
            &(&args.hub).into(),
            args.gpus.len(),
            &args.publisher_keys,
            args.trust_store.as_deref(),
        )
    };
    match refresh(entry) {
        Ok(result) => Ok(result),
        Err(error) => {
            let Some(required) = error.downcast_ref::<flare::hub::PublisherKeyRequired>() else {
                return Err(error);
            };
            crate::trust::import_publisher_key(
                required,
                &flare::trust::ReviewOptions {
                    publisher_keys: args.publisher_keys.clone(),
                    hub: (&args.hub).into(),
                    trust_store: args.trust_store.clone(),
                    ..Default::default()
                },
            )?;
            refresh(store.entry(&model, &version)?)
        }
    }
}
#[cfg(test)]
#[path = "hub_fixtures.rs"]
pub(crate) mod tests;

#[cfg(all(test, unix))]
pub fn remember_download(
    root: &Path,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    options: &HubOptions,
    gpu_count: usize,
    entry: &flare_helios::core::cache::Entry,
    expected_envelope: [u8; 32],
) -> Result<()> {
    flare::hub::remember_download(
        root,
        repository,
        revision,
        commit,
        filename,
        &options.into(),
        gpu_count,
        entry,
        expected_envelope,
    )
}
