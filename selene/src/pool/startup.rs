/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The registration window: the one time a pool client may register host memory.
//!
//! Registering host memory with the GPU consumes device memory, which a serving
//! instance must not do. So a client attaches to the manager's memory once, at
//! startup, and the window closes: either the first attach finished (and
//! [`super::engine::Engine::close_registration`] then makes the registrar refuse
//! everything), or attachment failed and startup must stop.

use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

/// A range this client registered at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredRegion {
    /// What the range is: `"arena"` and `"staging"` are the manager's shared files, mapped from
    /// their paths; `"data-staging"` is this process's own staging for a manager on another host.
    pub name: &'static str,
    /// The start of the range in this process.
    pub address: usize,
    /// Its length in bytes.
    pub bytes: usize,
}

/// A range the client is about to register, offered to the approval callback before any is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedRegion {
    /// As [`RegisteredRegion::name`].
    pub name: &'static str,
    /// Its length in bytes.
    pub bytes: u64,
}

/// Judges the complete registration plan. `Err(reason)` registers nothing.
pub type Approve = std::sync::Arc<dyn Fn(&[PlannedRegion]) -> Result<(), String> + Send + Sync>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The I/O thread has not reached the manager yet.
    Waiting,
    /// It has, and is registering.
    Registering,
    /// Registered; waiting for the manager to accept the session.
    Opening,
    /// Attached: the window has closed behind the registrations.
    Done,
    /// Startup failed; no further registration may begin.
    Closed,
}

struct State {
    phase: Phase,
    regions: crate::Result<Vec<RegisteredRegion>>,
    retry: String,
}

// Two attempts, each with up to 2 s connect, 5 s Hello and two 5 s Open waits, plus backoff.
pub(crate) const OPENING_WAIT: Duration = Duration::from_secs(40);
pub(crate) const NO_CONNECTION_ATTEMPT: &str = "no connection attempt completed";

pub(crate) struct Startup {
    state: Mutex<State>,
    changed: Condvar,
}

impl Startup {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                phase: Phase::Waiting,
                regions: Ok(Vec::new()),
                retry: NO_CONNECTION_ATTEMPT.into(),
            }),
            changed: Condvar::new(),
        }
    }

    /// The I/O thread asks to register. `false` once the window is closed.
    pub(crate) fn begin(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.phase == Phase::Waiting {
            state.phase = Phase::Registering;
            self.changed.notify_all();
            true
        } else {
            false
        }
    }

    /// Keep registrations pending until the manager accepts the session.
    pub(crate) fn registered(&self, regions: Vec<RegisteredRegion>) {
        let mut state = self.state.lock().unwrap();
        if state.phase == Phase::Registering {
            state.phase = Phase::Opening;
            state.regions = Ok(regions);
        }
        self.changed.notify_all();
    }

    pub(crate) fn opened(&self) {
        let mut state = self.state.lock().unwrap();
        if state.phase == Phase::Opening {
            state.phase = Phase::Done;
        }
        self.changed.notify_all();
    }

    /// The pool thread ended before attachment completed.
    pub(crate) fn close(&self) {
        self.fail("the pool thread stopped before attachment completed");
    }

    pub(crate) fn fail(&self, reason: impl Into<String>) {
        let mut state = self.state.lock().unwrap();
        if matches!(
            state.phase,
            Phase::Waiting | Phase::Registering | Phase::Opening
        ) {
            state.phase = Phase::Closed;
            state.regions = Err(crate::Error::new(reason));
        }
        self.changed.notify_all();
    }

    pub(crate) fn is_open(&self) -> bool {
        self.state.lock().unwrap().phase == Phase::Done
    }

    /// Whether the client gave up or was disabled before attaching.
    pub(crate) fn is_closed(&self) -> bool {
        self.state.lock().unwrap().phase == Phase::Closed
    }

    /// Wait for the manager and registration, preserving the cause of a failed attach.
    pub(crate) fn wait(&self, limit: Duration) -> crate::Result<Vec<RegisteredRegion>> {
        self.wait_with_opening(limit, Some(OPENING_WAIT))
    }

    pub(crate) fn retry(&self, reason: String) {
        self.state.lock().unwrap().retry = reason;
    }

    pub(crate) fn wait_with_opening(
        &self,
        limit: Duration,
        opening_limit: Option<Duration>,
    ) -> crate::Result<Vec<RegisteredRegion>> {
        let deadline = Instant::now() + limit;
        let mut state = self.state.lock().unwrap();
        while state.phase == Phase::Waiting {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                state.phase = Phase::Closed;
                state.regions = Err(reach_timeout(limit, &state.retry));
                break;
            };
            state = self.changed.wait_timeout(state, left).unwrap().0;
        }
        while state.phase == Phase::Registering {
            state = self.changed.wait(state).unwrap();
        }
        let deadline = opening_limit.map(|limit| (Instant::now() + limit, limit));
        while state.phase == Phase::Opening {
            let Some((deadline, opening_limit)) = deadline else {
                state = self.changed.wait(state).unwrap();
                continue;
            };
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                state.phase = Phase::Closed;
                state.regions = Err(crate::Error::new(format!(
                    "pool session opening did not finish within {opening_limit:?} ({}); check the manager before restarting",
                    state.retry
                )));
                break;
            };
            state = self.changed.wait_timeout(state, left).unwrap().0;
        }
        state.regions.clone().map_err(attachment_error)
    }
}

pub(crate) fn attachment_error(error: impl std::fmt::Display) -> crate::Error {
    crate::Error::new(format!(
        "cannot attach the configured prefix cache pool: {error}; remove --prefix-cache-pool on every rank to run with local tiers only"
    ))
}

pub(crate) fn reach_timeout(limit: Duration, reason: &str) -> crate::Error {
    crate::Error::new(format!(
        "the pool manager did not answer within {limit:?} of startup ({reason}); start the manager before the instance"
    ))
}

/// Wakes startup even when the pool thread unwinds.
pub(crate) struct StartupGuard(pub Arc<Startup>);

impl Drop for StartupGuard {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Registers nothing: the window is closed.
pub(crate) struct ClosedRegistrar;

impl crate::HostRegistrar for ClosedRegistrar {
    fn register(
        &self,
        _context: usize,
        _address: usize,
        _bytes: usize,
    ) -> crate::Result<Box<dyn crate::Registration>> {
        Err(crate::Error::new(
            "host memory can be registered only at startup, and the pool's registration is closed",
        ))
    }

    fn map_split(
        &self,
        _requests: &[(usize, i32, usize)],
    ) -> crate::Result<Box<dyn crate::SplitMapping>> {
        Err(crate::Error::new(
            "host memory can be mapped only at startup, and the pool's registration is closed",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_time_does_not_count_against_the_startup_wait() {
        let startup = Startup::new();
        assert!(startup.begin());
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                startup.wait_with_opening(Duration::ZERO, Some(Duration::from_millis(1)))
            });
            std::thread::sleep(Duration::from_millis(20));
            startup.registered(Vec::new());
            startup.opened();
            assert!(waiter.join().unwrap().is_ok());
        });
    }

    #[test]
    fn opening_expiry_fails_startup_and_cannot_be_completed_late() {
        let startup = Startup::new();
        assert!(startup.begin());
        startup.registered(Vec::new());
        let reason = startup
            .wait_with_opening(Duration::ZERO, Some(Duration::from_millis(1)))
            .unwrap_err()
            .to_string();
        assert!(
            reason.contains("session opening did not finish"),
            "{reason}"
        );
        assert!(reason.contains("1ms"), "{reason}");
        assert!(reason.contains("remove --prefix-cache-pool"), "{reason}");
        startup.opened();
        assert!(startup.is_closed());
        assert_eq!(
            startup.wait(Duration::ZERO).unwrap_err().to_string(),
            reason
        );
    }
}
