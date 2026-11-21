//! Hardware counters around the measured loop via `perf_event_open`,
//! user-space only: that is what `perf_event_paranoid = 2` permits.

#[cfg(target_os = "linux")]
mod imp {
    /// `perf_event_attr` up to `PERF_ATTR_SIZE_VER0`.
    #[repr(C)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    const DISABLED: u64 = 1;
    const EXCLUDE_KERNEL: u64 = 1 << 5;
    const EXCLUDE_HV: u64 = 1 << 6;
    const ENABLE: libc::c_ulong = 0x2400;
    const DISABLE: libc::c_ulong = 0x2401;
    const RESET: libc::c_ulong = 0x2403;

    pub struct Counter(libc::c_int);

    impl Counter {
        pub fn open(kind: u32, config: u64) -> Option<Self> {
            let attr = Attr {
                kind,
                size: 64,
                config,
                sample_period: 0,
                sample_type: 0,
                // TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING: a multiplexed
                // counter is scaled, and an unscheduled one reads as absent.
                read_format: 1 | 2,
                flags: DISABLED | EXCLUDE_KERNEL | EXCLUDE_HV,
                wakeup_events: 0,
                bp_type: 0,
                config1: 0,
            };
            // SAFETY: a well-formed attr; this thread, any CPU, no group.
            let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, &attr, 0, -1, -1, 0) };
            (fd >= 0).then_some(Self(fd as libc::c_int))
        }

        pub fn start(&self) {
            // SAFETY: ioctls on a counter this struct owns.
            unsafe {
                libc::ioctl(self.0, RESET, 0);
                libc::ioctl(self.0, ENABLE, 0);
            }
        }

        /// The count, scaled up by enabled/running time when the kernel
        /// multiplexed the counter; `None` when it never ran at all.
        pub fn stop(&self) -> Option<u64> {
            let mut value = [0_u64; 3];
            // SAFETY: a 24-byte read (value, enabled, running) into 24 bytes.
            let read = unsafe {
                libc::ioctl(self.0, DISABLE, 0);
                libc::read(self.0, value.as_mut_ptr().cast(), 24)
            };
            let [count, enabled, running] = value;
            (read == 24 && running > 0)
                .then(|| (count as u128 * u128::from(enabled) / u128::from(running)) as u64)
        }
    }

    impl Drop for Counter {
        fn drop(&mut self) {
            // SAFETY: the descriptor is ours and closed once.
            unsafe { libc::close(self.0) };
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub struct Counter;

    impl Counter {
        pub fn open(_kind: u32, _config: u64) -> Option<Self> {
            None
        }
        pub fn start(&self) {}
        pub fn stop(&self) -> Option<u64> {
            None
        }
    }
}

pub use imp::Counter;

const HARDWARE: u32 = 0;
const CACHE: u32 = 3;

/// Two groups of at most four, so a six-counter PMU never multiplexes.
pub const GROUPS: [&[(&str, u32, u64)]; 2] = [
    &[
        ("cycles", HARDWARE, 0),
        ("instructions", HARDWARE, 1),
        ("branch_misses", HARDWARE, 5),
        ("l1d_miss", CACHE, 0x1_0000),
    ],
    &[
        ("cache_miss", HARDWARE, 3),
        ("llc_miss", CACHE, 0x1_0002),
        ("dtlb_miss", CACHE, 0x1_0003),
    ],
];
