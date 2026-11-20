package org.udlex.opensearch;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** The read lock permits concurrent Rust lookups and prevents close/use races. */
final class NativeLexicon implements AutoCloseable {
    private final ReentrantReadWriteLock lifetime = new ReentrantReadWriteLock();
    private long handle;

    NativeLexicon(Path bundle, String language, boolean useRules) {
        handle = UdlexNative.open(encode(bundle.toString()), encode(language), useRules);
        if (handle == 0) {
            throw new IllegalStateException("udlex returned a null dictionary handle");
        }
    }

    /** In-place lookup with UdlexNative.lemma's contract; the filter's zero-allocation path. */
    int lemma(char[] term, int length) {
        lifetime.readLock().lock();
        try {
            if (handle == 0) {
                throw new IllegalStateException("udlex dictionary is closed");
            }
            return UdlexNative.lemma(handle, term, length);
        } finally {
            lifetime.readLock().unlock();
        }
    }

    /** Allocating convenience for tests and tools. */
    String lemma(String token) {
        char[] term = token.toCharArray();
        int n = lemma(term, term.length);
        if (n > term.length) n = lemma(term = Arrays.copyOf(term, n), token.length());
        return n < 0 ? token : new String(term, 0, n);
    }

    // String.getBytes is intrinsic and allocates once, but it writes '?' for an
    // unpaired surrogate; the scan keeps the fail-closed contract.
    static byte[] encode(String value) {
        for (int i = 0, n = value.length(); i < n; i++) {
            char c = value.charAt(i);
            if (Character.isHighSurrogate(c) && i + 1 < n && Character.isLowSurrogate(value.charAt(i + 1))) {
                i++;
            } else if (Character.isSurrogate(c)) {
                throw new IllegalArgumentException("udlex input contains an unpaired UTF-16 surrogate");
            }
        }
        return value.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
        lifetime.writeLock().lock();
        try {
            if (handle != 0) {
                UdlexNative.close(handle);
                handle = 0;
            }
        } finally {
            lifetime.writeLock().unlock();
        }
    }
}
