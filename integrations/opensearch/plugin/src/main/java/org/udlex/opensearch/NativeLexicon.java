package org.udlex.opensearch;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
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

    // Used only while opening a dictionary. JNI term lookup takes Lucene's char buffer directly.
    static byte[] encode(String value) {
        if (!StandardCharsets.UTF_8.newEncoder().canEncode(value)) {
            throw new IllegalArgumentException("udlex input contains an unpaired UTF-16 surrogate");
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
