package org.udlex.opensearch;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** No hot reload: a published bundle must stay immutable until this node stops. */
final class LexiconCache implements AutoCloseable {
    private record Key(Path bundle, String language, boolean useRules) {}
    private final Map<Key, NativeLexicon> handles = new HashMap<>();
    private boolean closed;

    synchronized NativeLexicon acquire(Path bundle, String language, boolean useRules) {
        if (closed) {
            throw new IllegalStateException("udlex plugin is closed");
        }
        return handles.computeIfAbsent(new Key(bundle, language, useRules),
                key -> new NativeLexicon(key.bundle(), key.language(), key.useRules()));
    }

    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            handles.values().forEach(NativeLexicon::close);
            handles.clear();
        }
    }
}
