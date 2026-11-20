package org.udlex.opensearch;

import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/** JNI uses byte arrays; Java's modified UTF-8 is never used for user data. */
final class UdlexNative {
    static {
        try {
            Path jar = Path.of(UdlexNative.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            Path library = jar.getParent().resolve("native").resolve(System.mapLibraryName("udlex_jni"));
            System.load(library.toAbsolutePath().toString());
        } catch (URISyntaxException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private UdlexNative() {}

    // Called by JNI so arbitrary native error text is decoded as real UTF-8 too.
    private static void fail(byte[] message) {
        throw new IllegalStateException(new String(message, StandardCharsets.UTF_8));
    }

    static native long open(byte[] bundle, byte[] language, boolean useRules);
    /**
     * Looks up term[0, length) and writes the lemma back into the same array. Returns the lemma
     * length, -1 when unchanged (nothing written), or a value above term.length when the lemma
     * does not fit (nothing written: grow the array and call again). All native failures throw.
     */
    static native int lemma(long handle, char[] term, int length);
    static native void close(long handle);
}
