package org.udlex.opensearch;

import java.net.URISyntaxException;
import java.nio.file.Path;

/** JNI uses byte arrays; Java's modified UTF-8 is never used for user data. */
final class UdlexNative {
    static {
        try {
            Path jar = Path.of(UdlexNative.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            String arch = System.getProperty("os.arch");
            String directory = switch (arch) {
                case "amd64", "x86_64" -> "x86_64";
                case "arm64", "aarch64" -> "aarch64";
                default -> arch;
            };
            Path library = jar.getParent().resolve("native").resolve(directory)
                    .resolve(System.mapLibraryName("udlex_jni"));
            loadLibrary(library);
        } catch (URISyntaxException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private UdlexNative() {}

    @SuppressWarnings("restricted") // The package-relative JNI library is part of the verified plugin ZIP.
    private static void loadLibrary(Path library) {
        System.load(library.toAbsolutePath().toString());
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
