package org.udlex.opensearch;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.FlagsAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.BytesRef;

/** A dependency-free executable contract test, run with -Xcheck:jni. */
public final class NativeContractTest {
    private static final String UNICODE = "🧪\u0000𐐨€ß🦊"; // 4-, 1-, 4-, 3-, 2-, 4-byte UTF-8

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Expected: base64(BUNDLE) base64(LANGUAGE) base64(TOKEN) base64(EXPECTED_LEMMA)");
        }
        for (int i = 0; i < args.length; i++) {
            args[i] = new String(Base64.getDecoder().decode(args[i]), StandardCharsets.UTF_8);
        }
        Path bundle = Path.of(args[0]).toRealPath();
        String language = args[1];
        String token = args[2];
        String expected = args[3];
        try (LexiconCache cache = new LexiconCache()) {
            NativeLexicon lexicon = cache.acquire(bundle, language, true);
            equal(lexicon, cache.acquire(bundle, language, true), "cache reuses immutable handle");
            equal(expected, lemma(lexicon, token), "known-token lemma");
            equal("", lemma(lexicon, ""), "empty-token roundtrip");
            for (String bad : new String[] {"\ud800", "\udc00", "a\ud800", "\udc00\ud800", "\ud800a"}) {
                expectFailure(IllegalArgumentException.class,
                        () -> lemma(lexicon, bad), "unpaired UTF-16 surrogate");
            }
            if (language.equals("rus")) { // shapes the generic arguments cannot force
                equal("человек", lemma(lexicon, "людей"), "lemma longer than the form grows the buffer");
                equal("кошка", lemma(lexicon, "Кошки"), "mixed-case form");
                equal("Москва", lemma(lexicon, "Москве"), "mixed-case lemma");
                char[] tight = "людей".toCharArray();
                equal(7, lexicon.lemma(tight, tight.length), "overflow reports the needed length");
                equal("людей", new String(tight), "overflow leaves the buffer intact");
            }
            NativeLexicon noRules = cache.acquire(bundle, language, false);
            equal(UNICODE, lemma(noRules, UNICODE), "supplementary Unicode and embedded NUL roundtrip");
            String longToken = UNICODE.repeat(12000);
            equal(longToken, lemma(noRules, longToken), "large unknown token");
            equal(expected, lemma(lexicon, token), "ordinary lookup after a large token");
            verifyTokenAttributes(lexicon, token, expected);
        }
        verifyNativeErrors(bundle, language);
        verifyConcurrentClose(bundle, language, token, expected);
        LexiconCache closedCache = new LexiconCache();
        NativeLexicon closed = closedCache.acquire(bundle, language, true);
        closedCache.close();
        closedCache.close();
        expectFailure(IllegalStateException.class, () -> lemma(closed, token), "use after close");
        expectFailure(IllegalStateException.class,
                () -> closedCache.acquire(bundle, language, true), "acquire after close");
        System.out.println("PASS: JNI UTF-8, fail-closed errors, cache lifetime, concurrent close, and "
                           + "token-stream attributes");
    }

    private static String lemma(NativeLexicon lexicon, String token) {
        char[] term = token.toCharArray();
        int length = lexicon.lemma(term, token.length());
        while (length > term.length) {
            term = Arrays.copyOf(term, length);
            length = lexicon.lemma(term, token.length());
        }
        return length < 0 ? token : new String(term, 0, length);
    }

    private static void verifyNativeErrors(Path bundle, String language) {
        expectFailure(IllegalStateException.class,
                ()
                        -> UdlexNative.open(NativeLexicon.encode(bundle.resolve("does-not-exist").toString()),
                                NativeLexicon.encode(language), true),
                "missing bundle fails closed");
        long raw = UdlexNative.open(
                NativeLexicon.encode(bundle.toString()), NativeLexicon.encode(language), false);
        try {
            expectFailure(IllegalStateException.class,
                    () -> UdlexNative.lemma(raw, new char[2], 3), "length beyond the buffer fails closed");
            expectFailure(IllegalStateException.class,
                    () -> UdlexNative.lemma(raw, null, 0), "null char array fails closed");
            expectFailure(IllegalStateException.class,
                    () -> UdlexNative.lemma(0, new char[0], 0), "null native handle fails closed");
            char[] spare = Arrays.copyOf("x\ud800".toCharArray(), 8);
            equal(-1, UdlexNative.lemma(raw, spare, 1),
                    "only term[0, length) is read: a trailing surrogate is ignored");
        } finally {
            UdlexNative.close(raw);
        }
    }

    private static void verifyTokenAttributes(NativeLexicon lexicon, String token, String expected)
            throws Exception {
        String[] terms = {token, token, "🧪", token, ""};
        String[] lemmas = {expected, token, "🧪", expected, ""};
        try (TokenStream stream = new UdlexTokenFilter(new FixtureStream(terms), lexicon)) {
            CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
            OffsetAttribute offset = stream.addAttribute(OffsetAttribute.class);
            PositionIncrementAttribute increment = stream.addAttribute(PositionIncrementAttribute.class);
            PositionLengthAttribute length = stream.addAttribute(PositionLengthAttribute.class);
            KeywordAttribute keyword = stream.addAttribute(KeywordAttribute.class);
            TypeAttribute type = stream.addAttribute(TypeAttribute.class);
            FlagsAttribute flags = stream.addAttribute(FlagsAttribute.class);
            PayloadAttribute payload = stream.addAttribute(PayloadAttribute.class);
            stream.reset();
            for (int i = 0; i < terms.length; i++) {
                check(stream.incrementToken(), "dropped token " + i);
                equal(lemmas[i], term.toString(), "term at " + i);
                equal(i * 100, offset.startOffset(), "start offset");
                equal(i * 100 + terms[i].length(), offset.endOffset(), "end offset");
                equal(i == 1 ? 0 : i + 1, increment.getPositionIncrement(), "position increment");
                equal(i + 1, length.getPositionLength(), "position length");
                equal(i == 1, keyword.isKeyword(), "keyword protection");
                equal("fixture-" + i, type.type(), "type");
                equal(i + 20, flags.getFlags(), "flags");
                equal(new BytesRef(new byte[] {(byte) i}), payload.getPayload(), "payload");
            }
            check(!stream.incrementToken(), "extra output token");
            stream.end();
            equal(1000, offset.endOffset(), "final offset");
        }
        equal(expected, lemma(lexicon, token), "closing a token stream must not close its shared dictionary");
    }

    private static void verifyConcurrentClose(Path bundle, String language, String token, String expected)
            throws Exception {
        NativeLexicon lexicon = new NativeLexicon(bundle, language, true);
        ExecutorService executor = Executors.newFixedThreadPool(4);
        CountDownLatch started = new CountDownLatch(4);
        CountDownLatch race = new CountDownLatch(1);
        List<Future<?>> workers = new ArrayList<>();
        try {
            for (int worker = 0; worker < 4; worker++) {
                workers.add(executor.submit(() -> {
                    equal(expected, lemma(lexicon, token), "concurrent initial lookup");
                    started.countDown();
                    try {
                        race.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(e);
                    }
                    for (int i = 0; i < 500; i++) {
                        try {
                            equal(expected, lemma(lexicon, token), "concurrent lookup");
                        } catch (IllegalStateException closed) {
                            equal("udlex dictionary is closed", closed.getMessage(), "close race error");
                            return;
                        }
                    }
                }));
            }
            check(started.await(30, java.util.concurrent.TimeUnit.SECONDS), "workers did not start");
            race.countDown();
            lexicon.close();
            for (Future<?> worker : workers) {
                worker.get(30, java.util.concurrent.TimeUnit.SECONDS);
            }
        } finally {
            race.countDown();
            lexicon.close();
            executor.shutdownNow();
        }
    }

    private static final class FixtureStream extends TokenStream {
        private final String[] terms;
        private int index;
        private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
        private final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
        private final PositionIncrementAttribute increment = addAttribute(PositionIncrementAttribute.class);
        private final PositionLengthAttribute length = addAttribute(PositionLengthAttribute.class);
        private final KeywordAttribute keyword = addAttribute(KeywordAttribute.class);
        private final TypeAttribute type = addAttribute(TypeAttribute.class);
        private final FlagsAttribute flags = addAttribute(FlagsAttribute.class);
        private final PayloadAttribute payload = addAttribute(PayloadAttribute.class);

        FixtureStream(String[] terms) {
            this.terms = terms;
        }

        @Override
        public void reset() throws java.io.IOException {
            super.reset();
            index = 0;
        }

        @Override
        public boolean incrementToken() {
            if (index == terms.length) {
                return false;
            }
            clearAttributes();
            term.append(terms[index]);
            offset.setOffset(index * 100, index * 100 + terms[index].length());
            increment.setPositionIncrement(index == 1 ? 0 : index + 1);
            length.setPositionLength(index + 1);
            keyword.setKeyword(index == 1);
            type.setType("fixture-" + index);
            flags.setFlags(index + 20);
            payload.setPayload(new BytesRef(new byte[] {(byte) index}));
            index++;
            return true;
        }

        @Override
        public void end() throws java.io.IOException {
            super.end();
            offset.setOffset(1000, 1000);
        }
    }

    private static void expectFailure(
            Class<? extends RuntimeException> expected, Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException failure) {
            if (expected.isInstance(failure)) {
                return;
            }
            throw new AssertionError(label + ": wrong exception", failure);
        }
        throw new AssertionError(label + ": did not fail");
    }

    private static void equal(Object expected, Object actual, String label) {
        if (!expected.equals(actual)) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }

    private static void check(boolean condition, String label) {
        if (!condition) {
            throw new AssertionError(label);
        }
    }
}
