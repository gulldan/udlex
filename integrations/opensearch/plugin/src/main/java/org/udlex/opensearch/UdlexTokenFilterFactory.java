package org.udlex.opensearch;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.analysis.TokenStream;
import org.opensearch.common.settings.Settings;
import org.opensearch.env.Environment;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.analysis.AbstractTokenFilterFactory;

final class UdlexTokenFilterFactory extends AbstractTokenFilterFactory {
    private final NativeLexicon lexicon;

    UdlexTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name,
            Settings settings, LexiconCache cache) throws IOException {
        super(indexSettings, name, settings);
        String language = settings.get("language");
        if (language == null || !language.matches("[a-z]{3}")) {
            throw new IllegalArgumentException("udlex language must be an ISO 639-3 code, e.g. rus");
        }
        String configuredPath = settings.get("bundle_path");
        if (configuredPath == null || configuredPath.isBlank()) {
            throw new IllegalArgumentException(
                    "udlex requires bundle_path inside the OpenSearch config directory");
        }
        // Native file I/O is outside Java's sandbox. Validate before crossing JNI.
        Path config = environment.configDir().toRealPath();
        Path bundle = config.resolve(configuredPath).toRealPath();
        if (!bundle.startsWith(config) || !Files.isDirectory(bundle)) {
            throw new IllegalArgumentException(
                    "udlex bundle_path must resolve to a directory inside OpenSearch config");
        }
        lexicon = cache.acquire(bundle, language, settings.getAsBoolean("use_rules", true));
    }

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new UdlexTokenFilter(tokenStream, lexicon);
    }
}
