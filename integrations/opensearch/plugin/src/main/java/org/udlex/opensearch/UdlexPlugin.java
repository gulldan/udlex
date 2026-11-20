package org.udlex.opensearch;

import java.io.IOException;
import java.util.Map;
import org.opensearch.index.analysis.TokenFilterFactory;
import org.opensearch.indices.analysis.AnalysisModule.AnalysisProvider;
import org.opensearch.plugins.AnalysisPlugin;
import org.opensearch.plugins.Plugin;

/** Node-scoped ownership: an immutable dictionary is shared by all index analyzers. */
public final class UdlexPlugin extends Plugin implements AnalysisPlugin {
    private final LexiconCache lexicons = new LexiconCache();

    @Override
    public Map<String, AnalysisProvider<TokenFilterFactory>> getTokenFilters() {
        return Map.of("udlex", AnalysisPlugin.requiresAnalysisSettings(
            (indexSettings, environment, name, settings) ->
                new UdlexTokenFilterFactory(indexSettings, environment, name, settings, lexicons)
        ));
    }

    @Override
    public void close() throws IOException {
        lexicons.close();
    }
}
