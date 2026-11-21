package org.udlex.opensearch;

import java.io.IOException;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;

/** One input token becomes one output token; all attributes except term stay intact. */
final class UdlexTokenFilter extends TokenFilter {
    private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    private final KeywordAttribute keyword = addAttribute(KeywordAttribute.class);
    private final NativeLexicon lexicon;

    UdlexTokenFilter(TokenStream input, NativeLexicon lexicon) {
        super(input);
        this.lexicon = lexicon;
    }

    @Override
    public boolean incrementToken() throws IOException {
        if (!input.incrementToken()) {
            return false;
        }
        if (!keyword.isKeyword()) {
            int inputLength = term.length();
            int n = lexicon.lemma(term.buffer(), inputLength);
            while (n > term.buffer().length) { // the lemma outgrew the buffer: grow it, look up again
                n = lexicon.lemma(term.resizeBuffer(n), inputLength);
            }
            if (n >= 0) {
                term.setLength(n);
            }
        }
        return true;
    }
}
