#include "org_udlex_opensearch_UdlexNative.h"
#include <udlex.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>

namespace {

// Our diagnostics contain only ASCII literals and numeric status codes. User
// text is converted separately using standard UTF-8 in both directions.
void throw_java(JNIEnv* env, const char* type, const char* message) {
    if (env->ExceptionCheck()) {
        return;
    }
    const jclass exception = env->FindClass(type);
    if (exception != nullptr) {
        env->ThrowNew(exception, message);
        env->DeleteLocalRef(exception);
    }
}

void fail(JNIEnv* env, const char* message) { throw_java(env, "java/lang/IllegalStateException", message); }

void fail_status(JNIEnv* env, const char* operation, int32_t status) {
    char message[96];
    std::snprintf(message, sizeof(message), "udlex %s failed with status %d", operation,
                  static_cast<int>(status));
    fail(env, message);
}

bool read_bytes(JNIEnv* env, jbyteArray input, std::vector<uint8_t>& out) {
    if (input == nullptr) {
        fail(env, "udlex JNI input must not be null");
        return false;
    }
    const jsize length = env->GetArrayLength(input);
    out.resize(static_cast<size_t>(length));
    if (length != 0) {
        env->GetByteArrayRegion(input, 0, length, reinterpret_cast<jbyte*>(out.data()));
    }
    return !env->ExceptionCheck();
}

udlex_native_lexicon* pointer(jlong handle) {
    static_assert(sizeof(uintptr_t) <= sizeof(jlong), "A Java long must hold a native pointer");
    // JNI transports this opaque pointer in a Java long and returns it unchanged.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<udlex_native_lexicon*>(static_cast<uintptr_t>(handle));
}

// JNI's own UTF conversions are "modified UTF-8", so both directions are by hand.
// Returns false on an unpaired surrogate (fail-closed, like the Java encoder did).
bool encode_utf8(const jchar* s, size_t n, std::vector<uint8_t>& out) {
    out.clear();
    for (size_t i = 0; i < n; ++i) {
        uint32_t c = s[i];
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < n && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000) {
            c = 0x10000 + ((c - 0xD800) << 10) + (s[++i] - 0xDC00);
        } else if (c >= 0xD800 && c < 0xE000) {
            return false;
        }
        if (c < 0x80) {
            out.push_back(static_cast<uint8_t>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<uint8_t>(0xC0 | (c >> 6)));
            out.push_back(static_cast<uint8_t>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<uint8_t>(0xE0 | (c >> 12)));
            out.push_back(static_cast<uint8_t>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<uint8_t>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<uint8_t>(0xF0 | (c >> 18)));
            out.push_back(static_cast<uint8_t>(0x80 | ((c >> 12) & 0x3F)));
            out.push_back(static_cast<uint8_t>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<uint8_t>(0x80 | (c & 0x3F)));
        }
    }
    return true;
}

// Strict: rejects overlongs, surrogates and code points above U+10FFFF.
bool decode_utf8(const uint8_t* s, size_t n, std::vector<jchar>& out) {
    static constexpr uint32_t minimum[] = {0, 0x80, 0x800, 0x10000};
    out.clear();
    for (size_t i = 0; i < n;) {
        uint32_t c = s[i++];
        unsigned int extra = 0;
        if (c >= 0x80) {
            if (c < 0xC2 || c >= 0xF5) {
                return false;
            }
            extra = 1U + static_cast<unsigned int>(c >= 0xE0) + static_cast<unsigned int>(c >= 0xF0);
        }
        if (extra > n - i) {
            return false;
        }
        c &= 0xFFu >> (extra + 1);
        for (unsigned int k = 0; k < extra; ++k, ++i) {
            if ((s[i] & 0xC0) != 0x80) {
                return false;
            }
            c = (c << 6) | (s[i] & 0x3F);
        }
        if (c < minimum[extra] || c > 0x10FFFF || (c >= 0xD800 && c < 0xE000)) {
            return false;
        }
        if (c < 0x10000) {
            out.push_back(static_cast<jchar>(c));
        } else {
            out.push_back(static_cast<jchar>(0xD800 | ((c - 0x10000) >> 10)));
            out.push_back(static_cast<jchar>(0xDC00 | (c & 0x3FF)));
        }
    }
    return true;
}

struct TermBuffers {
    std::vector<jchar> chars;
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;

    // OpenSearch workers are long-lived. Keep ordinary token buffers for reuse
    // and release exceptional allocations before returning to the pool.
    void release_oversized() {
        constexpr size_t limit = 64 * 1024;
        if (chars.capacity() > limit / sizeof(jchar) || input.capacity() > limit ||
            output.capacity() > limit) {
            *this = {};
        }
    }
};

jint lookup(JNIEnv* env, jlong handle, jcharArray term, jint length, TermBuffers& buffers) {
    if (handle == 0) {
        fail(env, "udlex dictionary handle is null");
        return -1;
    }
    if (term == nullptr) {
        fail(env, "udlex JNI term buffer must not be null");
        return -1;
    }
    jsize capacity = env->GetArrayLength(term);
    if (length < 0 || length > capacity) {
        fail(env, "udlex JNI term length exceeds its buffer");
        return -1;
    }
    auto& [chars, input, output] = buffers;
    chars.resize(static_cast<size_t>(length));
    if (length != 0) {
        env->GetCharArrayRegion(term, 0, length, chars.data());
    }
    if (env->ExceptionCheck()) {
        return -1;
    }
    if (!encode_utf8(chars.data(), chars.size(), input)) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "udlex input contains an unpaired UTF-16 surrogate");
        return -1;
    }
    const size_t initial_size = std::max<size_t>(64, input.size() + 32);
    if (output.size() < initial_size) {
        output.resize(initial_size);
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        size_t produced = 0;
        int32_t status = udlex_native_lemma(pointer(handle), input.data(), input.size(), 0, output.data(),
                                            output.size(), &produced);
        if (status == UDLEX_NATIVE_MORE_SPACE) {
            if (produced <= output.size() ||
                produced > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
                fail(env, "udlex returned an invalid required output size");
                return -1;
            }
            output.resize(produced);
            continue;
        }
        if (status < 0) {
            fail_status(env, "lemma", status);
            return -1;
        }
        if (status != UDLEX_NATIVE_OK || produced > output.size()) {
            fail(env, "udlex returned an invalid C ABI response");
            return -1;
        }
        if (produced == input.size() && std::equal(input.begin(), input.end(), output.begin())) {
            return -1;
        }
        if (!decode_utf8(output.data(), produced, chars)) {
            fail(env, "udlex returned invalid UTF-8");
            return -1;
        }
        jsize written =
            static_cast<jsize>(chars.size()); // UTF-16 units never exceed UTF-8 bytes, so this fits
        // Does not fit: the caller grows its buffer and asks again.
        if (written > capacity) {
            return written;
        }
        if (written != 0) {
            env->SetCharArrayRegion(term, 0, written, chars.data());
        }
        return written;
    }
    fail(env, "udlex output size changed between lookups");
    return -1;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL Java_org_udlex_opensearch_UdlexNative_open(JNIEnv* env, jclass,
                                                                              jbyteArray bundle,
                                                                              jbyteArray language,
                                                                              jboolean use_rules) {
    try {
        std::vector<uint8_t> bundle_bytes;
        std::vector<uint8_t> language_bytes;
        if (!read_bytes(env, bundle, bundle_bytes) || !read_bytes(env, language, language_bytes)) {
            return 0;
        }
        udlex_native_lexicon* lexicon = nullptr;
        int32_t status = udlex_native_load(bundle_bytes.data(), bundle_bytes.size(), language_bytes.data(),
                                           language_bytes.size(), use_rules == JNI_TRUE, &lexicon);
        if (status != UDLEX_NATIVE_OK || lexicon == nullptr) {
            if (lexicon != nullptr) {
                udlex_native_close(lexicon);
            }
            fail_status(env, "open", status);
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(lexicon));
    } catch (const std::bad_alloc&) {
        throw_java(env, "java/lang/OutOfMemoryError", "udlex JNI allocation failed");
    } catch (...) {
        fail(env, "udlex JNI open: unexpected native exception");
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_org_udlex_opensearch_UdlexNative_lemma(JNIEnv* env, jclass,
                                                                              jlong handle, jcharArray term,
                                                                              jint length) {
    thread_local TermBuffers buffers;
    jint result = -1;
    try {
        result = lookup(env, handle, term, length, buffers);
    } catch (const std::bad_alloc&) {
        throw_java(env, "java/lang/OutOfMemoryError", "udlex JNI allocation failed");
    } catch (...) {
        fail(env, "udlex JNI lemma: unexpected native exception");
    }
    buffers.release_oversized();
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_org_udlex_opensearch_UdlexNative_close(JNIEnv*, jclass, jlong handle) {
    if (handle != 0) {
        udlex_native_close(pointer(handle));
    }
}
