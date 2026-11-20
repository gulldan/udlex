#include <jni.h>
#include <udlex.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

void fail(JNIEnv* env, jclass owner, const uint8_t* message, size_t length) {
    if (env->ExceptionCheck()) return;
    if (length > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        static constexpr char too_long[] = "udlex error message exceeds JNI limits";
        message = reinterpret_cast<const uint8_t*>(too_long);
        length = sizeof(too_long) - 1;
    }
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(length));
    if (bytes == nullptr) return;
    if (length != 0) {
        env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(length), reinterpret_cast<const jbyte*>(message));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(bytes);
            return;
        }
    }
    jmethodID method = env->GetStaticMethodID(owner, "fail", "([B)V");
    if (method != nullptr) env->CallStaticVoidMethod(owner, method, bytes);
    env->DeleteLocalRef(bytes);
}

void fail(JNIEnv* env, jclass owner, const char* message) {
    fail(env, owner, reinterpret_cast<const uint8_t*>(message), std::strlen(message));
}

void fail(JNIEnv* env, jclass owner, const std::string& message) {
    fail(env, owner, reinterpret_cast<const uint8_t*>(message.data()), message.size());
}

void fail_status(JNIEnv* env, jclass owner, const char* operation, int32_t status) {
    fail(env, owner, std::string("udlex ") + operation + " failed with status " + std::to_string(status));
}

bool read_bytes(JNIEnv* env, jclass owner, jbyteArray input, std::vector<uint8_t>& out) {
    if (input == nullptr) {
        fail(env, owner, "udlex JNI input must not be null");
        return false;
    }
    jsize length = env->GetArrayLength(input);
    out.resize(static_cast<size_t>(length));
    if (length != 0) {
        env->GetByteArrayRegion(input, 0, length, reinterpret_cast<jbyte*>(out.data()));
    }
    return !env->ExceptionCheck();
}

udlex_native_lexicon* pointer(jlong handle) {
    return reinterpret_cast<udlex_native_lexicon*>(static_cast<uintptr_t>(handle));
}

void out_of_memory(JNIEnv* env) {
    if (env->ExceptionCheck()) return;
    jclass error = env->FindClass("java/lang/OutOfMemoryError");
    if (error != nullptr) env->ThrowNew(error, "udlex JNI allocation failed");
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
        int extra = c < 0x80 ? 0 : c < 0xC2 ? -1 : c < 0xE0 ? 1 : c < 0xF0 ? 2 : c < 0xF5 ? 3 : -1;
        if (extra < 0 || i + static_cast<size_t>(extra) > n) return false;
        c &= 0xFFu >> (extra + 1);
        for (int k = 0; k < extra; ++k, ++i) {
            if ((s[i] & 0xC0) != 0x80) return false;
            c = (c << 6) | (s[i] & 0x3F);
        }
        if (c < minimum[extra] || c > 0x10FFFF || (c >= 0xD800 && c < 0xE000)) return false;
        if (c < 0x10000) {
            out.push_back(static_cast<jchar>(c));
        } else {
            out.push_back(static_cast<jchar>(0xD800 | ((c - 0x10000) >> 10)));
            out.push_back(static_cast<jchar>(0xDC00 | (c & 0x3FF)));
        }
    }
    return true;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_udlex_opensearch_UdlexNative_open(JNIEnv* env, jclass owner, jbyteArray bundle,
                                         jbyteArray language, jboolean use_rules) {
    try {
        std::vector<uint8_t> bundle_bytes;
        std::vector<uint8_t> language_bytes;
        if (!read_bytes(env, owner, bundle, bundle_bytes) || !read_bytes(env, owner, language, language_bytes)) return 0;
        udlex_native_lexicon* lexicon = nullptr;
        int32_t status = udlex_native_load(bundle_bytes.data(), bundle_bytes.size(), language_bytes.data(),
                                           language_bytes.size(), use_rules == JNI_TRUE, &lexicon);
        if (status != UDLEX_NATIVE_OK || lexicon == nullptr) {
            if (lexicon != nullptr) udlex_native_close(lexicon);
            fail_status(env, owner, "open", status);
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(lexicon));
    } catch (const std::bad_alloc&) {
        out_of_memory(env);
    } catch (...) {
        fail(env, owner, "udlex JNI open: unexpected native exception");
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_org_udlex_opensearch_UdlexNative_lemma(JNIEnv* env, jclass owner, jlong handle, jcharArray term, jint length) {
    try {
        if (handle == 0) {
            fail(env, owner, "udlex dictionary handle is null");
            return -1;
        }
        if (term == nullptr) {
            fail(env, owner, "udlex JNI term buffer must not be null");
            return -1;
        }
        jsize capacity = env->GetArrayLength(term);
        if (length < 0 || length > capacity) {
            fail(env, owner, "udlex JNI term length exceeds its buffer");
            return -1;
        }
        // A token is a dozen bytes; heap allocations per call cost more than the lookup.
        thread_local std::vector<jchar> chars;
        thread_local std::vector<uint8_t> input;
        thread_local std::vector<uint8_t> output;
        chars.resize(static_cast<size_t>(length));
        if (length != 0) env->GetCharArrayRegion(term, 0, length, chars.data());
        if (env->ExceptionCheck()) return -1;
        if (!encode_utf8(chars.data(), chars.size(), input)) {
            jclass error = env->FindClass("java/lang/IllegalArgumentException");
            if (error != nullptr) env->ThrowNew(error, "udlex input contains an unpaired UTF-16 surrogate");
            return -1;
        }
        if (output.size() < std::max<size_t>(64, input.size() + 32)) output.resize(std::max<size_t>(64, input.size() + 32));
        for (int attempt = 0; attempt < 3; ++attempt) {
            size_t produced = 0;
            int32_t status = udlex_native_lemma(pointer(handle), input.data(), input.size(), 0,
                                               output.data(), output.size(), &produced);
            if (status == UDLEX_NATIVE_MORE_SPACE) {
                if (produced <= output.size() || produced > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
                    fail(env, owner, "udlex returned an invalid required output size");
                    return -1;
                }
                output.resize(produced);
                continue;
            }
            if (status < 0) {
                fail_status(env, owner, "lemma", status);
                return -1;
            }
            if (status != UDLEX_NATIVE_OK || produced > output.size()) {
                fail(env, owner, "udlex returned an invalid C ABI response");
                return -1;
            }
            if (produced == input.size() && std::equal(input.begin(), input.end(), output.begin())) return -1;
            if (!decode_utf8(output.data(), produced, chars)) {
                fail(env, owner, "udlex returned invalid UTF-8");
                return -1;
            }
            jsize written = static_cast<jsize>(chars.size()); // UTF-16 units never exceed UTF-8 bytes, so this fits
            // Does not fit: the caller grows its buffer and asks again.
            if (written > capacity) return written;
            if (written != 0) env->SetCharArrayRegion(term, 0, written, chars.data());
            return written;
        }
        fail(env, owner, "udlex output size did not converge");
    } catch (const std::bad_alloc&) {
        out_of_memory(env);
    } catch (...) {
        fail(env, owner, "udlex JNI lemma: unexpected native exception");
    }
    return -1;
}

extern "C" JNIEXPORT void JNICALL
Java_org_udlex_opensearch_UdlexNative_close(JNIEnv*, jclass, jlong handle) {
    if (handle != 0) udlex_native_close(pointer(handle));
}
