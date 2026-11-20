/* C interface to the udlex dictionary reader. */
#ifndef UDLEX_NATIVE_H
#define UDLEX_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UDLEX_NATIVE_OK 0
#define UDLEX_NATIVE_MORE_SPACE 1
#define UDLEX_NATIVE_INVALID_ARGUMENT (-1)
#define UDLEX_NATIVE_INVALID_UTF8 (-2)
#define UDLEX_NATIVE_OPEN_FAILED (-3) /* missing, unreadable, or invalid lexicon */
#define UDLEX_NATIVE_PANIC (-4)       /* internal Rust panic caught at the ABI */

/* All pointers and slots must remain live, aligned, and disjoint during a
 * call. An input or output byte buffer may be NULL when its length is 0.
 * Lengths above PTRDIFF_MAX are invalid. */
typedef struct udlex_native_lexicon udlex_native_lexicon;

/* The returned string is read-only, null-terminated, and must not be freed. */
const char *udlex_native_version(void);

/* Open a compiled lexicon directory using its UTF-8 path, without a terminator.
 * On failure, a valid out pointer is set to NULL. Keep dictionary files
 * immutable from the start of this call until every handle is closed. */
int32_t udlex_native_open(const uint8_t *path, size_t path_len,
                          udlex_native_lexicon **out);

/* Open a language in a bundle. Both strings are UTF-8 without terminators.
 * Set use_rules to 0 to disable suffix guesses. Ownership matches native_open. */
int32_t udlex_native_load(const uint8_t *bundle, size_t bundle_len,
                          const uint8_t *language, size_t language_len,
                          uint8_t use_rules, udlex_native_lexicon **out);

/* Close a live handle once. NULL is allowed. All reads must finish first. */
int32_t udlex_native_close(udlex_native_lexicon *lexicon);

/* Concurrent reads on a live handle may use separate buffers. Input bytes
 * and the result are UTF-8 without terminators. A nonzero sentence_initial
 * marks a sentence boundary. Unknown words can be returned unchanged.
 * Pass NULL with capacity 0 to query the size. On OK, out_len is the number
 * of bytes written. On MORE_SPACE, it is the required capacity and out is
 * untouched. Other errors leave out untouched and a valid out_len set to 0. */
int32_t udlex_native_lemma(const udlex_native_lexicon *lexicon,
                           const uint8_t *token, size_t token_len,
                           uint8_t sentence_initial, uint8_t *out,
                           size_t out_cap, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* UDLEX_NATIVE_H */
