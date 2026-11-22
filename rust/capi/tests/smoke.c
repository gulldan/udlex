#include "udlex.h"

#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  assert(argc == 3);
  assert(strcmp(udlex_native_version(), argv[2]) == 0);

  udlex_native_lexicon *lexicon = NULL;
  assert(udlex_native_open(NULL, 0, &lexicon) == UDLEX_NATIVE_INVALID_ARGUMENT);
  assert(lexicon == NULL);
  const uint8_t *directory = (const uint8_t *)argv[1];
  assert(udlex_native_open(directory, strlen(argv[1]), &lexicon) == UDLEX_NATIVE_OK);
  assert(lexicon != NULL);

  const uint8_t cats[] = {'c', 'a', 't', 's'};
  uint8_t output[8];
  size_t length = 0;
  memset(output, '!', sizeof output);
  assert(udlex_native_lemma(lexicon, cats, sizeof cats, 0, output, 1, &length) ==
         UDLEX_NATIVE_MORE_SPACE);
  assert(length == 3 && memcmp(output, "!!!!!!!!", sizeof output) == 0);
  assert(udlex_native_lemma(lexicon, cats, sizeof cats, 0, NULL, 0, &length) ==
         UDLEX_NATIVE_MORE_SPACE);
  assert(length == 3);
  assert(udlex_native_lemma(lexicon, cats, sizeof cats, 0, output, sizeof output,
                            &length) == UDLEX_NATIVE_OK);
  assert(length == 3 && memcmp(output, "cat!", 4) == 0);

  const uint8_t unknown[] = {'u', 'n', 'k', 'n', 'o', 'w', 'n'};
  assert(udlex_native_lemma(lexicon, unknown, sizeof unknown, 0, output,
                            sizeof output, &length) == UDLEX_NATIVE_OK);
  assert(length == sizeof unknown && memcmp(output, unknown, length) == 0);

  const uint8_t invalid[] = {0xff};
  assert(udlex_native_lemma(lexicon, invalid, sizeof invalid, 0, output,
                            sizeof output, &length) == UDLEX_NATIVE_INVALID_UTF8);
  assert(length == 0 && memcmp(output, unknown, sizeof unknown) == 0);
  assert(udlex_native_lemma(lexicon, cats, SIZE_MAX, 0, output, sizeof output,
                            &length) == UDLEX_NATIVE_INVALID_ARGUMENT);
  assert(length == 0 && memcmp(output, unknown, sizeof unknown) == 0);

  assert(udlex_native_close(lexicon) == UDLEX_NATIVE_OK);
  assert(udlex_native_close(NULL) == UDLEX_NATIVE_OK);

  /* The test fixture is installed as <bundle>/eng, just like an archive. */
  const char *slash = strrchr(argv[1], '/');
  assert(slash != NULL && slash != argv[1]);
  size_t bundle_len = (size_t)(slash - argv[1]);
  const uint8_t *language = (const uint8_t *)(slash + 1);
  for (uint8_t rules = 0; rules <= 1; ++rules) {
    assert(udlex_native_load(directory, bundle_len, language, strlen(slash + 1),
                             rules, &lexicon) == UDLEX_NATIVE_OK);
    assert(udlex_native_lemma(lexicon, cats, sizeof cats, 0, output, sizeof output,
                              &length) == UDLEX_NATIVE_OK);
    assert(length == 3 && memcmp(output, "cat", 3) == 0);
    assert(udlex_native_close(lexicon) == UDLEX_NATIVE_OK);
  }
  assert(udlex_native_load(directory, bundle_len, (const uint8_t *)"absent", 6,
                           1, &lexicon) == UDLEX_NATIVE_OPEN_FAILED);
  assert(lexicon == NULL);

  return 0;
}
