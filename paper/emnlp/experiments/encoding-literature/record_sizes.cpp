// Size and round-trip experiment for the occupied long slots of a format-5 exact table.
// Input is an intact exact.hybrid on stdin. The Python driver verifies its digest.
#include "fsst.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

using Byte = unsigned char;
using Bytes = std::vector<Byte>;
static_assert(std::endian::native == std::endian::little);

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
static T word(const Bytes& bytes, size_t pos) {
    check(pos <= bytes.size() && sizeof(T) <= bytes.size() - pos, "truncated word");
    T value;
    std::memcpy(&value, bytes.data() + pos, sizeof(value));
    return value;
}

// Fit one 8-bit codebook per table and visit each record independently through the
// re-imported decoder. Empty hash slots need only their existing pair of equal offsets.
static void measure(const char* name, const Bytes& bytes, const std::vector<size_t>& starts,
                    const fsst_options_t& options) {
    std::vector<const Byte*> records;
    std::vector<size_t> lengths;
    for (size_t i = 1; i < starts.size(); ++i) {
        if (starts[i] == starts[i - 1]) continue;
        records.push_back(bytes.data() + starts[i - 1]);
        lengths.push_back(starts[i] - starts[i - 1]);
    }
    if (records.empty()) {
        std::cout << ' ' << name << "=0";
        return;
    }
    auto* enc = Optfsst_create(records.size(), lengths.data(), records.data(), 0, &options);
    check(enc != nullptr, "encoder initialization failed");
    std::array<Byte, FSST_MAXHEADER> table{};
    const auto table_bytes = fsst_export(enc, table.data());
    fsst_decoder_t decoder{};
    check(table_bytes == fsst_import(&decoder, table.data()), "codebook round-trip failed");

    size_t coded_bytes = 0;
    constexpr size_t batch = 4096;
    std::vector<size_t> sizes(batch);
    std::vector<Byte*> outputs(batch);
    Bytes scratch(*std::max_element(lengths.begin(), lengths.end()) + 32);
    Bytes coded;
    for (size_t begin = 0; begin < records.size(); begin += batch) {
        const size_t count = std::min(batch, records.size() - begin);
        size_t raw_bytes = 0;
        for (size_t i = begin; i < begin + count; ++i) raw_bytes += lengths[i];
        coded.resize(2 * raw_bytes + 4096);
        const auto done = Optfsst_compress(enc, count, lengths.data() + begin, records.data() + begin,
                                          coded.size(), coded.data(), sizes.data(), outputs.data(), &options);
        check(done == count, "incomplete compression batch");
        for (size_t i = 0; i < count; ++i) {
            const size_t restored = fsst_decompress(&decoder, sizes[i], outputs[i], scratch.size(), scratch.data());
            check(restored == lengths[begin + i] &&
                  std::memcmp(scratch.data(), records[begin + i], restored) == 0, "record round-trip failed");
            coded_bytes += sizes[i];
        }
    }
    fsst_destroy(enc);
    check(coded_bytes <= std::numeric_limits<uint32_t>::max(), "encoded offsets exceed u32");
    std::cout << ' ' << name << '=' << coded_bytes + table_bytes;
}

int main() try {
    std::ios::sync_with_stdio(false);
    Bytes input(std::istreambuf_iterator<char>(std::cin), {});
    check(input.size() >= 128 && std::memcmp(input.data(), "UDLXHYB1", 8) == 0 &&
          word<uint32_t>(input, 8) == 8, "expected format-5 hybrid");
    const size_t long_count = word<uint64_t>(input, 24);
    const size_t buckets = word<uint64_t>(input, 40);
    const size_t slots = word<uint64_t>(input, 48);
    const size_t da_bytes = word<uint64_t>(input, 56);
    const size_t blob_bytes = word<uint64_t>(input, 64);
    auto align16 = [](size_t at) { return (at + 15) & ~size_t(15); };
    const size_t offsets = align16(align16(128 + da_bytes) + 2 * buckets);
    const size_t blob = align16(offsets + 4 * (slots + 1));
    check(blob <= input.size() && input.size() - blob == blob_bytes, "incorrect blob range");

    std::vector<size_t> original{blob}, edits{0};
    Bytes edit_blob;
    size_t seen = 0;
    check(word<uint32_t>(input, offsets) == 0, "incorrect first offset");
    for (size_t i = 1; i <= slots; ++i) {
        const size_t start = original.back();
        const size_t end = blob + word<uint32_t>(input, offsets + 4 * i);
        check(start <= end && end <= input.size(), "incorrect slot range");
        if (end != start) {
            const size_t form = word<uint16_t>(input, start);
            const size_t lemma = word<uint16_t>(input, start + 2);
            check(form > 8 && end - start == 4 + form + lemma, "incorrect slot payload");
            const Byte* key = input.data() + start + 4;
            const Byte* value = key + form;
            const size_t keep = std::mismatch(key, key + form, value, value + lemma).first - key;
            const uint16_t cut = static_cast<uint16_t>(form - keep);
            const size_t at = edit_blob.size();
            edit_blob.insert(edit_blob.end(), input.data() + start, input.data() + start + 2);
            edit_blob.push_back(static_cast<Byte>(cut));
            edit_blob.push_back(static_cast<Byte>(cut >> 8));
            edit_blob.insert(edit_blob.end(), key, key + form);
            edit_blob.insert(edit_blob.end(), value + keep, value + lemma);
            const size_t saved_form = word<uint16_t>(edit_blob, at);
            const size_t saved_cut = word<uint16_t>(edit_blob, at + 2);
            const Byte* saved_key = edit_blob.data() + at + 4;
            const size_t tail = edit_blob.size() - at - 4 - saved_form;
            check(saved_form == form && saved_cut <= form && form - saved_cut + tail == lemma &&
                  std::memcmp(saved_key, key, form) == 0 &&
                  std::memcmp(saved_key, value, form - saved_cut) == 0 &&
                  std::memcmp(saved_key + form, value + form - saved_cut, tail) == 0, "edit round-trip failed");
            ++seen;
        }
        original.push_back(end);
        edits.push_back(edit_blob.size());
    }
    check(original.back() == input.size() && seen == long_count, "incomplete slot traversal");
    std::cout << "long_keys=" << seen << " prefix=" << blob << " raw=" << blob_bytes << " edit=" << edit_blob.size();
    fsst_options_t base{0}, opt{FSST_OPT_DP_TRAIN | FSST_OPT_DP_ENCODE | FSST_OPT_TRIPLES | FSST_OPT_PRUNE};
    measure("fsst_raw", input, original, base);
    measure("opt_raw", input, original, opt);
    measure("fsst_edit", edit_blob, edits, base);
    measure("opt_edit", edit_blob, edits, opt);
    std::cout << '\n';
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}
