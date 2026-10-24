#include "tokenizer.hpp"

#include <fstream>
#include <stdexcept>

namespace photara::inference::detail {
namespace {

std::uint32_t next_code_point(const std::string& text, std::size_t& offset) {
    const auto byte = static_cast<unsigned char>(text[offset++]);
    if (byte < 0x80U) return byte;
    std::uint32_t code = byte;
    std::size_t extra = 0;
    if ((byte & 0xE0U) == 0xC0U) {
        code &= 0x1FU;
        extra = 1;
    } else if ((byte & 0xF0U) == 0xE0U) {
        code &= 0x0FU;
        extra = 2;
    } else if ((byte & 0xF8U) == 0xF0U) {
        code &= 0x07U;
        extra = 3;
    } else {
        return 0xFFFDU;
    }
    if (offset + extra > text.size()) return 0xFFFDU;
    for (std::size_t i = 0; i < extra; ++i) {
        const auto continuation =
            static_cast<unsigned char>(text[offset++]);
        if ((continuation & 0xC0U) != 0x80U) return 0xFFFDU;
        code = (code << 6U) | (continuation & 0x3FU);
    }
    return code;
}

void put_code_point(std::string& text, const std::uint32_t code) {
    if (code < 0x80U) {
        text.push_back(static_cast<char>(code));
    } else if (code < 0x800U) {
        text.push_back(static_cast<char>(0xC0U | (code >> 6U)));
        text.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    } else if (code < 0x10000U) {
        text.push_back(static_cast<char>(0xE0U | (code >> 12U)));
        text.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
        text.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    } else {
        text.push_back(static_cast<char>(0xF0U | (code >> 18U)));
        text.push_back(static_cast<char>(0x80U | ((code >> 12U) & 0x3FU)));
        text.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
        text.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
    }
}

bool is_space(const std::uint32_t code) {
    return code == ' ' || code == '\t' || code == '\n' || code == '\r' ||
           code == 0xA0U || code == 0x3000U ||
           (code >= 0x2000U && code <= 0x200AU);
}

bool is_control(const std::uint32_t code) {
    return (code < 0x20U && code != '\t' && code != '\n' && code != '\r') ||
           (code >= 0x7FU && code < 0xA0U);
}

bool is_punctuation(const std::uint32_t code) {
    if ((code >= 33U && code <= 47U) || (code >= 58U && code <= 64U) ||
        (code >= 91U && code <= 96U) || (code >= 123U && code <= 126U))
        return true;
    return (code >= 0x2010U && code <= 0x2027U) ||
           (code >= 0x3001U && code <= 0x3003U) ||
           (code >= 0x300CU && code <= 0x3011U) ||
           (code >= 0xFF01U && code <= 0xFF0FU) || code == 0xA1U ||
           code == 0xBFU || code == 0xABU || code == 0xBBU;
}

bool is_cjk(const std::uint32_t code) {
    return (code >= 0x4E00U && code <= 0x9FFFU) ||
           (code >= 0x3400U && code <= 0x4DBFU) ||
           (code >= 0x20000U && code <= 0x2A6DFU) ||
           (code >= 0xF900U && code <= 0xFAFFU) ||
           (code >= 0x2F800U && code <= 0x2FAFFU);
}

std::uint32_t fold(const std::uint32_t code) {
    if (code >= 'A' && code <= 'Z') return code + 32U;
    if (code < 0xC0U || code >= 0x180U) return code;
    static const char base[] =
        "aaaaaa*ceeeeiiii*nooooo**uuuuy**aaaaaa*ceeeeiiii*nooooo**uuuuy*y"
        "aaaaaaccccccccdd**eeeeeeeeeegggggggghh**iiiiiiiii***jjkk*llllll****nnnnnn***oooooo"
        "**rrrrrrsssssssstttt**uuuuuuuuuuuuwwyyyzzzzzz*";
    static_assert(sizeof(base) == 0x180U - 0xC0U + 1U);
    const char folded = base[code - 0xC0U];
    if (folded != '*') return static_cast<std::uint32_t>(folded);
    if (code <= 0xDEU && code != 0xD7U) return code + 0x20U;
    if (code >= 0x100U && (code & 1U) == 0U) return code + 1U;
    return code;
}

}  // namespace

void GroundingDinoTokenizer::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "Cannot open the Grounding DINO vocabulary: " + path.string());
    vocabulary_.clear();
    by_id_.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        vocabulary_.emplace(line, static_cast<std::int32_t>(by_id_.size()));
        by_id_.push_back(line);
    }
    if (by_id_.size() <= static_cast<std::size_t>(k_question) ||
        by_id_[static_cast<std::size_t>(k_cls)] != "[CLS]" ||
        by_id_[static_cast<std::size_t>(k_sep)] != "[SEP]" ||
        by_id_[static_cast<std::size_t>(k_period)] != ".")
        throw std::runtime_error(
            "The Grounding DINO vocabulary is not BERT uncased: " +
            path.string());
    const auto unknown = vocabulary_.find("[UNK]");
    unknown_ = unknown == vocabulary_.end() ? 100 : unknown->second;
}

std::int32_t GroundingDinoTokenizer::unknown() const noexcept {
    return unknown_;
}

void GroundingDinoTokenizer::wordpiece(
    const std::string& word, std::vector<std::int32_t>& output) const {
    if (word.size() > 100) {
        output.push_back(unknown());
        return;
    }
    std::size_t start = 0;
    while (start < word.size()) {
        std::size_t end = word.size();
        std::int32_t found = -1;
        while (start < end) {
            std::string piece = word.substr(start, end - start);
            if (start > 0) piece.insert(0, "##");
            const auto item = vocabulary_.find(piece);
            if (item != vocabulary_.end()) {
                found = item->second;
                break;
            }
            do {
                --end;
            } while (end > start &&
                     (static_cast<unsigned char>(word[end]) & 0xC0U) == 0x80U);
        }
        if (found < 0) {
            output.push_back(unknown());
            return;
        }
        output.push_back(found);
        start = end;
    }
}

std::vector<std::int32_t> GroundingDinoTokenizer::encode(
    const std::string& text) const {
    std::vector<std::string> words(1);
    const auto split = [&] {
        if (!words.back().empty()) words.emplace_back();
    };
    for (std::size_t i = 0; i < text.size();) {
        const auto code = next_code_point(text, i);
        if (code == 0U || code == 0xFFFDU || is_control(code)) continue;
        if (is_space(code)) {
            split();
            continue;
        }
        if (is_cjk(code) || is_punctuation(code)) {
            split();
            put_code_point(words.back(), fold(code));
            split();
            continue;
        }
        put_code_point(words.back(), fold(code));
    }
    std::vector<std::int32_t> ids{k_cls};
    ids.reserve(text.size() / 3U + 2U);
    for (const std::string& word : words)
        if (!word.empty()) wordpiece(word, ids);
    ids.push_back(k_sep);
    return ids;
}

}  // namespace photara::inference::detail
