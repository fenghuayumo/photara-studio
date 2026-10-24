#include "sam/model_cache.hpp"
#if defined(PHOTARA_HAS_INFERENCE)
#include "inference/model_cache.hpp"
#endif

#include <algorithm>
#include <array>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace photara::sam {
namespace {

constexpr std::uint64_t sam2_large_bytes = 450'930'592ULL;
constexpr std::uint64_t sam2_base_plus_bytes = 163'303'648ULL;
constexpr std::uint64_t sam2_small_bytes = 93'559'264ULL;
constexpr std::uint64_t sam2_tiny_bytes = 79'320'544ULL;
constexpr std::uint64_t sam3_f16_bytes = 1'837'873'470ULL;
constexpr std::uint64_t detector_tiny_bytes = 689'359'096ULL;
constexpr std::uint64_t detector_base_bytes = 933'400'872ULL;
constexpr std::uint64_t detector_vocab_bytes = 231'508ULL;

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return {};
    return std::filesystem::path(value);
}

std::filesystem::path local_app_data() {
#if defined(_WIN32)
    return env_path("LOCALAPPDATA");
#else
    if (const auto xdg = env_path("XDG_CACHE_HOME"); !xdg.empty()) return xdg;
    const auto home = env_path("HOME");
    return home.empty() ? std::filesystem::path{} : home / ".cache";
#endif
}

std::filesystem::path roaming_app_data() {
#if defined(_WIN32)
    return env_path("APPDATA");
#else
    if (const auto xdg = env_path("XDG_CONFIG_HOME"); !xdg.empty()) return xdg;
    const auto home = env_path("HOME");
    return home.empty() ? std::filesystem::path{} : home / ".config";
#endif
}

}  // namespace

const std::vector<MaskModelEntry>& mask_model_catalog() {
    static const std::vector<MaskModelEntry> catalog = {
        {"sam3-q4_0", "SAM 3 (q4_0)", k_model_file,
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-q4_0.ggml",
         k_model_bytes, MaskModelKind::sam3, true},
        {"sam3-f16", "SAM 3 (f16)", "sam3-f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-f16.ggml",
         sam3_f16_bytes, MaskModelKind::sam3, true},
        {"sam2.1-large", "SAM 2.1 Large", "sam2.1_hiera_large_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/"
         "sam2.1_hiera_large_f16.ggml",
         sam2_large_bytes, MaskModelKind::sam2, true},
        {k_default_sam2_model, "SAM 2.1 Base Plus",
         "sam2.1_hiera_base_plus_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/"
         "sam2.1_hiera_base_plus_f16.ggml",
         sam2_base_plus_bytes, MaskModelKind::sam2, true},
        {"sam2.1-small", "SAM 2.1 Small", "sam2.1_hiera_small_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/"
         "sam2.1_hiera_small_f16.ggml",
         sam2_small_bytes, MaskModelKind::sam2, true},
        {"sam2.1-tiny", "SAM 2.1 Tiny", "sam2.1_hiera_tiny_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/"
         "sam2.1_hiera_tiny_f16.ggml",
         sam2_tiny_bytes, MaskModelKind::sam2, true},
        {"sam2-tiny", "SAM2 Tiny (f16)", "sam2_hiera_tiny_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_tiny_f16.ggml",
         79254304ULL, MaskModelKind::sam2, true},
        {"sam2-tiny-q8_0", "SAM2 Tiny (q8_0)", "sam2_hiera_tiny_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_tiny_q8_0.ggml",
         42576992ULL, MaskModelKind::sam2, true},
        {"sam2-small", "SAM2 Small (f16)", "sam2_hiera_small_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_small_f16.ggml",
         93493024ULL, MaskModelKind::sam2, true},
        {"sam2-small-q8_0", "SAM2 Small (q8_0)", "sam2_hiera_small_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_small_q8_0.ggml",
         50180192ULL, MaskModelKind::sam2, true},
        {"sam2-base-plus", "SAM2 Base Plus (f16)", "sam2_hiera_base_plus_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_base_plus_f16.ggml",
         163237408ULL, MaskModelKind::sam2, true},
        {"sam2-base-plus-q8_0", "SAM2 Base Plus (q8_0)", "sam2_hiera_base_plus_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_base_plus_q8_0.ggml",
         87723296ULL, MaskModelKind::sam2, true},
        {"sam2-large", "SAM2 Large (f16)", "sam2_hiera_large_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_large_f16.ggml",
         450864352ULL, MaskModelKind::sam2, true},
        {"sam2-large-q8_0", "SAM2 Large (q8_0)", "sam2_hiera_large_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2_hiera_large_q8_0.ggml",
         241222880ULL, MaskModelKind::sam2, true},
        {"sam2.1-tiny-q8_0", "SAM2.1 Tiny (q8_0)", "sam2.1_hiera_tiny_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_tiny_q8_0.ggml",
         42595104ULL, MaskModelKind::sam2, true},
        {"sam2.1-small-q8_0", "SAM2.1 Small (q8_0)", "sam2.1_hiera_small_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_small_q8_0.ggml",
         50198304ULL, MaskModelKind::sam2, true},
        {"sam2.1-base-plus-q8_0", "SAM2.1 Base Plus (q8_0)", "sam2.1_hiera_base_plus_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_base_plus_q8_0.ggml",
         87741408ULL, MaskModelKind::sam2, true},
        {"sam2.1-large-q8_0", "SAM2.1 Large (q8_0)", "sam2.1_hiera_large_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_large_q8_0.ggml",
         241240992ULL, MaskModelKind::sam2, true},
        {"edgetam-f16", "EdgeTAM (f16)", "edgetam_f16.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/edgetam_f16.ggml",
         29011488ULL, MaskModelKind::edgetam, true},
        {"edgetam-q8_0", "EdgeTAM (q8_0)", "edgetam_q8_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/edgetam_q8_0.ggml",
         20005216ULL, MaskModelKind::edgetam, true},
        {"edgetam-q4_0", "EdgeTAM (q4_0)", "edgetam_q4_0.ggml",
         "https://huggingface.co/PABannier/sam3.cpp/resolve/main/edgetam_q4_0.ggml",
         15761120ULL, MaskModelKind::edgetam, true},
        {"grounding-dino-tiny", "Grounding DINO Tiny",
         k_default_text_detector_model,
         "https://huggingface.co/IDEA-Research/grounding-dino-tiny/"
         "resolve/main/model.safetensors",
         detector_tiny_bytes, MaskModelKind::text_detector, true},
        {"grounding-dino-base", "Grounding DINO Base",
         "grounding-dino-base.safetensors",
         "https://huggingface.co/IDEA-Research/grounding-dino-base/"
         "resolve/main/model.safetensors",
         detector_base_bytes, MaskModelKind::text_detector, true},
    };
    return catalog;
}

const MaskModelEntry* find_mask_model(const std::string_view id) {
    const auto& catalog = mask_model_catalog();
    const auto found = std::find_if(
        catalog.begin(), catalog.end(),
        [&](const MaskModelEntry& entry) { return id == entry.id; });
    return found == catalog.end() ? nullptr : &*found;
}

std::filesystem::path user_model_path(const std::string_view filename) {
    if (filename.empty()) return {};
    const std::string name(filename);
    const std::filesystem::path file(name);
#if defined(PHOTARA_HAS_INFERENCE)
    return inference::cached_model_path(name);
#else
    const auto base = local_app_data();
    if (base.empty()) return file;
    return base / "Photara" / "models" / file;
#endif
}

bool model_file_ready(
    const std::filesystem::path& path, const std::uint64_t expected_bytes) {
    const std::uintmax_t minimum =
        expected_bytes == 0 ? 1U : expected_bytes;
#if defined(PHOTARA_HAS_INFERENCE)
    return inference::model_file_ready(path, minimum);
#else
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error))
        return false;
    const auto size = std::filesystem::file_size(path, error);
    return !error && size >= minimum;
#endif
}

std::filesystem::path locate_model(const std::string_view id) {
    const MaskModelEntry* entry = find_mask_model(id);
    if (entry == nullptr) return {};
    if (const auto from_env = env_path("PHOTARA_SAM_MODEL");
        model_file_ready(from_env, entry->bytes))
        return from_env;
    const auto own = user_model_path(entry->filename);
    return model_file_ready(own, entry->bytes)
        ? own
        : std::filesystem::path{};
}

std::filesystem::path resolve_model(const std::filesystem::path& value) {
    if (value.empty()) return {};
    if (const auto* entry = find_mask_model(value.string()))
        return locate_model(entry->id);
    if (model_file_ready(value, 0)) return value;
    if (value.has_parent_path() || value.is_absolute()) return {};
    const auto cached = user_model_path(value.string());
    return model_file_ready(cached, 0) ? cached : std::filesystem::path{};
}

std::filesystem::path locate_text_detector_model() {
    if (const auto from_env = env_path("PHOTARA_TEXT_DETECTOR_MODEL");
        model_file_ready(from_env, detector_tiny_bytes))
        return from_env;
    const auto own = user_model_path(k_default_text_detector_model);
    return model_file_ready(own, detector_tiny_bytes)
        ? own
        : std::filesystem::path{};
}

std::filesystem::path locate_text_detector_vocab() {
    if (const auto from_env = env_path("PHOTARA_TEXT_DETECTOR_VOCAB");
        model_file_ready(from_env, detector_vocab_bytes))
        return from_env;
    const auto own = user_model_path(k_default_text_detector_vocab);
    return model_file_ready(own, detector_vocab_bytes)
        ? own
        : std::filesystem::path{};
}

bool text_detector_ready() {
    return !locate_text_detector_model().empty() &&
           !locate_text_detector_vocab().empty();
}

namespace {
const ModelLicense licenses[] = {
    {"sam3", "SAM 3 License (Meta)", k_license_url, "sam3-terms-v1"},
    {"sam2", "SAM 2 / SAM 2.1 (Apache-2.0)",
     "https://github.com/facebookresearch/sam2/blob/main/LICENSE", "apache-2.0-v1"},
    {"edgetam", "EdgeTAM (Apache-2.0)",
     "https://github.com/facebookresearch/EdgeTAM/blob/main/LICENSE", "apache-2.0-v1"},
    {"grounding-dino", "Grounding DINO (Apache-2.0)",
     "https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE", "apache-2.0-v1"},
    {"moge", "MoGe (MIT)",
     "https://github.com/microsoft/MoGe/blob/main/LICENSE", "mit-v1"},
};
std::string agreement_text(const ModelLicense& license) {
    return std::string("Photara model agreement v1\n") + license.id + "\n" +
        license.url + "\n" + license.revision + "\n";
}
std::filesystem::path agreement_path(const ModelLicense& license,
    const std::filesystem::path& directory) {
    if (!find_model_license(license.id))
        throw std::invalid_argument("Unknown model agreement");
    return directory / (std::string(license.id) + ".agreement");
}
}

const ModelLicense* find_model_license(const std::string_view id) {
    for (const auto& license : licenses) if (id == license.id) return &license;
    return nullptr;
}
const ModelLicense& model_license(const MaskModelKind kind) {
    switch (kind) {
    case MaskModelKind::sam3: return licenses[0];
    case MaskModelKind::sam2: return licenses[1];
    case MaskModelKind::edgetam: return licenses[2];
    case MaskModelKind::text_detector: return licenses[3];
    }
    throw std::invalid_argument("Unknown mask model family");
}
const ModelLicense& moge_model_license() { return licenses[4]; }

const ModelLicense& segmentation_license(const std::filesystem::path& model) {
    std::ifstream input(model, std::ios::binary);
    std::array<unsigned char, 24> header{};
    if (input.read(reinterpret_cast<char*>(header.data()), header.size())) {
        const auto word = [&](int offset) {
            std::uint32_t value = 0;
            for (int i = 0; i < 4; ++i) value |= std::uint32_t(header[offset+i]) << (8*i);
            return value;
        };
        if (word(0) == 0x73616d33U) return licenses[0];
        if (word(0) == 0x73616d32U) return word(20) == 2 ? licenses[2] : licenses[1];
    }
    const auto name = model.filename().string();
    for (const auto& entry : mask_model_catalog())
        if ((name == entry.filename || name == entry.id) && entry.kind != MaskModelKind::text_detector)
            return model_license(entry.kind);
    throw std::invalid_argument("Cannot identify the checkpoint's model licence. Select a supported mask model.");
}

std::filesystem::path model_agreements_directory() {
    const auto base = roaming_app_data();
    return base.empty() ? std::filesystem::path{} : base / "Photara" / "model-agreements";
}
bool license_accepted(const ModelLicense& license, const std::filesystem::path& directory) {
    if (directory.empty()) return false;
    std::ifstream input(agreement_path(license, directory), std::ios::binary);
    return input && std::string(std::istreambuf_iterator<char>(input), {}) == agreement_text(license);
}
void accept_model_license(const ModelLicense& license, const std::filesystem::path& directory) {
    if (directory.empty()) throw std::runtime_error("User configuration directory is unavailable");
    const auto path = agreement_path(license, directory);
    std::filesystem::create_directories(directory);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << agreement_text(license);
    output.close();
    if (!output) throw std::runtime_error("Could not record model licence acceptance");
}
void require_model_license(const ModelLicense& license) {
    if (!license_accepted(license))
        throw std::runtime_error(std::string("Review and accept the model licence before use: ") +
            license.label + " (" + license.url + "). In Studio open the model agreement dialog; "
            "on the CLI pass --accept-model-licenses " + license.id);
}
bool license_accepted() { return license_accepted(licenses[0]); }
void accept_license() { accept_model_license(licenses[0]); }

}  // namespace photara::sam
