#include "sam/model_cache.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
int main() {
    using namespace photara::sam;
    try {
        std::set<std::string> ids, names;
        for (const auto& model : mask_model_catalog()) {
            if (!ids.insert(model.id).second || !names.insert(model.filename).second ||
                !model.bytes || !model.licence_required || find_mask_model(model.id) != &model)
                throw std::runtime_error("Invalid model catalog");
            if (model.kind == MaskModelKind::text_detector &&
                std::filesystem::path(model.filename).extension() != ".safetensors")
                throw std::runtime_error("GGML text detection needs safetensors");
        }
        const auto root = std::filesystem::temp_directory_path() / "photara-mask-cache-test";
        std::filesystem::create_directories(root);
        const auto agreements = root / "agreements";
        const auto& sam2 = model_license(MaskModelKind::sam2);
        const auto& detector = model_license(MaskModelKind::text_detector);
        if (license_accepted(sam2, agreements)) throw std::runtime_error("Unexpected preaccepted terms");
        accept_model_license(sam2, agreements);
        if (!license_accepted(sam2, agreements) || license_accepted(detector, agreements))
            throw std::runtime_error("Model family agreements are not independent");
        auto revised = sam2;
        revised.revision = "changed-test-revision";
        if (license_accepted(revised, agreements)) throw std::runtime_error("Stale agreement accepted");
        { std::ofstream out(agreements / "sam2.agreement"); out << "sam2"; }
        if (license_accepted(sam2, agreements)) throw std::runtime_error("Corrupt agreement accepted");
        std::filesystem::remove(agreements / "sam2.agreement");
        std::filesystem::remove(agreements);
        const auto renamed = root / "renamed-checkpoint";
        { std::ofstream out(renamed, std::ios::binary);
          const unsigned char header[24] = {0x32,0x6d,0x61,0x73,1,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,2,0,0,0};
          out.write(reinterpret_cast<const char*>(header),sizeof(header)); }
        if (std::string(segmentation_license(renamed).id) != "edgetam")
            throw std::runtime_error("Renamed checkpoint bypasses its family agreement");
        std::filesystem::remove(renamed);
        const auto path = root / "partial.ggml";
        { std::ofstream out(path, std::ios::binary); out << "1234"; }
        const bool valid = model_file_ready(path, 4) && !model_file_ready(path, 8) &&
            resolve_model(path) == path && resolve_model(root / "missing.ggml").empty() &&
            resolve_model({}).empty() &&
            !find_mask_model("unknown-model");
        std::filesystem::remove(path);
        std::filesystem::remove(root);
        if (!valid) throw std::runtime_error("Partial checkpoint accepted or model resolution failed");
        std::cout << "mask model cache contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
