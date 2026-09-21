#include "splat/dataset.hpp"
#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "core/logging.hpp"
#include <iostream>
#include <stdexcept>

// Fixed-camera regression: retain observations in memory across mesh variants.
// Both variants consume exactly the same fusion result and observations.
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: mvs_compare dataset images output_dir");
        photara::core::Logger::instance().configure(argv[3]);
        photara::splat::DatasetLoadRequest request;
        request.source = argv[1];
        request.image_directory = argv[2];
        auto loaded = photara::splat::load_splat_dataset(request);
        auto& scene = loaded.scene;
        photara::mvs::DensifyOptions options;
        photara::mvs::apply_quality_preset(options, photara::mvs::DensifyQuality::default_quality);
        options.mask_dir = request.image_directory.parent_path() / "masks";
        options.build_mesh = false;
        options.mesh_max_points = 500000;
        photara::mvs::prepare_imported_scene(scene, options);
        const std::filesystem::path output = argv[3];
        std::filesystem::create_directories(output);
        photara::mvs::densify(scene, options);
        photara::mvs::save_dense_ply(scene.dense_cloud, output / "dense.ply");
        for (int variant = 0; variant < 2; ++variant) {
            options.mesh_use_free_space_support = variant != 0;
            photara::mvs::reconstruct_mesh(scene, options);
            const auto name = variant ? "weak" : "no_weak";
            photara::mvs::save_mesh_ply(scene.mesh, output / (std::string(name) + ".ply"));
            std::cout << name << " vertices=" << scene.mesh.vertices.size() << " faces=" << scene.mesh.faces.size() << std::endl;
        }
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
