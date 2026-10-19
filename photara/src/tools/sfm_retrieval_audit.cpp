// Read-only candidate retrieval on retained native features. No reference model.
#include "sfm/checkpoint.hpp"
#include "sfm/retrieval.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace photara::sfm;
    try {
        if (argc != 5) throw std::runtime_error(
            "features-checkpoint vocabulary output.csv native|systematic");
        const std::filesystem::path input(argv[1]);
        const std::string policy(argv[4]);
        if (policy != "native" && policy != "systematic")
            throw std::runtime_error("unknown sampling policy");
        if (std::filesystem::exists(argv[3]))
            throw std::runtime_error("output already exists");
        const auto stem = input.stem().string();
        const auto key = std::stoull(stem.substr(stem.find('-') + 1), nullptr, 16);
        CheckpointOptions checkpoint;
        checkpoint.directory = input.parent_path();
        checkpoint.write = false;
        Scene scene;
        if (!CheckpointStore(checkpoint).load_scene(CheckpointStage::features, key, scene))
            throw std::runtime_error("cannot read features checkpoint");
        RetrievalOptions options;
        options.vocabulary_path = argv[2];
        if (!std::filesystem::exists(options.vocabulary_path))
            throw std::runtime_error("vocabulary must already exist");
        options.systematic_sampling = policy == "systematic";
        const auto pairs = retrieve_image_pairs(scene.images, options);
        std::ofstream output(argv[3]);
        output << "id1,id2,name1,name2,score\n" << std::setprecision(9);
        for (const auto& pair : pairs)
            output << pair.first << ',' << pair.second << ','
                   << std::quoted(scene.images[pair.first].path.filename().string()) << ','
                   << std::quoted(scene.images[pair.second].path.filename().string()) << ',' << pair.score << '\n';
        if (!output) throw std::runtime_error("cannot write output");
        std::cout << "policy=" << policy << " images=" << scene.images.size()
                  << " pairs=" << pairs.size() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
