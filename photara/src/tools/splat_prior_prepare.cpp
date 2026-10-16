#include "splat/prior_prepare.hpp"
#include "splat/dataset.hpp"
#include "splat/prior.hpp"
#include "training_data_loader.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 8 && argc != 10) {
            std::cerr << "Usage: photara_splat_prior_prepare dataset images model.gguf output-dir max-resolution tokens eval-split-every [lambda_d lambda_n]\n"
                         "Weights default to zero: no inference or cache creation when both are disabled.\n";
            return 1;
        }
        const float lambda_d = argc == 10 ? std::stof(argv[8]) : 0.F;
        const float lambda_n = argc == 10 ? std::stof(argv[9]) : 0.F;
        if (!std::isfinite(lambda_d) || lambda_d < 0.F || !std::isfinite(lambda_n) || lambda_n < 0.F)
            throw std::invalid_argument("lambda_d and lambda_n must be finite and nonnegative");
        const bool need_depth = lambda_d > 0.F, need_normals = lambda_n > 0.F;
        if (!need_depth && !need_normals) {
            std::cout << "Prior preparation skipped: lambda_d=0 lambda_n=0\n";
            return 0;
        }
        photara::splat::DatasetLoadRequest request;
        request.source = argv[1]; request.image_directory = argv[2];
        auto loaded = photara::splat::load_splat_dataset(request);
        photara::splat::TrainingOptions training;
        training.max_image_dimension = static_cast<unsigned>(std::stoul(argv[5]));
        training.use_source_resolution = true;
        training.undistort_to_pinhole = true;
        training.evaluation_split_every = static_cast<unsigned>(std::stoul(argv[7]));
        training.prior_directory = argv[4];
        training.lambda_d = lambda_d; training.lambda_n = lambda_n;
        photara::splat::prepare_training_priors(loaded.scene, training, argv[3],
            static_cast<unsigned>(std::stoul(argv[6])));
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
