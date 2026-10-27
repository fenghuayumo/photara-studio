#include "splat/visualize.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    using namespace photara::splat;
    const auto path = std::filesystem::temp_directory_path() / "photara_preview_clip_sidecar_test";
    VisualizeOptions sent; sent.rasterization.near_plane = 0.0125F;
    sent.channel = VisualizationChannel::depth;
    VisualizeOptions received; std::uint64_t revision = 0;
    if (!write_visualization_sidecar(path, sent, 7) ||
        !load_visualization_sidecar(path, received, revision) || revision != 7 ||
        std::abs(received.rasterization.near_plane - 0.0125F) > 1e-6F ||
        received.channel != VisualizationChannel::depth) return 1;
    // Legacy files keep the caller's policy rather than silently resetting it.
    { std::ofstream old(path); old << "8\nsplat\n2.5\n2.8\n0 1 0 1 0.1 10\n"; }
    received.rasterization.near_plane = 0.33F;
    if (!load_visualization_sidecar(path, received, revision) || revision != 8 ||
        std::abs(received.rasterization.near_plane - 0.33F) > 1e-6F) return 2;
    // An incomplete new-format write must not be mistaken for a legacy file.
    { std::ofstream partial(path); partial << "9\nsplat+near\n2.5\n2.8\n0 1 0 1 0.1 10\n"; }
    if (load_visualization_sidecar(path, received, revision) || revision != 8 ||
        std::abs(received.rasterization.near_plane - 0.33F) > 1e-6F) return 3;
    // An invalid update must also leave the last complete state intact.
    { std::ofstream bad(path); bad << "9\nsplat+near\n2.5\n2.8\n0 1 0 1 0.1 10\n-1\n"; }
    if (load_visualization_sidecar(path, received, revision) || revision != 8 ||
        std::abs(received.rasterization.near_plane - 0.33F) > 1e-6F) return 3;
    std::filesystem::remove(path);
    std::cout << "preview near clip sidecar tests passed\n";
}
