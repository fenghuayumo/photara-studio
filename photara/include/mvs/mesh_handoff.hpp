#pragma once

#include "mvs/types.hpp"

#include <chrono>
#include <filesystem>
#include <memory>

namespace photara::mvs {

// Publishes an immutable binary mesh through pagefile/POSIX shared memory.
// Keep the owner alive until the reader has opened its own mapping. Mesh data
// is copied directly into the mapping; saving can keep using the source mesh.
class MeshHandoff {
public:
    MeshHandoff(const Mesh& mesh, const std::filesystem::path& ready);
    ~MeshHandoff();
    MeshHandoff(const MeshHandoff&) = delete;
    MeshHandoff& operator=(const MeshHandoff&) = delete;
    bool wait_for_reader(std::chrono::milliseconds timeout) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Validates the mapped binary layout before allocating, then returns an owned
// mesh. Opening the mapping is acknowledged before decoding, so the publisher
// can exit while the reader finishes preparing the viewport.
Mesh receive_mesh_handoff(const std::filesystem::path& ready);

}  // namespace photara::mvs
