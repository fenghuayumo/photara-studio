#pragma once

#if defined(PHOTARA_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace photara::features {

struct OnnxSessionConfig {
    std::filesystem::path model_path;
    bool use_cuda{true};
    bool allow_cpu_fallback{true};
    const char* env_name{"Photara.ONNX"};
};

// Thin owning wrapper around Ort::Env + Ort::Session (not thread-safe to Run).
class OnnxSession {
public:
    explicit OnnxSession(OnnxSessionConfig config) : config_(std::move(config)) {
        if (config_.model_path.empty())
            throw std::invalid_argument("ONNX model path is required");
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.use_cuda) {
            try {
                OrtCUDAProviderOptions cuda{};
                session_options_.AppendExecutionProvider_CUDA(cuda);
            } catch (const Ort::Exception&) {
                if (!config_.allow_cpu_fallback) throw;
            }
        }
#if defined(_WIN32)
        session_ = std::make_unique<Ort::Session>(
            env_, config_.model_path.wstring().c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(
            env_, config_.model_path.string().c_str(), session_options_);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t num_inputs = session_->GetInputCount();
        const std::size_t num_outputs = session_->GetOutputCount();
        input_names_.reserve(num_inputs);
        input_name_ptrs_.reserve(num_inputs);
        for (std::size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name.get());
            input_name_ptrs_.push_back(input_names_.back().c_str());
        }
        output_names_.reserve(num_outputs);
        output_name_ptrs_.reserve(num_outputs);
        for (std::size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(name.get());
            output_name_ptrs_.push_back(output_names_.back().c_str());
        }
    }

    [[nodiscard]] Ort::Session& session() { return *session_; }
    [[nodiscard]] const Ort::Session& session() const { return *session_; }
    [[nodiscard]] std::mutex& mutex() const { return mutex_; }
    [[nodiscard]] const std::vector<const char*>& input_names() const {
        return input_name_ptrs_;
    }
    [[nodiscard]] const std::vector<const char*>& output_names() const {
        return output_name_ptrs_;
    }
    [[nodiscard]] const std::vector<std::string>& input_name_strings() const {
        return input_names_;
    }
    [[nodiscard]] const std::vector<std::string>& output_name_strings() const {
        return output_names_;
    }
    [[nodiscard]] const OnnxSessionConfig& config() const { return config_; }

private:
    OnnxSessionConfig config_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Photara.ONNX"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    mutable std::mutex mutex_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
};

}  // namespace photara::features
#endif
