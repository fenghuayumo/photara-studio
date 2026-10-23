#pragma once

#include "control/control_protocol.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace editor {
struct App;
}

namespace editor::control {

class ControlBus {
public:
    ControlBus() = default;
    ControlBus(const ControlBus&) = delete;
    ControlBus& operator=(const ControlBus&) = delete;

    void set_session_id(std::string session_id);
    [[nodiscard]] const std::string& session_id() const noexcept {
        return session_id_;
    }

    // Thread-safe. The command is executed on the Studio main thread during
    // pump_main_thread() or pump_post_present().
    std::shared_future<json> submit(
        std::string method, json params, std::string client,
        std::string request_id = {});

    void pump_main_thread(App& app);
    void pump_post_present(App& app);
    void stop();

    [[nodiscard]] std::uint64_t revision() const noexcept {
        return revision_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] const std::string& active_operation() const noexcept {
        return active_operation_;
    }
    [[nodiscard]] const std::string& active_client() const noexcept {
        return active_client_;
    }
    [[nodiscard]] const std::string& active_stage() const noexcept {
        return active_stage_;
    }

    // Begins a job operation and returns its public id. Main thread only.
    std::string begin_operation(
        const std::string& stage, const std::string& client);
    void mark_operation_cancelled() { operation_cancelled_ = true; }

    void push_event(std::string type, json data);
    [[nodiscard]] json events_since(std::uint64_t since) const;
    [[nodiscard]] json recent_events(std::size_t limit) const;
    [[nodiscard]] json operation_status(const App& app, const std::string& id);

private:
    struct Pending {
        std::string method;
        json params;
        std::string client;
        std::string request_id;
        std::shared_ptr<std::promise<json>> promise;
        std::chrono::steady_clock::time_point deadline;
    };
    struct EventRecord {
        std::uint64_t id{};
        std::string type;
        json data;
        std::int64_t time_ms{};
    };

    static void resolve(Pending& pending, json value);
    json execute_pending(App& app, Pending& pending);
    static bool should_defer(const Pending& pending);
    void sync_revision(const App& app);
    void finish_operation_if_done(const App& app);

    mutable std::mutex queue_mutex_;
    std::deque<Pending> main_queue_;
    std::deque<Pending> post_present_queue_;
    bool stopping_{};
    struct RequestRecord {
        std::string signature;
        std::shared_future<json> result;
    };
    std::map<std::string, RequestRecord> requests_;
    std::deque<std::string> request_order_;

    mutable std::mutex event_mutex_;
    std::deque<EventRecord> events_;
    std::uint64_t next_event_id_{1};

    std::atomic<std::uint64_t> revision_{0};
    std::string session_id_;
    std::string active_operation_;
    std::string active_client_;
    std::string active_stage_;
    bool operation_cancelled_{};
    std::uint64_t next_operation_id_{1};
    std::string last_state_key_;
    std::map<std::string, json> completed_operations_;
    std::deque<std::string> completed_order_;
    std::chrono::steady_clock::time_point last_state_event_{};
};

}  // namespace editor::control
