#include "control/control_bus.hpp"

#include "control/control_commands.hpp"
#include "control/control_state.hpp"

#include "app.hpp"

#include <chrono>
#include <exception>
#include <mutex>
#include <utility>

namespace editor::control {
namespace {

std::int64_t now_milliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

void ControlBus::set_session_id(std::string session_id) {
    session_id_ = std::move(session_id);
}

std::shared_future<json> ControlBus::submit(
    std::string method, json params, std::string client,
    std::string request_id) {
    auto promise = std::make_shared<std::promise<json>>();
    std::shared_future<json> future = promise->get_future().share();
    Pending pending{
        std::move(method), std::move(params), std::move(client),
        std::move(request_id), std::move(promise),
        std::chrono::steady_clock::now() + std::chrono::seconds(5)};
    {
        std::lock_guard lock(queue_mutex_);
        const std::string key = json::array({pending.client, pending.request_id}).dump();
        const std::string signature = json::array({pending.method, pending.params}).dump();
        if (!pending.request_id.empty()) {
            const auto found = requests_.find(key);
            if (found != requests_.end()) {
                if (found->second.signature == signature) return found->second.result;
                resolve(pending, failure("CONFLICT", "request_id was used with different arguments"));
                return future;
            }
        }
        if (stopping_) {
            resolve(pending, failure("SHUTDOWN", "Studio control is stopping"));
            return future;
        }
        if (main_queue_.size() + post_present_queue_.size() >= 256) {
            resolve(pending, failure("BUSY", "Control queue is full"));
            return future;
        }
        if (!pending.request_id.empty()) {
            // Keep a bounded retry window. Never evict an in-flight request.
            while (requests_.size() >= 1024 && !request_order_.empty()) {
                const auto oldest = requests_.find(request_order_.front());
                if (oldest->second.result.wait_for(std::chrono::seconds(0)) !=
                    std::future_status::ready) break;
                requests_.erase(oldest);
                request_order_.pop_front();
            }
            if (requests_.size() >= 1024) {
                resolve(pending, failure("BUSY", "Retry cache is full"));
                return future;
            }
            requests_.emplace(key, RequestRecord{signature, future});
            request_order_.push_back(key);
        }
        main_queue_.push_back(std::move(pending));
    }
    return future;
}

json ControlBus::execute_pending(App& app, Pending& pending) {
    if (std::chrono::steady_clock::now() >= pending.deadline)
        return failure("EXPIRED", "Command expired before execution; no changes were made");
    try {
        return execute_command(app, *this, pending.method, pending.params, pending.client);
    } catch (const json::exception& error) {
        return failure("INVALID_REQUEST", error.what());
    } catch (const std::exception& error) {
        return failure("INTERNAL", error.what());
    } catch (...) {
        return failure("INTERNAL", "Unknown command failure");
    }
}

void ControlBus::resolve(Pending& pending, json value) {
    if (!pending.promise) return;
    try {
        pending.promise->set_value(std::move(value));
    } catch (const std::exception&) {
        // The HTTP client timed out or disconnected.
    }
    pending.promise.reset();
}

bool ControlBus::should_defer(const Pending& pending) {
    if (pending.method == "studio.job.start") return true;
    if (pending.method == "studio.edit.apply" || pending.method == "studio.export.mesh") return true;
    if (pending.method == "studio.project.save" || pending.method == "studio.export.model") return true;
    if (pending.method == "studio.project.open" ||
        pending.method == "studio.project.set_source")
        return true;
    if (pending.method == "studio.job.control") {
        return pending.params.contains("action") && pending.params["action"] == "stop";
    }
    return false;
}

void ControlBus::pump_main_thread(App& app) {
    std::deque<Pending> queue;
    {
        std::lock_guard lock(queue_mutex_);
        queue.swap(main_queue_);
    }
    for (Pending& pending : queue) {
        if (should_defer(pending)) {
            std::lock_guard lock(queue_mutex_);
            post_present_queue_.push_back(std::move(pending));
            continue;
        }
        resolve(pending, execute_pending(app, pending));
    }
    try {
        sync_revision(app);
    } catch (...) {
        // A malformed state must not take down the editor.
    }
}

void ControlBus::pump_post_present(App& app) {
    std::deque<Pending> queue;
    {
        std::lock_guard lock(queue_mutex_);
        queue.swap(post_present_queue_);
    }
    for (Pending& pending : queue) {
        resolve(pending, execute_pending(app, pending));
    }
    try {
        sync_revision(app);
    } catch (...) {
        // A malformed state must not take down the editor.
    }
}

void ControlBus::stop() {
    std::deque<Pending> main_queue;
    std::deque<Pending> post_queue;
    {
        std::lock_guard lock(queue_mutex_);
        if (stopping_) return;
        stopping_ = true;
        main_queue.swap(main_queue_);
        post_queue.swap(post_present_queue_);
    }
    for (Pending& pending : main_queue)
        resolve(pending, failure("SHUTDOWN", "Studio control is stopping"));
    for (Pending& pending : post_queue)
        resolve(pending, failure("SHUTDOWN", "Studio control is stopping"));
}

std::string ControlBus::begin_operation(
    const std::string& stage, const std::string& client) {
    active_operation_ = "op_" + std::to_string(next_operation_id_++);
    active_client_ = client;
    active_stage_ = stage;
    operation_cancelled_ = false;
    ++revision_;
    push_event("job.started", {
        {"op_id", active_operation_},
        {"stage", active_stage_},
        {"client", active_client_},
        {"revision", revision()},
    });
    return active_operation_;
}

void ControlBus::sync_revision(const App& app) {
    std::string key = state_revision_key(app);
    if (key != last_state_key_) {
        last_state_key_ = std::move(key);
        const std::uint64_t next = ++revision_;
        const auto now = std::chrono::steady_clock::now();
        const bool publish =
            last_state_event_.time_since_epoch().count() == 0 ||
            now - last_state_event_ >= std::chrono::milliseconds(250);
        if (publish) {
            last_state_event_ = now;
            push_event("state.changed", {{"revision", next}});
            if (!active_operation_.empty())
                push_event("job.progress", snapshot_job(app, *this));
        }
    }
    finish_operation_if_done(app);
}

void ControlBus::finish_operation_if_done(const App& app) {
    if (active_operation_.empty()) return;
    if (app.job.running() || app.pending_train_prepare.valid() ||
        app.active_job != JobKind::none)
        return;
    ++revision_;
    json terminal = snapshot_job(app, *this);
    terminal["status"] = operation_cancelled_ || app.job.exit_code() == 2 ? "cancelled" :
        (app.job.exit_code() != 0 || !app.monitor.last_error().empty() ? "failed" : "succeeded");
    terminal["running"] = false;
    terminal["preparing"] = false;
    terminal["finishing"] = false;
    terminal["requested_stage"] = active_stage_;
    terminal["message"] = app.message;
    completed_operations_[active_operation_] = terminal;
    completed_order_.push_back(active_operation_);
    while (completed_order_.size() > 128) {
        completed_operations_.erase(completed_order_.front());
        completed_order_.pop_front();
    }
    const json finished{
        {"op_id", active_operation_},
        {"stage", active_stage_},
        {"client", active_client_},
        {"exit_code", app.job.exit_code()},
        {"message", app.message},
        {"last_error", app.monitor.last_error()},
        {"status", terminal["status"]},
        {"revision", revision()},
    };
    active_operation_.clear();
    active_client_.clear();
    active_stage_.clear();
    push_event("job.finished", finished);
}

json ControlBus::operation_status(const App& app, const std::string& id) {
    finish_operation_if_done(app);
    if (id.empty() || id == active_operation_) {
        json job = snapshot_job(app, *this);
        const int exit_code = app.job.exit_code();
        job["status"] = job.value("preparing", false) ? "preparing" :
            job.value("running", false) ? (job.value("paused", false) ? "paused" : "running") :
            job.value("finishing", false) ? "finishing" : exit_code < 0 ? "idle" :
            exit_code == 2 ? "cancelled" :
            exit_code != 0 || !app.monitor.last_error().empty() ? "failed" : "succeeded";
        return success({{"job", std::move(job)}});
    }
    const auto found = completed_operations_.find(id);
    if (found == completed_operations_.end())
        return failure("NOT_FOUND", "Operation is unknown or no longer retained", {{"op_id", id}});
    return success({{"job", found->second}});
}

void ControlBus::push_event(std::string type, json data) {
    std::lock_guard lock(event_mutex_);
    EventRecord event;
    event.id = next_event_id_++;
    event.type = std::move(type);
    event.data = std::move(data);
    event.time_ms = now_milliseconds();
    events_.push_back(std::move(event));
    while (events_.size() > 2000) events_.pop_front();
}

json ControlBus::events_since(const std::uint64_t since) const {
    std::lock_guard lock(event_mutex_);
    json result = json::array();
    for (const EventRecord& event : events_) {
        if (event.id <= since) continue;
        result.push_back({
            {"id", event.id},
            {"type", event.type},
            {"time_ms", event.time_ms},
            {"data", event.data},
        });
    }
    return success({
        {"events", std::move(result)},
        {"next", next_event_id_ - 1},
        {"revision", revision()},
    });
}

json ControlBus::recent_events(const std::size_t limit) const {
    std::lock_guard lock(event_mutex_);
    json result = json::array();
    const std::size_t begin =
        events_.size() > limit ? events_.size() - limit : 0;
    for (std::size_t i = begin; i < events_.size(); ++i) {
        const EventRecord& event = events_[i];
        result.push_back({
            {"id", event.id},
            {"type", event.type},
            {"time_ms", event.time_ms},
            {"data", event.data},
        });
    }
    return success({
        {"events", std::move(result)},
        {"next", next_event_id_ - 1},
        {"revision", revision()},
    });
}

}  // namespace editor::control
