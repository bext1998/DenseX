#pragma once

#include "json.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// DenseX: stores only fixed labels and numeric request summaries.
class server_request_log {
    using clock = std::chrono::steady_clock;
    static constexpr size_t capacity = 128;
    static constexpr size_t task_capacity = 4096;
    struct task_info {
        uint64_t request_id;
        bool started = false;
    };
    struct entry {
        uint64_t id;
        std::string endpoint;
        int64_t started_at;
        clock::time_point queued;
        clock::time_point processing;
        clock::time_point finished;
        bool has_queue = false;
        bool has_processing = false;
        bool has_finished = false;
        size_t task_count = 0;
        size_t pending = 0;
        uint64_t input_tokens = 0;
        uint64_t output_tokens = 0;
        bool output_known = true;
        std::string status = "processing";
        std::string reason;
    };

    std::mutex mutex;
    uint64_t next_id = 0;
    uint64_t sequence = 0;
    uint64_t dropped_requests = 0;
    uint64_t requests_evicted = 0;
    uint64_t events_evicted = 0;
    std::map<uint64_t, entry> active;
    std::map<int, task_info> tasks;
    std::deque<entry> history;
    std::deque<common_json> events;
    std::string instance_id;

    static int64_t wall_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    static int64_t duration_ms(clock::time_point start, clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    void event_locked(const char * code) {
        events.push_back(common_json {{"sequence", ++sequence}, {"at", wall_ms()}, {"code", code}});
        if (events.size() > capacity) { events.pop_front(); ++events_evicted; }
    }
    void finish_locked(std::map<uint64_t, entry>::iterator it) {
        auto & e = it->second;
        e.finished = clock::now();
        e.has_finished = true;
        if (e.reason.empty()) e.reason = "completed";
        e.status = e.reason == "cancelled" ? "cancelled" :
            (e.reason == "invalid_request" || e.reason == "server_error") ? "error" : "completed";
        history.push_back(e);
        if (history.size() > capacity) { history.pop_front(); ++requests_evicted; }
        active.erase(it);
        ++sequence;
    }
    static common_json serialize(const entry & e, clock::time_point now) {
        const auto end = e.has_finished ? e.finished : now;
        common_json queue_ms = nullptr;
        common_json processing_ms = nullptr;
        if (e.has_queue) queue_ms = duration_ms(e.queued, e.has_processing ? e.processing : end);
        if (e.has_processing) processing_ms = duration_ms(e.processing, end);
        return {
            {"id", e.id}, {"endpoint", e.endpoint}, {"started_at", e.started_at},
            {"status", e.status}, {"finish_reason", !e.has_finished || e.reason.empty() ? common_json(nullptr) : common_json(e.reason)},
            {"queue_ms", queue_ms}, {"processing_ms", processing_ms},
            {"input_tokens", e.task_count ? common_json(e.input_tokens) : common_json(nullptr)},
            {"output_tokens", e.has_finished && e.output_known ? common_json(e.output_tokens) : common_json(nullptr)},
            {"unavailable_reason", e.task_count == 0 ? common_json("request not tokenized") :
                !e.has_finished ? common_json("request in progress") : !e.output_known ? common_json("partial task counters unavailable") : common_json(nullptr)},
            {"task_count", e.task_count},
        };
    }

public:
    explicit server_request_log(std::string instance) : instance_id(std::move(instance)) { event_locked("model_loaded"); }

    uint64_t begin(const char * endpoint) {
        std::lock_guard<std::mutex> lock(mutex);
        if (active.size() >= capacity) { ++dropped_requests; ++sequence; return 0; }
        const uint64_t id = ++next_id;
        entry e;
        e.id = id;
        e.endpoint = endpoint;
        e.started_at = wall_ms();
        active.emplace(id, std::move(e));
        ++sequence;
        return id;
    }
    void enqueue(uint64_t id, const std::vector<std::pair<int, uint64_t>> & batch) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = active.find(id);
        if (it == active.end()) return;
        if (batch.empty() || batch.size() > task_capacity - tasks.size()) {
            active.erase(it);
            ++dropped_requests;
            ++sequence;
            return;
        }
        auto & e = it->second;
        e.has_queue = true;
        e.queued = clock::now();
        e.status = "queued";
        e.task_count = batch.size();
        e.pending = batch.size();
        for (const auto & item : batch) {
            tasks.emplace(item.first, task_info {id, false});
            e.input_tokens += item.second;
        }
        ++sequence;
    }
    void start_task(int task_id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto t = tasks.find(task_id);
        if (t == tasks.end() || t->second.started) return;
        t->second.started = true;
        auto & e = active.at(t->second.request_id);
        if (!e.has_processing) { e.processing = clock::now(); e.has_processing = true; }
        e.status = "processing";
        ++sequence;
    }
    void end_task(int task_id, const char * reason, bool tokens_known, uint64_t output_tokens) {
        std::lock_guard<std::mutex> lock(mutex);
        auto t = tasks.find(task_id);
        if (t == tasks.end()) return;
        auto it = active.find(t->second.request_id);
        tasks.erase(t);
        if (it == active.end()) return;
        auto & e = it->second;
        e.output_known &= tokens_known;
        if (tokens_known) e.output_tokens += output_tokens;
        const std::string value(reason);
        if (value == "server_error" || value == "invalid_request") {
            e.reason = value;
            event_locked("request_error");
        } else if ((value == "cancelled" && e.reason != "server_error" && e.reason != "invalid_request") || e.reason.empty() || e.reason == "completed") {
            e.reason = value;
        }
        if (--e.pending == 0) finish_locked(it);
        else ++sequence;
    }
    void fail_http(uint64_t id, const char * reason) {
        if (!id) return;
        std::lock_guard<std::mutex> lock(mutex);
        auto it = active.find(id);
        if (it == active.end()) return;
        it->second.reason = reason;
        event_locked("request_error");
        if (!it->second.pending) finish_locked(it);
    }
    common_json snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        auto now = clock::now();
        common_json request_list = common_json::array();
        common_json event_list = common_json::array();
        for (const auto & e : history) request_list.push_back(serialize(e, now));
        for (const auto & e : active) request_list.push_back(serialize(e.second, now));
        for (const auto & e : events) event_list.push_back(e);
        return {{"schema_version", 1}, {"instance_id", instance_id}, {"sampled_at", wall_ms()},
            {"sequence", sequence}, {"scope", "server_lifetime"},
            {"capacities", common_json {{"requests", capacity}, {"active_requests", capacity}, {"active_tasks", task_capacity}, {"events", capacity}}},
            {"dropped_requests", dropped_requests}, {"requests_evicted", requests_evicted}, {"events_evicted", events_evicted},
            {"requests", request_list}, {"events", event_list}};
    }
};
