#pragma once

#include "common.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <list>
#include <mutex>
#include <thread>

// DenseX: only detached, owning host buffers may cross the writer boundary.
struct server_doc_cache_job {
    std::string path;
    uint64_t settings = 0;
    std::vector<char> packed;
    llama_tokens tokens;
    std::vector<uint8_t> target;
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> draft;
    std::vector<uint8_t> spec;
};

struct server_doc_cache_result {
    std::string path;
    std::string tmp;
    std::string error;
    size_t bytes = 0;
};

class server_doc_cache_writer {
public:
    static constexpr uint64_t budget = 3072ULL * 1024 * 1024;
    static constexpr uint64_t headroom = 2048ULL * 1024 * 1024;
    static constexpr uint32_t magic = 0x32444344; // DenseX private container: DCD2
    static constexpr uint32_t version = 1;

    explicit server_doc_cache_writer(std::function<void()> wake);
    ~server_doc_cache_writer();
    bool reserve(uint64_t bytes, uint64_t available);
    void cancel_reservation();
    void submit(std::unique_ptr<server_doc_cache_job> job);
    std::unique_ptr<server_doc_cache_result> take_result();
    void shutdown();

    static uint64_t available_memory();
    static size_t read_target(const std::string & path, llama_tokens & packed, std::vector<uint8_t> & target);
    static bool publish(server_doc_cache_result & result);

private:
    void run();
    std::function<void()> wake;
    std::mutex mutex;
    std::condition_variable cv;
    bool busy = false;
    bool stopping = false;
    std::unique_ptr<server_doc_cache_job> pending;
    std::unique_ptr<server_doc_cache_result> result;
    std::thread worker;
};
