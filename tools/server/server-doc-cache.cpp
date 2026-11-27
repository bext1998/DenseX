#include "server-doc-cache.h"

#include <chrono>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <stdexcept>

bool server_doc_cache_is_filename(const std::string & name) {
    if (name.size() < 22 || name[16] != '-' || name.compare(name.size() - 4, 4, ".bin") != 0) {
        return false;
    }
    for (size_t i = 0; i < 16; ++i) {
        if (!((name[i] >= '0' && name[i] <= '9') || (name[i] >= 'a' && name[i] <= 'f'))) {
            return false;
        }
    }
    const char * begin = name.data() + 17;
    const char * end = name.data() + name.size() - 4;
    if (*begin < '0' || *begin > '9' || (*begin == '0' && end - begin > 1)) {
        return false;
    }
    size_t tokens = 0;
    const auto parsed = std::from_chars(begin, end, tokens);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {
constexpr uint32_t ckpt_magic = 0x504b4353;
constexpr uint32_t draft_magic = 0x54464453;
constexpr uint32_t tok_magic = 0x4b4f5432;

template<class T> void put(std::ofstream & out, const T & value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
template<class T> void put_buffer(std::ofstream & out, const std::vector<T> & value) {
    put(out, uint64_t(value.size() * sizeof(T)));
    out.write(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(T));
}
void close_checked(std::ofstream & out) {
    out.flush();
    out.close();
    if (!out) {
        throw std::runtime_error("incomplete write/flush/close");
    }
}
// DenseX #88：檔案佈局照抄 llama_state_seq_save_file（檔頭＋token blob＋state）與 server 既有的
// checkpoint／draft 附錄，所以同步版與背景版寫出的檔案逐位元相同，可互相比對 SHA256 或互相載入。
void write_job(const server_doc_cache_job & job, const std::string & tmp) {
    if (job.packed.size() % sizeof(llama_token) != 0 || job.packed.size() / sizeof(llama_token) > 0xffffffffULL) {
        throw std::runtime_error("invalid prompt token blob");
    }
    if (job.target_offset > job.target.size()) {
        throw std::runtime_error("invalid state snapshot offset");
    }
    std::ofstream out(std::filesystem::u8path(tmp), std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open state temporary file");
    }
    put(out, uint32_t(LLAMA_STATE_SEQ_MAGIC));
    put(out, uint32_t(LLAMA_STATE_SEQ_VERSION));
    put(out, uint32_t(job.packed.size() / sizeof(llama_token)));
    out.write(job.packed.data(), (std::streamsize) job.packed.size());
    out.write(reinterpret_cast<const char *>(job.target.data() + job.target_offset),
              (std::streamsize) (job.target.size() - job.target_offset));
    // DenseX: 同步版在清單為空時不寫這段附錄，這裡也必須略過。
    if (!job.checkpoints.empty()) {
        put(out, ckpt_magic);
        put(out, uint32_t(1));
        put(out, uint32_t(job.checkpoints.size()));
        for (const auto & cur : job.checkpoints) {
            put(out, cur.n_tokens);
            put(out, cur.pos_min);
            put(out, cur.pos_max);
            put_buffer(out, cur.data_tgt);
            put_buffer(out, cur.data_dft);
            put_buffer(out, cur.data_spec);
        }
    }
    if (!job.draft.empty() || !job.spec.empty()) {
        put(out, draft_magic);
        put(out, uint32_t(1));
        put_buffer(out, job.draft);
        put_buffer(out, job.spec);
    }
    close_checked(out);
    std::ofstream tok(std::filesystem::u8path(tmp + ".tok.tmp"), std::ios::binary | std::ios::trunc);
    if (!tok) {
        throw std::runtime_error("cannot open token temporary file");
    }
    put(tok, tok_magic);
    put(tok, uint32_t(1));
    put(tok, job.settings);
    put(tok, uint64_t(job.tokens.size()));
    tok.write(reinterpret_cast<const char *>(job.tokens.data()), job.tokens.size() * sizeof(llama_token));
    close_checked(tok);
}
}

server_doc_cache_writer::server_doc_cache_writer(std::function<void()> wake) : wake(std::move(wake)), worker([this] { run(); }) {}
server_doc_cache_writer::~server_doc_cache_writer() { shutdown(); }

uint64_t server_doc_cache_writer::available_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX mem = {};
    mem.dwLength = sizeof(mem);
    return GlobalMemoryStatusEx(&mem) ? mem.ullAvailPhys : 0;
#elif defined(__linux__)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    return pages > 0 && page_size > 0 ? uint64_t(pages) * uint64_t(page_size) : 0;
#else
    return 0; // DenseX: unknown headroom declines capture instead of claiming protection.
#endif
}

bool server_doc_cache_writer::reserve(uint64_t bytes, uint64_t available) {
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping || busy || bytes > budget || available < bytes + headroom) {
        return false;
    }
    busy = true;
    return true;
}
void server_doc_cache_writer::cancel_reservation() {
    std::lock_guard<std::mutex> lock(mutex);
    busy = false;
}
void server_doc_cache_writer::submit(std::unique_ptr<server_doc_cache_job> job) {
    std::lock_guard<std::mutex> lock(mutex);
    pending = std::move(job);
    cv.notify_one();
}
std::unique_ptr<server_doc_cache_result> server_doc_cache_writer::take_result() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!result) {
        return nullptr;
    }
    busy = false;
    return std::move(result);
}
void server_doc_cache_writer::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    cv.notify_one();
    if (worker.joinable()) {
        worker.join();
    }
}
void server_doc_cache_writer::run() {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return stopping || pending != nullptr; });
        if (!pending) {
            return;
        }
        auto job = std::move(pending);
        lock.unlock();
        auto done = std::make_unique<server_doc_cache_result>();
        done->path = job->path;
        done->tmp = job->path + "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
        const int64_t t_start = ggml_time_us();
        try {
            write_job(*job, done->tmp);
            done->bytes = size_t(std::filesystem::file_size(std::filesystem::u8path(done->tmp)));
        } catch (const std::exception & err) {
            done->error = err.what();
        }
        done->write_us = ggml_time_us() - t_start;
        job.reset();
        lock.lock();
        result = std::move(done);
        lock.unlock();
        wake();
    }
}

bool server_doc_cache_writer::publish(server_doc_cache_result & result) {
    if (!result.error.empty()) {
        return false;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(fs::u8path(result.path), ec) || ec) {
        result.error = "destination already exists or cannot be inspected; temporary files retained";
        return false;
    }
    fs::rename(fs::u8path(result.tmp + ".tok.tmp"), fs::u8path(result.path + ".tok"), ec);
    if (!ec) {
        fs::rename(fs::u8path(result.tmp), fs::u8path(result.path), ec);
    }
    if (ec) {
        result.error = ec.message();
        return false;
    }
    return true;
}

size_t server_doc_cache_writer::read_target(const std::string & path, llama_tokens & packed, std::vector<uint8_t> & target) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!in) {
        return 0;
    }
    const uint64_t size = uint64_t(in.tellg());
    if (size > budget || size < 24 || available_memory() < size + headroom || !in.seekg(0)) {
        return 0;
    }
    uint64_t pos = 0;
    auto read = [&](void * dst, uint64_t n) {
        if (n > size - pos || !in.read(static_cast<char *>(dst), n)) {
            return false;
        }
        pos += n;
        return true;
    };
    auto skip_buffer = [&]() {
        uint64_t n = 0;
        if (!read(&n, sizeof(n)) || n > size - pos || !in.seekg(n, std::ios::cur)) {
            return false;
        }
        pos += n;
        return true;
    };
    uint32_t m = 0, v = 0;
    uint64_t n = 0;
    if (!read(&m, 4) || m != legacy_magic || !read(&v, 4) || v != legacy_version ||
        !read(&n, 8) || n > 10000000 || n > (size - pos) / sizeof(llama_token)) {
        return 0;
    }
    packed.resize(size_t(n));
    if (!read(packed.data(), n * sizeof(llama_token)) || !read(&n, 8) || n == 0 || n > size - pos) {
        return 0;
    }
    // DenseX: admit the target staging buffer before allocating; release it before appendix restore.
    target.resize(size_t(n));
    if (!read(target.data(), n)) {
        return 0;
    }
    const size_t offset = size_t(pos);
    uint32_t count = 0;
    if (!read(&m, 4) || m != ckpt_magic || !read(&v, 4) || v != 1 || !read(&count, 4)) {
        return 0;
    }
    for (uint32_t i = 0; i < count; ++i) {
        common_prompt_checkpoint cur;
        if (!read(&cur.n_tokens, sizeof(cur.n_tokens)) || !read(&cur.pos_min, sizeof(cur.pos_min)) ||
            !read(&cur.pos_max, sizeof(cur.pos_max)) || !skip_buffer() || !skip_buffer() || !skip_buffer()) {
            return 0;
        }
    }
    if (!read(&m, 4) || m != draft_magic || !read(&v, 4) || v != 1 ||
        !skip_buffer() || !skip_buffer() || pos != size) {
        return 0;
    }
    return offset;
}
