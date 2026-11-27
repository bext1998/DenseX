#pragma once

#include "common.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <list>
#include <mutex>
#include <thread>

// DenseX #88：文件快取的非同步存檔。
// 做什麼：模型執行緒把 prefill 後的狀態複製到主機記憶體，背景執行緒再寫入跟同步版逐位元相同的檔案。
// 輸入／輸出：server_doc_cache_job（全部是主機緩衝）-> `<hex>-<n>.bin` 與 `.tok`，`.tmp` 完成後才改名。
// 開關：`--densex-doc-cache-async`（預設關），必須同時給 `--densex-doc-cache`。
// 依賴：common 的 checkpoint 型別；背景執行緒不碰 llama_context、slot 或 GPU。

// DenseX: recognize only names produced by the document cache.
bool server_doc_cache_is_filename(const std::string & name);

// DenseX: only detached, owning host buffers may cross the writer boundary.
struct server_doc_cache_job {
    std::string path;
    uint64_t settings = 0;
    std::vector<char> packed;
    llama_tokens tokens;
    // llama_state_seq_get_data_ext 的完整輸出，開頭是 io magic 與 sequence id。
    std::vector<uint8_t> target;
    // 檔案格式裡沒有上面那段 8 byte 標頭，寫檔時從這裡開始複製，省掉 GB 等級的搬移。
    size_t target_offset = 0;
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> draft;
    std::vector<uint8_t> spec;
};

struct server_doc_cache_result {
    std::string path;
    std::string tmp;
    std::string error;
    size_t bytes = 0;
    int64_t write_us = 0;
};

class server_doc_cache_writer {
public:
    static constexpr uint64_t budget = 3072ULL * 1024 * 1024;
    static constexpr uint64_t headroom = 2048ULL * 1024 * 1024;
    // #53 版寫出的私有容器（DCD2）。新檔已改回同步版格式；讀取仍接受這段期間留下的舊檔。
    static constexpr uint32_t legacy_magic = 0x32444344; // DCD2
    static constexpr uint32_t legacy_version = 1;

    explicit server_doc_cache_writer(std::function<void()> wake);
    ~server_doc_cache_writer();
    bool reserve(uint64_t bytes, uint64_t available);
    void cancel_reservation();
    void submit(std::unique_ptr<server_doc_cache_job> job);
    std::unique_ptr<server_doc_cache_result> take_result();
    void shutdown();

    static uint64_t available_memory();
    // 只給舊 DCD2 容器用：拆出 packed token 與 state 緩衝。
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
