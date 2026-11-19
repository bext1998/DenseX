#include "arg.h"

#include "build-info.h"
#include "common.h"
#include "log.h"
#include "download.h"
#include "hf-cache.h"
#include "json.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

//
// downloader
//

// DenseX：只有證明暫存檔與遠端物件是同一版本時才能續傳。sidecar 把驗證器與可信
// 摘要存在 <path>.downloadInProgress 旁邊。
struct download_meta {
    int         version = 1;
    std::string url;
    std::string etag;
    std::string last_modified;
    std::string sha256;
    size_t      size = 0;
};

static void download_meta_save(const std::string & path, const download_meta & m) {
    const std::string data =
        "version=1\n"
        "url=" + m.url + "\n" +
        "etag=" + m.etag + "\n" +
        "last-modified=" + m.last_modified + "\n" +
        "size=" + std::to_string(m.size) + "\n" +
        "sha256=" + m.sha256 + "\n";
    try {
        fs_write_atomic(std::filesystem::u8path(path), data);
    } catch (const std::exception & e) {
        LOG_WRN("%s: could not save resume metadata %s: %s\n", __func__, path.c_str(), e.what());
    }
}

static bool download_meta_load(const std::string & path, download_meta & m) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) {
        return false;
    }
    download_meta out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key   = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (key == "version") {
            try {
                out.version = std::stoi(value);
            } catch (const std::exception &) {
                return false;
            }
        } else if (key == "url") {
            out.url = value;
        } else if (key == "etag") {
            out.etag = value;
        } else if (key == "last-modified") {
            out.last_modified = value;
        } else if (key == "size") {
            try {
                out.size = (size_t) std::stoull(value);
            } catch (const std::exception &) {
                return false;
            }
        } else if (key == "sha256") {
            out.sha256 = value;
        }
    }
    if (out.version != 1) {
        return false;
    }
    m = std::move(out);
    return true;
}

// DenseX：只有明確標示 SHA-256 的標頭才可信。一般 ETag 不行：HF Xet repo 送的是
// 64 hex 的 xet hash，Git blob 是 40 hex 的 SHA-1，兩者都不是檔案內容的 SHA-256。
static bool is_sha256_hex(const std::string & s) {
    if (s.size() != 64) {
        return false;
    }
    for (const char c : s) {
        if (!std::isxdigit((unsigned char) c)) {
            return false;
        }
    }
    return true;
}

static std::string normalize_sha256(std::string value) {
    while (!value.empty() && (value.front() == '"' || value.front() == '\'' || std::isspace((unsigned char) value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && (value.back() == '"' || value.back() == '\'' || std::isspace((unsigned char) value.back()))) {
        value.pop_back();
    }
    static const std::string prefix = "sha256:";
    if (value.size() > prefix.size() && value.compare(0, prefix.size(), prefix) == 0) {
        value = value.substr(prefix.size());
    }
    for (char & c : value) {
        c = (char) std::tolower((unsigned char) c);
    }
    return is_sha256_hex(value) ? value : std::string();
}

static size_t download_file_size(const std::string & path) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::u8path(path), ec)) {
        return 0;
    }
    const auto size = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    if (ec) {
        LOG_WRN("%s: could not read the size of %s: %s\n", __func__, path.c_str(), ec.message().c_str());
        return 0;
    }
    return (size_t) size;
}

static bool download_file_restart(const std::string & path) {
    std::error_code ec;
    std::filesystem::resize_file(std::filesystem::u8path(path), 0, ec);
    if (ec) {
        LOG_ERR("%s: could not truncate %s: %s\n", __func__, path.c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

// DenseX：暫存檔通過所有檢查後才替換完成檔。在那之前舊檔留在原位，交換到一半
// 失敗時會還原。
static bool download_file_replace(const std::string & from, const std::string & to) {
    std::error_code ec;
    const std::string backup = to + ".downloadInProgress.old";
    const bool had_old = std::filesystem::exists(std::filesystem::u8path(to), ec);

    if (had_old) {
        std::filesystem::remove(std::filesystem::u8path(backup), ec);
        ec.clear();
        std::filesystem::rename(std::filesystem::u8path(to), std::filesystem::u8path(backup), ec);
        if (ec) {
            LOG_ERR("%s: could not move the previous file aside: %s -> %s: %s\n", __func__, to.c_str(), backup.c_str(), ec.message().c_str());
            return false;
        }
    }
    std::filesystem::rename(std::filesystem::u8path(from), std::filesystem::u8path(to), ec);
    if (ec) {
        LOG_ERR("%s: could not move the downloaded file into place: %s -> %s: %s\n", __func__, from.c_str(), to.c_str(), ec.message().c_str());
        if (had_old) {
            std::error_code ec_restore;
            std::filesystem::rename(std::filesystem::u8path(backup), std::filesystem::u8path(to), ec_restore);
        }
        return false;
    }
    if (had_old) {
        std::filesystem::remove(std::filesystem::u8path(backup), ec);
    }
    return true;
}

static bool parse_content_range(const std::string & value, size_t & start, size_t & end, size_t & total) {
    // 格式："bytes <start>-<end>/<total>"，total 可以是 "*"。
    static const std::regex re(R"(^\s*bytes\s+([0-9]+)-([0-9]+)/([0-9]+|\*)\s*$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(value, m, re)) {
        return false;
    }
    try {
        start = (size_t) std::stoull(m[1].str());
        end   = (size_t) std::stoull(m[2].str());
        total = m[3].str() == "*" ? 0 : (size_t) std::stoull(m[3].str());
    } catch (const std::exception &) {
        return false;
    }
    return end >= start;
}

// DenseX：SHA-256（FIPS 180-4）。common/ 沒有雜湊函式庫，下載器也不該為了這個
// 依賴 TLS 函式庫，所以放一份精簡實作在這裡。
struct sha256_ctx {
    uint32_t h[8];
    uint64_t len;
    uint8_t  buf[64];
    size_t   buf_len;
};

static uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_init(sha256_ctx & ctx) {
    ctx.h[0] = 0x6a09e667u;
    ctx.h[1] = 0xbb67ae85u;
    ctx.h[2] = 0x3c6ef372u;
    ctx.h[3] = 0xa54ff53au;
    ctx.h[4] = 0x510e527fu;
    ctx.h[5] = 0x9b05688cu;
    ctx.h[6] = 0x1f83d9abu;
    ctx.h[7] = 0x5be0cd19u;
    ctx.len = 0;
    ctx.buf_len = 0;
}

static void sha256_transform(sha256_ctx & ctx, const uint8_t * p) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t) p[4 * i + 0] << 24) |
               ((uint32_t) p[4 * i + 1] << 16) |
               ((uint32_t) p[4 * i + 2] <<  8) |
               ((uint32_t) p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(w[i -  2], 17) ^ sha256_rotr(w[i -  2], 19) ^ (w[i -  2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = ctx.h[0], b = ctx.h[1], c = ctx.h[2], d = ctx.h[3];
    uint32_t e = ctx.h[4], f = ctx.h[5], g = ctx.h[6], h = ctx.h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    ctx.h[0] += a;
    ctx.h[1] += b;
    ctx.h[2] += c;
    ctx.h[3] += d;
    ctx.h[4] += e;
    ctx.h[5] += f;
    ctx.h[6] += g;
    ctx.h[7] += h;
}

static void sha256_update(sha256_ctx & ctx, const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    ctx.len += len;
    if (ctx.buf_len > 0) {
        const size_t take = std::min(len, sizeof(ctx.buf) - ctx.buf_len);
        std::memcpy(ctx.buf + ctx.buf_len, p, take);
        ctx.buf_len += take;
        p += take;
        len -= take;
        if (ctx.buf_len == sizeof(ctx.buf)) {
            sha256_transform(ctx, ctx.buf);
            ctx.buf_len = 0;
        }
    }
    while (len >= sizeof(ctx.buf)) {
        sha256_transform(ctx, p);
        p += sizeof(ctx.buf);
        len -= sizeof(ctx.buf);
    }
    if (len > 0) {
        std::memcpy(ctx.buf, p, len);
        ctx.buf_len = len;
    }
}

static void sha256_final(sha256_ctx & ctx, uint8_t out[32]) {
    const uint64_t bits = ctx.len * 8;
    uint8_t pad[72];
    std::memset(pad, 0, sizeof(pad));
    pad[0] = 0x80;
    const size_t pad_len = (ctx.buf_len < 56) ? (56 - ctx.buf_len) : (120 - ctx.buf_len);
    sha256_update(ctx, pad, pad_len);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) {
        len_be[i] = (uint8_t) (bits >> (56 - 8 * i));
    }
    sha256_update(ctx, len_be, sizeof(len_be));
    for (int i = 0; i < 8; ++i) {
        out[4 * i + 0] = (uint8_t) (ctx.h[i] >> 24);
        out[4 * i + 1] = (uint8_t) (ctx.h[i] >> 16);
        out[4 * i + 2] = (uint8_t) (ctx.h[i] >>  8);
        out[4 * i + 3] = (uint8_t) (ctx.h[i]);
    }
}

static std::string sha256_file_hex(const std::string & path) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) {
        return {};
    }
    sha256_ctx c;
    sha256_init(c);
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = in.gcount();
        if (n > 0) {
            sha256_update(c, buf.data(), (size_t) n);
        }
    }
    if (in.bad()) {
        return {};
    }
    uint8_t digest[32];
    sha256_final(c, digest);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const uint8_t v : digest) {
        out.push_back(hex[v >> 4]);
        out.push_back(hex[v & 15]);
    }
    return out;
}

static void write_etag(const std::string & path, const std::string & etag) {
    const std::string etag_path = path + ".etag";
    fs_write_atomic(std::filesystem::u8path(etag_path), etag);
    LOG_DBG("%s: file etag saved: %s\n", __func__, etag_path.c_str());
}

static std::string read_etag(const std::string & path) {
    const std::string etag_path = path + ".etag";
    if (!std::filesystem::exists(etag_path)) {
        return {};
    }
    std::ifstream etag_in(etag_path);
    if (!etag_in) {
        LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, etag_path.c_str());
        return {};
    }
    std::string etag;
    std::getline(etag_in, etag);
    return etag;
}

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

class ProgressBar : public common_download_callback {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
        return common_is_tty(stdout);
    }

public:
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = p.url;

        if (auto pos = filename.rfind('/'); pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        if (auto pos = filename.find('?'); pos != std::string::npos) {
            filename = filename.substr(0, pos);
        }
        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

enum class download_pull_result {
    ok,
    failed,
    restart,
};

// DenseX：需要丟掉暫存檔、從第 0 位元組重抓時回傳 "restart"（範圍過期、If-Range
// 被忽略）。其他拒絕都是硬失敗，不會把寫一半的檔案留成完成檔。
static download_pull_result common_pull_file(httplib::Client & cli,
                                             const std::string & resolve_path,
                                             const std::string & path_tmp,
                                             bool supports_ranges,
                                             const std::string & if_range,
                                             common_download_progress & p,
                                             common_download_callback * callback) {
    std::ofstream ofs;
    bool opened = false;
    bool restart = false;

    auto open_file = [&](bool truncate) -> bool {
        if (opened) {
            return true;
        }
        ofs.open(std::filesystem::u8path(path_tmp),
                 std::ios::binary | (truncate ? std::ios::trunc : std::ios::app));
        if (!ofs.is_open()) {
            LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
            return false;
        }
        opened = true;
        return true;
    };

    httplib::Headers headers;
    if (supports_ranges && p.downloaded > 0 && !if_range.empty()) {
        headers.emplace("Range", "bytes=" + std::to_string(p.downloaded) + "-");
        headers.emplace("If-Range", if_range);
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t progress_step = 0;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (p.downloaded > 0) {
                if (response.status == 200) {
                    // DenseX：If-Range 沒對上，或伺服器忽略 Range。把這個 body
                    // 接在舊前綴後面會混到兩個版本。
                    LOG_WRN("%s: server sent a full body instead of a partial response; restarting from zero\n", func);
                    if (!open_file(true)) {
                        return false;
                    }
                    p.downloaded = 0;
                    if (response.has_header("Content-Length")) {
                        try {
                            const size_t content_length = (size_t) std::stoull(response.get_header_value("Content-Length"));
                            if (p.total != 0 && content_length != p.total) {
                                LOG_ERR("%s: Content-Length %zu does not match the expected size %zu\n", func, content_length, p.total);
                                return false;
                            }
                            p.total = content_length;
                        } catch (const std::exception & e) {
                            LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                        }
                    }
                    return true;
                }
                if (response.status == 416) {
                    LOG_WRN("%s: server rejected the resume range; restarting from zero\n", func);
                    restart = true;
                    return false;
                }
                if (response.status != 206) {
                    LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                    return false;
                }
                // DenseX：只有 Content-Range 正好從要求的位元組開始、且屬於同一個
                // 完整檔案時，206 才能用。
                if (!response.has_header("Content-Range")) {
                    LOG_ERR("%s: 206 response without a Content-Range header\n", func);
                    return false;
                }
                size_t start = 0, end = 0, total = 0;
                if (!parse_content_range(response.get_header_value("Content-Range"), start, end, total)) {
                    LOG_ERR("%s: malformed Content-Range: %s\n", func, response.get_header_value("Content-Range").c_str());
                    return false;
                }
                if (start != p.downloaded) {
                    LOG_ERR("%s: Content-Range starts at %zu but the local file ends at %zu\n", func, start, p.downloaded);
                    return false;
                }
                if (total != 0 && p.total != 0 && total != p.total) {
                    LOG_ERR("%s: Content-Range total is %zu but the expected size is %zu\n", func, total, p.total);
                    return false;
                }
                if (p.total == 0 && total != 0) {
                    p.total = total;
                }
                if (response.has_header("Content-Length")) {
                    try {
                        const size_t content_length = (size_t) std::stoull(response.get_header_value("Content-Length"));
                        if (content_length != end - start + 1) {
                            LOG_ERR("%s: Content-Length %zu does not match the Content-Range length %zu\n",
                                    func, content_length, end - start + 1);
                            return false;
                        }
                    } catch (const std::exception & e) {
                        LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                    }
                }
                if (!open_file(false)) {
                    return false;
                }
                return true;
            }

            if (response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (!open_file(true)) {
                return false;
            }
            if (response.has_header("Content-Length")) {
                try {
                    const size_t content_length = (size_t) std::stoull(response.get_header_value("Content-Length"));
                    if (p.total != 0 && content_length != p.total) {
                        LOG_ERR("%s: Content-Length %zu does not match the expected size %zu\n", func, content_length, p.total);
                        return false;
                    }
                    p.total = content_length;
                } catch (const std::exception & e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            if (!opened) {
                LOG_ERR("%s: response body arrived before the destination file was open\n", func);
                return false;
            }
            ofs.write(data, (std::streamsize) len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", func, path_tmp.c_str());
                return false;
            }
            p.downloaded += len;
            progress_step += len;

            if (progress_step >= p.total / 1000 || p.downloaded == p.total) {
                if (callback) {
                    callback->on_update(p);
                    if (callback->is_cancelled()) {
                        return false;
                    }
                }
                progress_step = 0;
            }
            return true;
        },
        nullptr
    );

    if (opened) {
        ofs.close();
        if (!ofs) {
            LOG_ERR("%s: error closing file: %s\n", func, path_tmp.c_str());
            return download_pull_result::failed;
        }
    }

    if (restart) {
        return download_pull_result::restart;
    }

    if (!res) {
        LOG_ERR("%s: download failed: %s (status: %d)\n",
                __func__,
                httplib::to_string(res.error()).c_str(),
                res ? res->status : -1);
        return download_pull_result::failed;
    }

    return download_pull_result::ok;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string & url,
                                              const std::string & path,
                                              const common_download_opts & opts,
                                              bool skip_etag) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    const bool file_exists = std::filesystem::exists(std::filesystem::u8path(path));

    if (file_exists && skip_etag) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return 304; // 304 Not Modified - fake cached response
    }

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (!opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    cli.set_default_headers(headers);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_DBG("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    // DenseX：HF 在 resolve 回應用 X-Linked-ETag 帶 LFS SHA-256，httplib 跟隨
    // 轉址到 CDN 後這個標頭就沒了。先看不跟隨的第一個回應，再正常 HEAD 取得大小
    // 與 range 支援。
    cli.set_follow_location(false);
    auto head = cli.Head(parts.path);
    std::string linked_sha256;
    if (head) {
        if (head->has_header("X-Linked-ETag")) {
            linked_sha256 = normalize_sha256(head->get_header_value("X-Linked-ETag"));
        }
        if (head->status >= 300 && head->status < 400 && head->has_header("Location")) {
            cli.set_follow_location(true);
            head = cli.Head(parts.path);
        }
    }
    cli.set_follow_location(true);
    if (head && head->has_header("X-Linked-ETag")) {
        const std::string header_sha256 = normalize_sha256(head->get_header_value("X-Linked-ETag"));
        if (!header_sha256.empty()) {
            if (!linked_sha256.empty() && linked_sha256 != header_sha256) {
                LOG_ERR("%s: X-Linked-ETag changed between the resolve and the final response\n", __func__);
                return -1;
            }
            linked_sha256 = header_sha256;
        }
    }

    if (!head || head->status < 200 || head->status >= 300) {
        LOG_TRC("%s: HEAD failed, status: %d\n", __func__, head ? head->status : -1);
        if (file_exists) {
            LOG_TRC("%s: using cached file (HEAD failed): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        return head ? head->status : -1;
    }

    std::string etag;
    if (head->has_header("ETag")) {
        etag = head->get_header_value("ETag");
    }
    std::string last_modified;
    if (head->has_header("Last-Modified")) {
        last_modified = head->get_header_value("Last-Modified");
    }

    common_download_progress p;
    p.url = url;
    if (head->has_header("Content-Length")) {
        try {
            p.total = std::stoull(head->get_header_value("Content-Length"));
        } catch (const std::exception& e) {
            LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, e.what());
        }
    }

    bool supports_ranges = false;
    if (head->has_header("Accept-Ranges")) {
        supports_ranges = head->get_header_value("Accept-Ranges") != "none";
    }

    // DenseX：預期摘要來自呼叫端（HF LFS oid、OCI blob digest）或 X-Linked-ETag。
    // 一般 ETag 絕不當成 SHA-256 使用。
    std::string trusted_sha256;
    if (!opts.expected_sha256.empty()) {
        trusted_sha256 = normalize_sha256(opts.expected_sha256);
        if (trusted_sha256.empty()) {
            LOG_WRN("%s: ignoring expected sha256 that is not a 64-hex digest: %s\n", __func__, opts.expected_sha256.c_str());
        }
    }
    if (!linked_sha256.empty()) {
        if (!trusted_sha256.empty() && trusted_sha256 != linked_sha256) {
            LOG_ERR("%s: expected sha256 %s does not match the server X-Linked-ETag %s\n",
                    __func__, trusted_sha256.c_str(), linked_sha256.c_str());
            return -1;
        }
        trusted_sha256 = linked_sha256;
    }

    if (file_exists) {
        if (etag.empty()) {
            LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_DBG("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        // DenseX：新檔通過所有檢查前，舊檔留在原位，更新失敗不會毀掉可用的模型。
        LOG_INF("%s: remote file changed; keeping the previous file until the new one is verified\n", __func__);
    }

    { // silent
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }

    bool success = false;
    const std::string path_temporary = path + ".downloadInProgress";
    const std::string path_meta      = path_temporary + ".meta";
    int delay = retry_delay_seconds;

    // DenseX：If-Range 需要強驗證器。沒有驗證器就無法把本機暫存檔綁到遠端版本，
    // 這種情況一律不續傳。
    const bool strong_etag = !etag.empty() && etag.rfind("W/", 0) != 0;
    const std::string if_range = strong_etag ? etag : last_modified;
    const bool can_resume = supports_ranges && !if_range.empty();

    bool resume = false;
    if (std::filesystem::exists(std::filesystem::u8path(path_temporary))) {
        download_meta meta;
        const bool meta_ok = download_meta_load(path_meta, meta);
        resume = can_resume &&
                 meta_ok &&
                 meta.url           == url &&
                 meta.etag          == etag &&
                 meta.last_modified == last_modified &&
                 meta.size          == p.total &&
                 meta.sha256        == trusted_sha256;
        if (!resume) {
            const char * why = !can_resume      ? "no strong validator or range support" :
                               !meta_ok         ? "missing or invalid resume metadata" :
                               meta.url != url  ? "url" :
                               meta.etag != etag ? "etag" :
                               meta.last_modified != last_modified ? "last-modified" :
                               meta.size != p.total ? "size" : "sha256";
            LOG_WRN("%s: partial file does not belong to the current remote version (%s); restarting from zero\n", __func__, why);
            if (!download_file_restart(path_temporary)) {
                return -1;
            }
        }
    }

    download_meta meta;
    meta.url           = url;
    meta.etag          = etag;
    meta.last_modified = last_modified;
    meta.sha256        = trusted_sha256;
    meta.size          = p.total;
    download_meta_save(path_meta, meta);

    if (opts.callback) {
        opts.callback->on_start(p);
    }

    for (int i = 0; i < max_attempts; ++i) {
        if (opts.callback && opts.callback->is_cancelled()) {
            break;
        }
        if (i) {
            LOG_WRN("%s: retrying after %d seconds...\n", __func__, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            delay *= retry_delay_seconds;
        }

        size_t existing_size = download_file_size(path_temporary);
        if (!can_resume || (p.total != 0 && existing_size > p.total)) {
            if (existing_size > 0 && !download_file_restart(path_temporary)) {
                break;
            }
            existing_size = 0;
        }

        p.downloaded = existing_size;

        LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        const download_pull_result rc = common_pull_file(cli, parts.path, path_temporary, supports_ranges,
                                                         can_resume ? if_range : std::string(), p, opts.callback);
        if (rc == download_pull_result::restart) {
            if (!download_file_restart(path_temporary)) {
                break;
            }
            continue;
        }
        if (rc != download_pull_result::ok) {
            continue;
        }

        const size_t final_size = download_file_size(path_temporary);
        if (p.total != 0 && final_size != p.total) {
            LOG_ERR("%s: downloaded %zu bytes but the remote file has %zu bytes\n", __func__, final_size, p.total);
            if (final_size > p.total && !download_file_restart(path_temporary)) {
                break;
            }
            continue;
        }

        if (!trusted_sha256.empty()) {
            const std::string actual = sha256_file_hex(path_temporary);
            if (actual.empty() || actual != trusted_sha256) {
                LOG_ERR("%s: sha256 mismatch: expected %s, got %s\n",
                        __func__, trusted_sha256.c_str(), actual.empty() ? "(read error)" : actual.c_str());
                if (!download_file_restart(path_temporary)) {
                    break;
                }
                continue;
            }
            LOG_DBG("%s: verified sha256 %s\n", __func__, actual.c_str());
        } else {
            LOG_WRN("%s: no trusted sha256 for this download; only Content-Range and size were verified\n", __func__);
        }

        if (!download_file_replace(path_temporary, path)) {
            break;
        }
        if (!etag.empty() && !skip_etag) {
            write_etag(path, etag);
        }
        {
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(path_meta), ec);
        }
        success = true;
        break;
    }

    if (opts.callback) {
        opts.callback->on_done(p, success);
    }
    if (opts.callback && opts.callback->is_cancelled() &&
        std::filesystem::exists(std::filesystem::u8path(path_temporary))) {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path_temporary), ec);
        if (ec) {
            LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, path_temporary.c_str());
        }
        std::filesystem::remove(std::filesystem::u8path(path_meta), ec);
    }
    if (!success) {
        LOG_ERR("%s: download failed after %d attempts; the partial file is kept for a later resume: %s\n",
                __func__, max_attempts, path_temporary.c_str());
        return -1; // max attempts reached
    }

    return head->status;
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }

    if (params.timeout > 0) {
        cli.set_read_timeout(params.timeout, 0);
        cli.set_write_timeout(params.timeout, 0);
    }

    std::vector<char> buf;
    auto res = cli.Get(parts.path, headers,
        [&](const char *data, size_t len) {
            buf.insert(buf.end(), data, data + len);
            return params.max_size == 0 ||
                   buf.size() <= static_cast<size_t>(params.max_size);
        },
        nullptr
    );

    if (!res) {
        throw std::runtime_error("error: cannot make GET request");
    }

    return { res->status, std::move(buf) };
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts,
                                bool skip_etag) {
    if (!opts.offline) {
        ProgressBar tty_cb;
        common_download_opts online_opts = opts;
        if (!online_opts.callback) {
            online_opts.callback = &tty_cb;
        }
        return common_download_file_single_online(url, path, online_opts, skip_etag);
    }

    if (!std::filesystem::exists(std::filesystem::u8path(path))) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, path.c_str());

    // notify the callback that the file was cached
    if (opts.callback) {
        common_download_progress p;
        p.url = url;
        p.cached = true;
        opts.callback->on_start(p);
        opts.callback->on_done(p, true);
    }

    return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
    std::smatch m;

    std::string prefix = path;
    if (!string_remove_suffix(prefix, ".gguf")) {
        return {};
    }

    int index = 1;
    int count = 1;

    if (std::regex_match(prefix, m, re_split)) {
        index = std::stoi(m[2].str());
        count = std::stoi(m[3].str());
        prefix = m[1].str();
    }

    std::string tag;
    if (std::regex_search(prefix, m, re_tag)) {
        tag = m[1].str();
        for (char & c : tag) {
            c = std::toupper((unsigned char)c);
        }
    }

    return {std::move(prefix), std::move(tag), index, count};
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int extract_quant_bits(const std::string & filename) {
    auto split = get_gguf_split_info(filename);

    auto pos = split.tag.find_first_of("0123456789");
    if (pos == std::string::npos) {
        return 0;
    }

    return std::stoi(split.tag.substr(pos));
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                          const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

// pick the best sibling GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp"),
// preferring deeper shared directory prefix with the model, then exact `tag` match,
// then closest quantization to the tag when given, or to the model otherwise
static hf_cache::hf_file find_best_sibling(const hf_cache::hf_files & files,
                                           const std::string        & model,
                                           const std::string        & keyword,
                                           const std::string        & tag = "") {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool best_exact = false;
    bool found = false;

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    int model_bits = 0;
    if (!tag_upper.empty()) {
        auto pos = tag_upper.find_first_of("0123456789");
        model_bits = pos == std::string::npos ? 0 : std::stoi(tag_upper.substr(pos));
    } else {
        model_bits = extract_quant_bits(model);
    }
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find(keyword) == std::string::npos) {
            continue;
        }

        auto sib_parts = string_split<std::string>(f.path, '/');
        auto sib_dir = sib_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      sib_parts.begin(), sib_dir);
        if (dir != sib_dir) {
            continue;
        }

        size_t depth = dir - sib_parts.begin();
        auto bits = extract_quant_bits(f.path);
        auto diff = std::abs(bits - model_bits);

        std::string path_upper = f.path;
        for (char & c : path_upper) {
            c = (char) std::toupper((unsigned char) c);
        }
        bool exact = !tag_upper.empty() && path_upper.find("-" + tag_upper + ".") != std::string::npos;

        if (!found || depth > best_depth ||
            (depth == best_depth && exact && !best_exact) ||
            (depth == best_depth && exact == best_exact && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    return best;
}

static hf_cache::hf_file find_best_mmproj(const hf_cache::hf_files & files,
                                          const std::string        & model) {
    return find_best_sibling(files, model, "mmproj");
}

static hf_cache::hf_file find_best_mtp(const hf_cache::hf_files & files,
                                       const std::string        & model,
                                       const std::string        & tag = "") {
    return find_best_sibling(files, model, "mtp-", tag);
}

static hf_cache::hf_file find_best_dflash(const hf_cache::hf_files & files,
                                          const std::string        & model,
                                          const std::string        & tag = "") {
    return find_best_sibling(files, model, "dflash-", tag);
}

static hf_cache::hf_file find_best_dspark(const hf_cache::hf_files & files,
                                          const std::string        & model,
                                          const std::string        & tag = "") {
    return find_best_sibling(files, model, "dspark-", tag);
}

static bool gguf_filename_is_model(const std::string & filepath) {
    if (!string_ends_with(filepath, ".gguf")) {
        return false;
    }

    std::string filename = filepath;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }

    return filename.find("mmproj")  == std::string::npos &&
           filename.find("imatrix") == std::string::npos &&
           filename.find("mtp-")    == std::string::npos &&
           filename.find("eagle3-") == std::string::npos &&
           filename.find("dflash-") == std::string::npos &&
           filename.find("dspark-") == std::string::npos;
}

static hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                         const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts) {
    common_download_hf_plan plan;
    hf_cache::hf_files all;

    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    if (!opts.offline) {
        all = hf_cache::get_repo_files(repo, opts.bearer_token);
    }
    if (all.empty()) {
        all = hf_cache::get_cached_files(repo);
    }
    if (all.empty()) {
        return plan;
    }

    // if preset.ini exists in the repo root, download only that file
    for (const auto & f : all) {
        if (f.path == "preset.ini") {
            plan.preset = f;
            return plan;
        }
    }

    hf_cache::hf_file primary;

    if (!model.hf_file.empty()) {
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    } else {
        primary = find_best_model(all, tag);
        // a requested sidecar can resolve on its own, without a full model of the same tag
        if (primary.path.empty() && !opts.download_mtp && !opts.download_dflash && !opts.download_dspark) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    }

    if (!primary.path.empty()) {
        plan.primary = primary;
        plan.model_files = get_split_files(all, primary);
    }

    if (opts.download_mmproj && !primary.path.empty()) {
        plan.mmproj = find_best_mmproj(all, primary.path);
    }
    if (opts.download_mtp) {
        plan.mtp = find_best_mtp(all, primary.path, tag);
    }
    if (opts.download_dflash) {
        plan.dflash = find_best_dflash(all, primary.path, tag);
    }
    if (opts.download_dspark) {
        plan.dspark = find_best_dspark(all, primary.path, tag);
    }

    if (primary.path.empty() &&
        plan.mtp.local_path.empty() && plan.dflash.local_path.empty() && plan.dspark.local_path.empty()) {
        LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
        list_available_gguf_files(all);
    }

    return plan;
}

void common_download_run_tasks(const std::vector<common_download_task> & tasks) {
    std::vector<std::future<int>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task]() {
                return common_download_file_single(task.url, task.local_path, task.opts, task.is_hf);
            }
        ));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        std::string url = tasks[i].url;
        int status = futures[i].get();
        bool is_ok = is_http_status_ok(status);
        if (!is_ok) {
            throw std::runtime_error(string_format("Download '%s' failed with status code: %d", url.c_str(), status));
        }
    }
}

std::vector<std::string> common_download_get_all_parts(const std::string & url) {
    auto split = get_gguf_split_info(url);

    if (split.count <= 1) {
        return {url};
    }

    std::vector<std::string> parts;
    for (int i = 1; i <= split.count; i++) {
        auto suffix = string_format("-%05d-of-%05d.gguf", i, split.count);
        parts.push_back(split.prefix + suffix);
    }
    return parts;
}

//
// Docker registry functions
//

static std::string common_docker_get_token(const std::string & repo) {
    std::string url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull";

    common_remote_params params;
    auto                 res = common_remote_get_content(url, params);

    if (res.first != 200) {
        throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(res.first));
    }

    std::string response_str(res.second.begin(), res.second.end());
    common_json response = common_json::parse(response_str);

    if (!response.contains("token")) {
        throw std::runtime_error("Docker registry token response missing 'token' field");
    }

    return response["token"].get<std::string>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    // Parse ai/smollm2:135M-Q4_0
    size_t      colon_pos = docker.find(':');
    std::string repo, tag;
    if (colon_pos != std::string::npos) {
        repo = docker.substr(0, colon_pos);
        tag  = docker.substr(colon_pos + 1);
    } else {
        repo = docker;
        tag  = "latest";
    }

    // ai/ is the default
    size_t      slash_pos = docker.find('/');
    if (slash_pos == std::string::npos) {
        repo.insert(0, "ai/");
    }

    LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, repo.c_str(), tag.c_str());
    try {
        // --- helper: digest validation ---
        auto validate_oci_digest = [](const std::string & digest) -> std::string {
            // Expected: algo:hex ; start with sha256 (64 hex chars)
            // You can extend this map if supporting other algorithms in future.
            static const std::regex re("^sha256:([a-fA-F0-9]{64})$");
            std::smatch m;
            if (!std::regex_match(digest, m, re)) {
                throw std::runtime_error("Invalid OCI digest format received in manifest: " + digest);
            }
            // normalize hex to lowercase
            std::string normalized = digest;
            std::transform(normalized.begin()+7, normalized.end(), normalized.begin()+7, [](unsigned char c){
                return std::tolower(c);
            });
            return normalized;
        };

        std::string token = common_docker_get_token(repo);  // Get authentication token

        // Get manifest
        // TODO: cache the manifest response so that it appears in the model list
        const std::string    url_prefix = "https://registry-1.docker.io/v2/" + repo;
        std::string          manifest_url = url_prefix + "/manifests/" + tag;
        common_remote_params manifest_params;
        manifest_params.headers.push_back({"Authorization", "Bearer " + token});
        manifest_params.headers.push_back({"Accept",
            "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
        });
        auto manifest_res = common_remote_get_content(manifest_url, manifest_params);
        if (manifest_res.first != 200) {
            throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(manifest_res.first));
        }

        std::string manifest_str(manifest_res.second.begin(), manifest_res.second.end());
        common_json manifest = common_json::parse(manifest_str);
        std::string gguf_digest;  // Find the GGUF layer
        if (manifest.contains("layers")) {
            for (const auto & layer : manifest["layers"]) {
                if (layer.contains("mediaType")) {
                    std::string media_type = layer["mediaType"].get<std::string>();
                    if (media_type == "application/vnd.docker.ai.gguf.v3" ||
                        media_type.find("gguf") != std::string::npos) {
                        gguf_digest = layer["digest"].get<std::string>();
                        break;
                    }
                }
            }
        }

        if (gguf_digest.empty()) {
            throw std::runtime_error("No GGUF layer found in Docker manifest");
        }

        // Validate & normalize digest
        gguf_digest = validate_oci_digest(gguf_digest);
        LOG_DBG("%s: Using validated digest: %s\n", __func__, gguf_digest.c_str());

        // Prepare local filename
        std::string model_filename = repo;
        std::replace(model_filename.begin(), model_filename.end(), '/', '_');
        model_filename += "_" + tag + ".gguf";
        std::string local_path = fs_path_to_utf8(fs_get_cache_file(model_filename));

        const std::string blob_url = url_prefix + "/blobs/" + gguf_digest;
        common_download_opts opts;
        opts.bearer_token = token;
        // DenseX：OCI manifest 的 sha256 digest 是可信的完成檔摘要。
        opts.expected_sha256 = gguf_digest.substr(7);
        const int http_status = common_download_file_single(blob_url, local_path, opts);
        if (!is_http_status_ok(http_status)) {
            throw std::runtime_error("Failed to download Docker Model");
        }

        LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, local_path.c_str());
        return local_path;
    } catch (const std::exception & e) {
        LOG_ERR("%s: Docker Model download failed: %s\n", __func__, e.what());
        throw;
    }
}

std::vector<common_cached_model_info> common_list_cached_models() {
    std::unordered_set<std::string> seen;
    std::vector<common_cached_model_info> result;

    auto files = hf_cache::get_cached_files();

    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.index != 1 || split.tag.empty() ||
            split.prefix.find("mmproj")  != std::string::npos ||
            split.prefix.find("mtp-")    != std::string::npos ||
            split.prefix.find("eagle3-") != std::string::npos ||
            split.prefix.find("dflash-") != std::string::npos ||
            split.prefix.find("dspark-") != std::string::npos) {
            continue;
        }
        if (seen.insert(f.repo_id + ":" + split.tag).second) {
            result.push_back({f.repo_id, split.tag});
        }
    }

    return result;
}

std::string common_download_resolve_path(const std::string & hf_repo_with_tag, const std::string & hf_file) {
    auto [repo, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    auto files = hf_cache::get_cached_files(repo);
    if (files.empty()) {
        return "";
    }

    if (!hf_file.empty()) {
        for (const auto & f : files) {
            if (f.path == hf_file) {
                return f.local_path;
            }
        }
        return "";
    }

    return find_best_model(files, tag).local_path;
}

bool common_download_remove(const std::string & hf_repo_with_tag) {
    namespace fs = std::filesystem;

    auto [repo_id, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    if (tag.empty()) {
        return hf_cache::remove_cached_repo(repo_id);
    }

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    auto files = hf_cache::get_cached_files(repo_id);
    if (files.empty()) {
        return false;
    }

    // collect snapshot entries whose tag matches
    std::vector<fs::path> to_remove;
    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.tag == tag_upper) {
            to_remove.emplace_back(f.local_path);
        }
    }

    if (to_remove.empty()) {
        return false;
    }

    // resolve blob paths from symlinks before deleting snapshot entries
    std::vector<fs::path> blobs_to_check;
    for (const auto & p : to_remove) {
        std::error_code ec;
        if (fs::is_symlink(p, ec)) {
            auto target = fs::read_symlink(p, ec);
            if (!ec) {
                blobs_to_check.push_back((p.parent_path() / target).lexically_normal());
            }
        }
    }

    // remove snapshot entries
    for (const auto & p : to_remove) {
        std::error_code ec;
        fs::remove(p, ec);
        if (ec) {
            LOG_WRN("%s: failed to remove %s: %s\n", __func__, p.string().c_str(), ec.message().c_str());
        }
    }

    if (blobs_to_check.empty()) {
        return true;
    }

    // collect blobs still referenced by remaining snapshot entries
    std::unordered_set<std::string> still_referenced;
    for (const auto & f : hf_cache::get_cached_files(repo_id)) {
        fs::path p(f.local_path);
        std::error_code ec;
        if (fs::is_symlink(p, ec)) {
            auto target = fs::read_symlink(p, ec);
            if (!ec) {
                still_referenced.insert((p.parent_path() / target).lexically_normal().string());
            }
        }
    }

    // remove orphaned blobs
    for (const auto & blob : blobs_to_check) {
        if (still_referenced.find(blob.string()) == still_referenced.end()) {
            std::error_code ec;
            fs::remove(blob, ec);
            if (ec) {
                LOG_WRN("%s: failed to remove blob %s: %s\n", __func__, blob.string().c_str(), ec.message().c_str());
            }
        }
    }

    return true;
}
