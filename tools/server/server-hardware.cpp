#include "server-hardware.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace {

using json = common_json;

json unavailable(const char * unit, const char * scope, const char * source, const std::string & reason) {
    return {{"available", false}, {"value", nullptr}, {"unit", unit}, {"scope", scope}, {"window", "recent_sample"}, {"source", source}, {"reason", reason}};
}

template <typename T>
json measured(T value, const char * unit, const char * scope, const char * source) {
    return {{"available", true}, {"value", value}, {"unit", unit}, {"scope", scope}, {"window", "recent_sample"}, {"source", source}};
}

json empty_sample(const char * reason) {
    json m = json::object();
    m["process_ram_bytes"] = unavailable("bytes", "server process working set", "Windows K32GetProcessMemoryInfo", reason);
    m["system_ram_used_bytes"] = unavailable("bytes", "system physical memory", "Windows GlobalMemoryStatusEx", reason);
    m["system_ram_total_bytes"] = unavailable("bytes", "system physical memory", "Windows GlobalMemoryStatusEx", reason);
    m["gpu_vram_used_bytes"] = unavailable("bytes", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_vram_total_bytes"] = unavailable("bytes", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_temperature_c"] = unavailable("C", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_graphics_clock_mhz"] = unavailable("MHz", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_memory_clock_mhz"] = unavailable("MHz", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_power_w"] = unavailable("W", "NVML device index 0, entire GPU", "NVML", reason);
    m["gpu_name"] = unavailable("", "NVML device index 0, entire GPU", "NVML", reason);
    m["process_gpu_dedicated_bytes"] = unavailable("bytes", "server process, all adapters", "Windows PDH GPU Process Memory", reason);
    m["process_gpu_shared_bytes"] = unavailable("bytes", "server process, all adapters", "Windows PDH GPU Process Memory", reason);
    return {{"sampled_at", nullptr}, {"sample_interval_ms", 1000}, {"metrics", m}};
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc = {};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[21];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

#ifdef _WIN32
class sensors {
    using nv_init = int (__cdecl *)();
    using nv_shutdown = int (__cdecl *)();
    using nv_handle = int (__cdecl *)(unsigned int, void **);
    using nv_uint = int (__cdecl *)(void *, unsigned int *);
    using nv_enum_uint = int (__cdecl *)(void *, int, unsigned int *);
    using nv_name = int (__cdecl *)(void *, char *, unsigned int);
    struct nv_memory { unsigned long long total, free, used; };
    using nv_mem = int (__cdecl *)(void *, nv_memory *);

    HMODULE nvml_ = nullptr;
    HMODULE pdh_ = nullptr;
    void * gpu_ = nullptr;
    nv_shutdown shutdown_ = nullptr;
    bool initialized_ = false;
    PDH_HQUERY query_ = nullptr;
    PDH_HCOUNTER dedicated_ = nullptr, shared_ = nullptr;
    decltype(&PdhOpenQueryW) open_ = nullptr;
    decltype(&PdhAddEnglishCounterW) add_ = nullptr;
    decltype(&PdhCollectQueryData) collect_ = nullptr;
    decltype(&PdhGetFormattedCounterArrayW) array_ = nullptr;
    decltype(&PdhCloseQuery) close_ = nullptr;

    template <typename T> T nv(const char * name) const { return reinterpret_cast<T>(GetProcAddress(nvml_, name)); }
    template <typename T> T pdh(const char * name) const { return reinterpret_cast<T>(GetProcAddress(pdh_, name)); }

    json counter(PDH_HCOUNTER handle) {
        const char * scope = "server process, all adapters";
        const char * source = "Windows PDH GPU Process Memory";
        if (!handle || !array_) return unavailable("bytes", scope, source, "counter unavailable");
        DWORD size = 0, count = 0;
        PDH_STATUS status = array_(handle, PDH_FMT_LARGE | PDH_FMT_NOSCALE, &size, &count, nullptr);
        if (status != PDH_MORE_DATA || size == 0) return unavailable("bytes", scope, source, "counter instances unavailable");
        std::vector<unsigned char> buffer(size);
        auto * items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W *>(buffer.data());
        status = array_(handle, PDH_FMT_LARGE | PDH_FMT_NOSCALE, &size, &count, items);
        if (status != ERROR_SUCCESS) return unavailable("bytes", scope, source, "counter read failed");
        const std::wstring prefix = L"pid_" + std::to_wstring(GetCurrentProcessId()) + L"_";
        unsigned long long total = 0;
        bool found = false;
        for (DWORD i = 0; i < count; ++i) {
            if (!items[i].szName || std::wstring(items[i].szName).find(prefix) != 0) continue;
            found = true;
            const auto & value = items[i].FmtValue;
            if ((value.CStatus != PDH_CSTATUS_VALID_DATA && value.CStatus != PDH_CSTATUS_NEW_DATA) || value.largeValue < 0) {
                return unavailable("bytes", scope, source, "invalid process counter instance");
            }
            total += static_cast<unsigned long long>(value.largeValue);
        }
        if (!found) return unavailable("bytes", scope, source, "no process counter instances");
        return measured(total, "bytes", scope, source);
    }

public:
    sensors() {
        nvml_ = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (nvml_) {
            auto init = nv<nv_init>("nvmlInit_v2");
            shutdown_ = nv<nv_shutdown>("nvmlShutdown");
            auto handle = nv<nv_handle>("nvmlDeviceGetHandleByIndex_v2");
            if (init && shutdown_ && handle && init() == 0) {
                initialized_ = true;
                if (handle(0, &gpu_) != 0) gpu_ = nullptr;
            }
        }
        pdh_ = LoadLibraryExW(L"pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (pdh_) {
            open_ = pdh<decltype(open_)>("PdhOpenQueryW");
            add_ = pdh<decltype(add_)>("PdhAddEnglishCounterW");
            collect_ = pdh<decltype(collect_)>("PdhCollectQueryData");
            array_ = pdh<decltype(array_)>("PdhGetFormattedCounterArrayW");
            close_ = pdh<decltype(close_)>("PdhCloseQuery");
            if (open_ && add_ && collect_ && array_ && close_ && open_(nullptr, 0, &query_) == ERROR_SUCCESS) {
                if (add_(query_, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &dedicated_) != ERROR_SUCCESS) dedicated_ = nullptr;
                if (add_(query_, L"\\GPU Process Memory(*)\\Shared Usage", 0, &shared_) != ERROR_SUCCESS) shared_ = nullptr;
            }
        }
    }
    ~sensors() {
        if (query_ && close_) close_(query_);
        if (pdh_) FreeLibrary(pdh_);
        if (initialized_) shutdown_();
        if (nvml_) FreeLibrary(nvml_);
    }
    void sample(json & m) {
        PROCESS_MEMORY_COUNTERS pm = {};
        pm.cb = sizeof(pm);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof(pm))) {
            m["process_ram_bytes"] = measured(static_cast<unsigned long long>(pm.WorkingSetSize), "bytes", "server process working set", "Windows K32GetProcessMemoryInfo");
        }
        MEMORYSTATUSEX mem = {};
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            m["system_ram_used_bytes"] = measured(mem.ullTotalPhys - mem.ullAvailPhys, "bytes", "system physical memory", "Windows GlobalMemoryStatusEx");
            m["system_ram_total_bytes"] = measured(mem.ullTotalPhys, "bytes", "system physical memory", "Windows GlobalMemoryStatusEx");
        }
        if (query_ && collect_(query_) == ERROR_SUCCESS) {
            m["process_gpu_dedicated_bytes"] = counter(dedicated_);
            m["process_gpu_shared_bytes"] = counter(shared_);
        }
        if (!gpu_) return;
        const char * scope = "NVML device index 0, entire GPU";
        nv_memory memory = {};
        auto get_mem = nv<nv_mem>("nvmlDeviceGetMemoryInfo");
        if (get_mem && get_mem(gpu_, &memory) == 0) {
            m["gpu_vram_used_bytes"] = measured(memory.used, "bytes", scope, "NVML");
            m["gpu_vram_total_bytes"] = measured(memory.total, "bytes", scope, "NVML");
        }
        char name[256] = {};
        auto get_name = nv<nv_name>("nvmlDeviceGetName");
        if (get_name && get_name(gpu_, name, sizeof(name)) == 0) m["gpu_name"] = measured(std::string(name), "", scope, "NVML");
        unsigned int value = 0;
        auto temperature = nv<nv_enum_uint>("nvmlDeviceGetTemperature");
        if (temperature && temperature(gpu_, 0, &value) == 0) m["gpu_temperature_c"] = measured(value, "C", scope, "NVML");
        auto clock = nv<nv_enum_uint>("nvmlDeviceGetClockInfo");
        if (clock && clock(gpu_, 0, &value) == 0) m["gpu_graphics_clock_mhz"] = measured(value, "MHz", scope, "NVML");
        if (clock && clock(gpu_, 2, &value) == 0) m["gpu_memory_clock_mhz"] = measured(value, "MHz", scope, "NVML");
        auto power = nv<nv_uint>("nvmlDeviceGetPowerUsage");
        if (power && power(gpu_, &value) == 0) m["gpu_power_w"] = measured(value / 1000.0, "W", scope, "NVML");
    }
};
#endif
}

densex_hardware_sampler::densex_hardware_sampler() : snapshot_(empty_sample("not sampled yet")) {}
densex_hardware_sampler::~densex_hardware_sampler() { stop(); }

void densex_hardware_sampler::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_.joinable()) return;
    stopping_ = false;
    thread_ = std::thread(&densex_hardware_sampler::run, this);
}

void densex_hardware_sampler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

common_json densex_hardware_sampler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

void densex_hardware_sampler::run() {
#ifdef _WIN32
    sensors source;
#endif
    for (;;) {
        const auto next_sample = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
#ifdef _WIN32
        json sample = empty_sample("API unavailable or call failed");
        source.sample(sample["metrics"]);
#else
        json sample = empty_sample("unsupported platform");
#endif
        sample["sampled_at"] = utc_now();
        std::unique_lock<std::mutex> lock(mutex_);
        snapshot_ = std::move(sample);
        if (wake_.wait_until(lock, next_sample, [this] { return stopping_; })) break;
    }
}
