#pragma once

#include "json.h"

#include <condition_variable>
#include <mutex>
#include <thread>

class densex_hardware_sampler {
public:
    densex_hardware_sampler();
    ~densex_hardware_sampler();

    densex_hardware_sampler(const densex_hardware_sampler &) = delete;
    densex_hardware_sampler & operator=(const densex_hardware_sampler &) = delete;

    void start();
    void stop();
    common_json snapshot() const;

private:
    void run();

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::thread thread_;
    common_json snapshot_;
};
