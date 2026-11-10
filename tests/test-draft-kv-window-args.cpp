#include "arg.h"
#include "common.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static bool parse(std::vector<std::string> options, common_params & params) {
    options.insert(options.begin(), "test-draft-kv-window-args");
    std::vector<char *> argv;
    for (auto & option : options) {
        argv.push_back(option.data());
    }
    return common_params_parse(int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER);
}

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        common_params windowed;
        require(parse({"-np", "1", "--spec-type", "draft-mtp", "--spec-draft-kv-window", "8192"}, windowed),
                "valid windowed MTP arguments rejected");
        require(windowed.speculative.draft.kv_window == 8192, "window not parsed");
        const auto target = common_context_params_to_llama(windowed);
        require(target.kv_window == 0 && target.kv_rollback == 0, "draft option leaked into target parameters");

        common_params disabled;
        require(parse({"-np", "2", "--spec-draft-kv-window", "0"}, disabled), "W=0 changed legacy argument validation");
        require(disabled.speculative.draft.kv_window == 0, "W=0 changed the default");

        const std::vector<std::vector<std::string>> invalid = {
            {"--spec-draft-kv-window", "-1"},
            {"--spec-draft-kv-window", "4294967296"},
            {"--spec-draft-kv-window", "8192x"},
            {"--spec-draft-kv-window", "8192"},
            {"-np", "2", "--spec-type", "draft-mtp", "--spec-draft-kv-window", "8192"},
        };
        for (const auto & options : invalid) {
            common_params params;
            require(!parse(options, params), "invalid window arguments accepted");
        }
        std::puts("DRAFT_WINDOW_ARGS_PASSED");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL draft window arguments: %s\n", e.what());
        return 1;
    }
}
