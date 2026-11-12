#include "llama-densex-env.h"

#include "llama-impl.h"

#include <cstdlib>

// DenseX: single read point for the llama-side DENSEX_* switches (issue #2). The environment is read
// and parsed once, on the first call, and the values actually in use are logged once. Same defaults and
// the same parsing rules as the call sites this replaced: no switch was added and no rule changed.
const llama_densex_env & llama_densex_env_get() {
    static const llama_densex_env env = [] {
        const char * gdn_deferred      = getenv("DENSEX_GDN_DEFERRED");
        const char * draft_vocab_order = getenv("DENSEX_DRAFT_VOCAB_ORDER");
        const char * draft_vocab       = getenv("DENSEX_DRAFT_VOCAB");
        const char * kv_vmm            = getenv("DENSEX_KV_VMM");
        const char * kv_lazy           = getenv("DENSEX_KV_LAZY");
        const char * kv_lazy_cells     = getenv("DENSEX_KV_LAZY_CELLS");

        llama_densex_env e;
        e.gdn_deferred      = gdn_deferred == nullptr ? -1 : (atoi(gdn_deferred) != 0 ? 1 : 0);
        e.draft_vocab_order = draft_vocab_order != nullptr ? std::string(draft_vocab_order) : std::string();
        e.draft_vocab       = draft_vocab != nullptr ? (int64_t) atoll(draft_vocab) : (int64_t) 0;
        e.kv_vmm            = kv_vmm != nullptr && atoi(kv_vmm) != 0 ? 1 : 0;
        e.kv_lazy           = kv_lazy != nullptr && atoi(kv_lazy) != 0 ? 1 : 0;

        // cells mapped at once, both for the first map and for every later expansion of a lazily mapped
        // KV buffer (issue #8). a page is 2 MiB on this driver, so anything below one page's worth of
        // cells would map one page at a time and pay for the mapping on every token
        e.kv_lazy_cells = kv_lazy_cells != nullptr ? (int64_t) atoll(kv_lazy_cells) : (int64_t) 8192;
        if (e.kv_lazy_cells < 1) {
            LLAMA_LOG_WARN("%s: DENSEX_KV_LAZY_CELLS=%lld is not a number of cells to map - using 8192\n",
                    __func__, (long long) e.kv_lazy_cells);
            e.kv_lazy_cells = 8192;
        }

        if (e.kv_lazy && !e.kv_vmm) {
            LLAMA_LOG_WARN("%s: DENSEX_KV_LAZY=1 but DENSEX_KV_VMM is not set - the KV cache stays a plain"
                    " cudaMalloc buffer and nothing is mapped on demand (DENSEX_KV_LAZY needs DENSEX_KV_VMM=1)\n",
                    __func__);
        }

        LLAMA_LOG_INFO("densex env: GDN_DEFERRED=%d DRAFT_VOCAB=%lld DRAFT_VOCAB_ORDER=%s KV_VMM=%d"
                " KV_LAZY=%d (0 = off, 1 = on, needs KV_VMM=1) KV_LAZY_CELLS=%lld (>0, cells mapped per"
                " step, default 8192)\n",
                e.gdn_deferred, (long long) e.draft_vocab,
                e.draft_vocab_order.empty() ? "<unset>" : e.draft_vocab_order.c_str(), e.kv_vmm,
                e.kv_lazy, (long long) e.kv_lazy_cells);
        return e;
    }();

    return env;
}
