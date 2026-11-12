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

        llama_densex_env e;
        e.gdn_deferred      = gdn_deferred == nullptr ? -1 : (atoi(gdn_deferred) != 0 ? 1 : 0);
        e.draft_vocab_order = draft_vocab_order != nullptr ? std::string(draft_vocab_order) : std::string();
        e.draft_vocab       = draft_vocab != nullptr ? (int64_t) atoll(draft_vocab) : (int64_t) 0;
        e.kv_vmm            = kv_vmm != nullptr && atoi(kv_vmm) != 0 ? 1 : 0;

        LLAMA_LOG_INFO("densex env: GDN_DEFERRED=%d DRAFT_VOCAB=%lld DRAFT_VOCAB_ORDER=%s KV_VMM=%d\n",
                e.gdn_deferred, (long long) e.draft_vocab,
                e.draft_vocab_order.empty() ? "<unset>" : e.draft_vocab_order.c_str(), e.kv_vmm);
        return e;
    }();

    return env;
}
