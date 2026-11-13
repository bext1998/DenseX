// Regression test for issue #32: MTP drafting must stop at the per-sequence draft cap
// (draft_params::n_max) as well as the global cap, exactly like draft-simple does, so no
// draft decode is spent on tokens that common_speculative_draft() truncates away afterwards.
//
// Fixture: an in-process qwen35 GGUF built from metadata only (block_count = 3 with
// nextn_predict_layers = 1 => 2 trunk layers + 1 MTP block); tensor data comes from the
// callback below. The KV list mirrors get_gguf_ctx() in tests/test-llama-archs.cpp with
// two deltas, marked in build_metadata(). CPU only: no model files, no backend discovery.

#include "common.h"
#include "speculative.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct test_fail : std::runtime_error {
    explicit test_fail(const std::string & what) : std::runtime_error(what) {}
};

void require(bool ok, const std::string & msg) {
    if (!ok) {
        throw test_fail(msg);
    }
}

// ---------------------------------------------------------------- fixture --

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// same random-fill approach as tests/test-llama-archs.cpp
void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;
    size_t seed = params.seed;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, params.stdev);

    // note: Mamba A must be negative (state decay)
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            const float val = dis(gen);
            tmp[i] = is_ssm_a ? -fabsf(val) : val;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            const float val = dis(gen);
            tmp[i] = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(val) : val);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

gguf_ptr build_metadata() {
    gguf_ptr ret(gguf_init_empty(), gguf_free);
    llama_model_saver ms(LLM_ARCH_QWEN35, ret.get());

    // geometry: same defaults as get_gguf_ctx()
    const uint32_t n_ctx   = 256;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_embd_head = n_embd / n_head;
    // delta 1: get_gguf_ctx() writes block_count = 2 with no nextn key, which cannot
    // create an MTP draft context (llama-context.cpp rejects nextn_predict_layers == 0).
    // A real qwen35 MTP file has block_count = n_layer_all and nextn = 1.
    const uint32_t n_layer_all = 3; // 2 trunk layers + 1 MTP block

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer_all);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,      uint32_t(1)); // delta 2: the MTP block
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,        uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,      uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,          uint32_t(131072));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,          uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,   uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,
              std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer_all);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer_all);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-6f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            uint32_t(256)); // qwen35 branch of get_gguf_ctx()
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);

    // dummy tokenizer, same as get_gguf_ctx()
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);
        ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    return ret;
}

// ---------------------------------------------------------------- helpers --

struct seq_state {
    std::vector<llama_token> prompt;
    std::vector<llama_token> result;
    llama_pos pos0 = 0;
};

void set_dp(common_speculative * spec, llama_seq_id seq, int32_t cap, seq_state & st) {
    st.result.clear();
    auto & dp = common_speculative_get_draft_params(spec, seq);
    dp.drafting = true;
    dp.n_max    = cap;      // <= 0 means "no per-sequence limit"
    dp.pos0     = st.pos0;
    dp.id_last  = st.prompt.back();
    dp.prompt   = &st.prompt;
    dp.result   = &st.result;
    dp.result_q = nullptr;
    dp.temp     = 0.7f;
    dp.seed     = 42;
}

// The MTP loop queues one anchor row at pos0 and one extra row per accepted draft token
// (pos0 + i + 1), so rows in the draft KV == draft decodes executed for that sequence.
int draft_decodes(const llama_memory_t mem_dft, llama_seq_id seq, llama_pos pos0) {
    const llama_pos max_pos = llama_memory_seq_pos_max(mem_dft, seq);
    return max_pos < pos0 ? 0 : (int) (max_pos - pos0 + 1);
}

void check_round(const std::string & name, common_speculative * spec, seq_state & s0, seq_state & s1,
                 int want_dec0, int want_dec1, int want_res0, int want_res1,
                 const llama_memory_t mem_tgt, const llama_memory_t mem_dft) {
    // production always advances pos0, while this test drafts again at the same position;
    // drop the previous round's draft rows first (M-RoPE requires X < Y on the anchor row)
    llama_memory_seq_rm(mem_dft, 0, s0.pos0, -1);
    llama_memory_seq_rm(mem_dft, 1, s1.pos0, -1);

    const llama_pos tgt0 = llama_memory_seq_pos_max(mem_tgt, 0);
    const llama_pos tgt1 = llama_memory_seq_pos_max(mem_tgt, 1);

    common_speculative_draft(spec);

    require(draft_decodes(mem_dft, 0, s0.pos0) == want_dec0,
            name + ": seq0 draft decodes = " + std::to_string(draft_decodes(mem_dft, 0, s0.pos0)) +
            ", expected " + std::to_string(want_dec0));
    require(draft_decodes(mem_dft, 1, s1.pos0) == want_dec1,
            name + ": seq1 draft decodes = " + std::to_string(draft_decodes(mem_dft, 1, s1.pos0)) +
            ", expected " + std::to_string(want_dec1));
    require((int) s0.result.size() == want_res0,
            name + ": seq0 result = " + std::to_string(s0.result.size()) +
            ", expected " + std::to_string(want_res0));
    require((int) s1.result.size() == want_res1,
            name + ": seq1 result = " + std::to_string(s1.result.size()) +
            ", expected " + std::to_string(want_res1));
    require(!common_speculative_get_draft_params(spec, 0).drafting,
            name + ": seq0 drafting flag not cleared");
    require(!common_speculative_get_draft_params(spec, 1).drafting,
            name + ": seq1 drafting flag not cleared");
    require(llama_memory_seq_pos_max(mem_tgt, 0) == tgt0 && llama_memory_seq_pos_max(mem_tgt, 1) == tgt1,
            name + ": target KV moved while drafting");
    std::printf("%s: seq0 decodes=%d result=%zu | seq1 decodes=%d result=%zu\n",
                name.c_str(), draft_decodes(mem_dft, 0, s0.pos0), s0.result.size(),
                draft_decodes(mem_dft, 1, s1.pos0), s1.result.size());
}

common_speculative * make_spec(llama_context * ctx_tgt, llama_context * ctx_dft,
                                int32_t n_min, float p_min) {
    common_params_speculative sp;
    sp.types           = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sp.draft.n_max     = 3;      // global cap for every scenario
    sp.draft.n_min     = n_min;
    sp.draft.p_min     = p_min;
    sp.draft.backend_sampling = false; // deterministic greedy drafting on CPU
    sp.draft.ctx_tgt   = ctx_tgt;
    sp.draft.ctx_dft   = ctx_dft;
    common_speculative * spec = common_speculative_init(sp, 2);
    require(spec != nullptr, "common_speculative_init returned null");
    require(common_speculative_n_max(spec) == 3, "MTP implementation was not registered");
    return spec;
}

} // namespace

int main() {
    try {
        llama_backend_init();

        // model: qwen35 + one MTP block, random weights
        gguf_ptr gguf(build_metadata());
        tensor_data_params tp = { 42, 0.1f };
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp     = true; // like the server with --spec-type draft-mtp: the nextn
                                // block must be loaded (otherwise blk.2 tensors are SKIPped)
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp), llama_model_free);
        require(model != nullptr, "failed to create the qwen35 MTP fixture model");

        // contexts: target + independent MTP draft context (same weights, no ctx_other so
        // the draft KV is its own memory and every queued draft row is observable)
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 64;
        cp.n_batch         = 64;
        cp.n_ubatch        = 64;
        cp.n_seq_max       = 2;
        cp.n_threads       = 2;
        cp.n_threads_batch = 2;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx_tgt(
                llama_init_from_model(model.get(), cp), llama_free);
        require(ctx_tgt != nullptr, "failed to create the target context");

        llama_context_params cpd = cp;
        cpd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx_dft(
                llama_init_from_model(model.get(), cpd), llama_free);
        require(ctx_dft != nullptr, "failed to create the MTP draft context (missing nextn?)");

        const llama_memory_t mem_tgt = llama_get_memory(ctx_tgt.get());
        const llama_memory_t mem_dft = llama_get_memory(ctx_dft.get());

        seq_state s0, s1;
        for (llama_token t = 1; t <= 12; t++) {
            s0.prompt.push_back(t);
        }
        for (llama_token t = 21; t <= 28; t++) {
            s1.prompt.push_back(t);
        }
        s0.pos0 = (llama_pos) s0.prompt.size(); // 12
        s1.pos0 = (llama_pos) s1.prompt.size(); // 8

        // decode the prompts through the target and hand every batch to the drafter
        // (output flag on every position: MTP needs the target hidden state of each row)
        // decode only the not-yet-decoded tail of each prompt: absolute positions are the
        // indices, from0/from1 are the first index that still needs decoding (0 for the
        // initial prompt, the old pos0 for an extension)
        auto feed = [&](common_speculative * spec, llama_pos from0, llama_pos from1) {
            common_batch batch(ctx_tgt.get());
            for (size_t i = (size_t) from0; i < s0.prompt.size(); i++) {
                batch.add(s0.prompt[i], (llama_pos) i, 0, true);
            }
            for (size_t i = (size_t) from1; i < s1.prompt.size(); i++) {
                batch.add(s1.prompt[i], (llama_pos) i, 1, true);
            }
            const int rc = llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            require(rc == 0, "target prompt decode failed: " + std::to_string(rc));
            require(common_speculative_process(spec, batch), "common_speculative_process failed");
        };

        // ------------------------------------------------ group A: n_min=0, p_min=0
        common_speculative * specA = make_spec(ctx_tgt.get(), ctx_dft.get(), 0, 0.0f);
        common_speculative_begin(specA, 0, s0.prompt);
        common_speculative_begin(specA, 1, s1.prompt);
        feed(specA, 0, 0);
        // the catch-up decode must have filled the draft KV up to pos0-1
        require(llama_memory_seq_pos_max(mem_dft, 0) == s0.pos0 - 1,
                "seq0 draft KV not caught up with the prompt");
        require(llama_memory_seq_pos_max(mem_dft, 1) == s1.pos0 - 1,
                "seq1 draft KV not caught up with the prompt");

        // global cap 3, per-sequence caps 1 / 2 / unlimited / mixed
        set_dp(specA, 0, 1, s0);  set_dp(specA, 1, 1, s1);
        check_round("A1 caps=1,1",   specA, s0, s1, 1, 1, 1, 1, mem_tgt, mem_dft);

        set_dp(specA, 0, 2, s0);  set_dp(specA, 1, 2, s1);
        check_round("A2 caps=2,2",   specA, s0, s1, 2, 2, 2, 2, mem_tgt, mem_dft);

        set_dp(specA, 0, -1, s0); set_dp(specA, 1, -1, s1);
        check_round("A3 caps=unlim", specA, s0, s1, 3, 3, 3, 3, mem_tgt, mem_dft);

        set_dp(specA, 0, 1, s0);  set_dp(specA, 1, 2, s1);
        check_round("A4 caps=1,2",   specA, s0, s1, 1, 2, 1, 2, mem_tgt, mem_dft);
        common_speculative_free(specA);

        // ------------------------------------------------ group B: n_min=2 > cap 1
        common_speculative * specB = make_spec(ctx_tgt.get(), ctx_dft.get(), 2, 0.0f);
        // drop the stale draft rows of group A, extend both prompts so the fresh drafter
        // gets its own target hidden states through process()
        llama_memory_seq_rm(mem_dft, 0, s0.pos0, -1);
        llama_memory_seq_rm(mem_dft, 1, s1.pos0, -1);
        for (llama_token t = 31; t <= 34; t++) {
            s0.prompt.push_back(t);
        }
        for (llama_token t = 41; t <= 44; t++) {
            s1.prompt.push_back(t);
        }
        const llama_pos ext0 = s0.pos0;
        const llama_pos ext1 = s1.pos0;
        s0.pos0 = (llama_pos) s0.prompt.size(); // 16
        s1.pos0 = (llama_pos) s1.prompt.size(); // 12
        std::printf("B setup: dft pos_max after rm: seq0=%d seq1=%d; tgt pos_max seq0=%d seq1=%d; pos0 -> %d/%d\n",
                    (int) llama_memory_seq_pos_max(mem_dft, 0), (int) llama_memory_seq_pos_max(mem_dft, 1),
                    (int) llama_memory_seq_pos_max(mem_tgt, 0), (int) llama_memory_seq_pos_max(mem_tgt, 1),
                    (int) s0.pos0, (int) s1.pos0);
        feed(specB, ext0, ext1);
        common_speculative_begin(specB, 0, s0.prompt);
        common_speculative_begin(specB, 1, s1.prompt);
        const llama_pos pm0 = llama_memory_seq_pos_max(mem_dft, 0);
        const llama_pos pm1 = llama_memory_seq_pos_max(mem_dft, 1);
        std::printf("B setup: after feed dft pos_max seq0=%d seq1=%d (expect %d/%d) tgt seq0=%d seq1=%d\n",
                    (int) pm0, (int) pm1, (int) (s0.pos0 - 1), (int) (s1.pos0 - 1),
                    (int) llama_memory_seq_pos_max(mem_tgt, 0), (int) llama_memory_seq_pos_max(mem_tgt, 1));
        require(pm0 == s0.pos0 - 1,
                "seq0 catch-up after extension failed: pos_max=" + std::to_string(pm0) +
                ", expected " + std::to_string(s0.pos0 - 1));

        // cap 1 < n_min 2: drafting stops after 1 decode, then the result is discarded
        set_dp(specB, 0, 1, s0);  set_dp(specB, 1, 3, s1);
        check_round("B1 n_min=2 caps=1,3", specB, s0, s1, 1, 3, 0, 3, mem_tgt, mem_dft);
        common_speculative_free(specB);

        // ------------------------------------------------ group C: confidence stop
        common_speculative * specC = make_spec(ctx_tgt.get(), ctx_dft.get(), 0, 1.1f);
        llama_memory_seq_rm(mem_dft, 0, s0.pos0, -1);
        llama_memory_seq_rm(mem_dft, 1, s1.pos0, -1);
        for (llama_token t = 51; t <= 54; t++) {
            s0.prompt.push_back(t);
        }
        for (llama_token t = 61; t <= 64; t++) {
            s1.prompt.push_back(t);
        }
        const llama_pos ext2 = s0.pos0;
        const llama_pos ext3 = s1.pos0;
        s0.pos0 = (llama_pos) s0.prompt.size(); // 20
        s1.pos0 = (llama_pos) s1.prompt.size(); // 16
        feed(specC, ext2, ext3);
        common_speculative_begin(specC, 0, s0.prompt);
        common_speculative_begin(specC, 1, s1.prompt);
        require(llama_memory_seq_pos_max(mem_dft, 0) == s0.pos0 - 1, "seq0 catch-up after 2nd extension failed");

        // p_min above 1.0 rejects the very first candidate: anchor decode only, empty result
        set_dp(specC, 0, 3, s0);  set_dp(specC, 1, 3, s1);
        check_round("C1 p_min=1.1 caps=3,3", specC, s0, s1, 1, 1, 0, 0, mem_tgt, mem_dft);
        common_speculative_free(specC);

        llama_free(ctx_dft.release());
        llama_free(ctx_tgt.release());
        llama_model_free(model.release());
        llama_backend_free();
        std::printf("PASS test-mtp-seq-nmax\n");
        return EXIT_SUCCESS;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL test-mtp-seq-nmax: %s\n", e.what());
        return EXIT_FAILURE;
    }
}