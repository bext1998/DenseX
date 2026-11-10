// CPU-only oracle for a context-local MTP KV window. No model files or backend discovery.
// The reference has the same attention mask W, but enough retention/cells to never evict.
// A scalar oracle also checks the mask and attention, independently of both caches.
#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-io.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t W = 247;
constexpr uint32_t R = 8;
constexpr uint32_t U = 1;
constexpr uint32_t C = ((W + R + U + 255)/256)*256;
constexpr uint32_t REF_C = 2048;
constexpr int D = 8;
constexpr int HQ = 2;
constexpr int HKV = 1;
constexpr int VOCAB = 7;
constexpr llama_pos ANCHOR = 3*C - 7;
static_assert(C == 256 && W + R + U == C, "exercise the tight padded capacity");

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// Integer-based inputs are reproducible across standard libraries. Different variants at
// the same absolute position make stale speculative/verify rows observable after seq_rm.
float value(llama_pos pos, int variant, int lane, uint32_t salt) {
    uint32_t x = uint32_t(pos)*1664525u + uint32_t(variant)*1013904223u + uint32_t(lane)*747796405u + salt;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return float(int(x % 2001u) - 1000)/1000.0f;
}

float projection(int lane, int token) {
    return value(token, 0, lane, 917u)*0.25f;
}

struct batch {
    llama_ubatch ub{};
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq;
    std::vector<llama_seq_id *> seq;
    llama_seq_id id = 0;

    batch(llama_pos first, uint32_t count) : pos(count), n_seq(count, 1), seq(count, &id) {
        for (uint32_t i = 0; i < count; ++i) {
            pos[i] = first + i;
        }
        ub.b_equal_seqs = 1;
        ub.n_tokens = count;
        ub.n_seq_tokens = count;
        ub.n_seqs = 1;
        ub.n_seqs_unq = 1;
        ub.n_pos = 1;
        ub.pos = pos.data();
        ub.n_seq_id = n_seq.data();
        ub.seq_id = seq.data();
        ub.seq_id_unq = &id;
    }
};

struct result {
    std::vector<float> attention;
    std::vector<float> logits; // a fixed tiny projection, not full-model draft logits
};

struct cache {
    std::unique_ptr<llama_kv_cache> kv;
    bool v_trans;
    uint32_t window;
    int wraps = 0;
    int last_slot = -1;

    cache(const llama_model & model, uint32_t capacity, uint32_t retention, bool trans, const char * name,
            uint32_t mask_window = W, bool omit_retention = false) : v_trans(trans), window(mask_window) {
        const auto swa = window ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE;
        if (omit_retention) {
            kv.reset(new llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
                        trans, false, true, capacity, 1, 256, window, swa, nullptr, {}, {}, {}, name));
        } else {
            kv.reset(new llama_kv_cache(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
                        trans, false, true, capacity, 1, 256, window, swa, nullptr, {}, {}, {}, name, retention));
        }
    }

    result append(ggml_backend_t cpu, llama_pos pos, int variant) {
        batch input(pos, U);
        const auto sinfo = kv->find_slot(input.ub, false);
        require(!sinfo.empty(), "cache has no slot at position " + std::to_string(pos));
        const int slot = int(sinfo.idxs[0][0]);
        if (last_slot >= 0 && slot < last_slot) {
            ++wraps;
        }
        last_slot = slot;
        kv->apply_ubatch(sinfo, input.ub);
        const uint32_t n_kv = kv->get_n_kv(sinfo);

        const ggml_init_params params{2*1024*1024, nullptr, true};
        context_ptr ctx(ggml_init(params), ggml_free);
        require(bool(ctx), "ggml_init failed");
        auto * gf = ggml_new_graph_custom(ctx.get(), 128, false);
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, HQ, U);
        auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, HKV, U);
        auto * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, HKV, U);
        auto * ki = kv->build_input_k_idxs(ctx.get(), input.ub);
        auto * vi = kv->build_input_v_idxs(ctx.get(), input.ub);
        auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, U, 1, 1);
        auto * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D*HQ, VOCAB);

        // Match the production graph's explicit KV-store-before-attention ordering.
        ggml_build_forward_expand(gf, kv->cpy_k(ctx.get(), k, ki, 0, sinfo));
        ggml_build_forward_expand(gf, kv->cpy_v(ctx.get(), v, vi, 0, sinfo));
        auto * keys = ggml_permute(ctx.get(), kv->get_k(ctx.get(), 0, n_kv, sinfo), 0, 2, 1, 3);
        auto * vals = ggml_permute(ctx.get(), kv->get_v(ctx.get(), 0, n_kv, sinfo), 0, 2, 1, 3);
        if (!v_trans) {
            vals = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), vals));
        }
        auto * queries = ggml_permute(ctx.get(), q, 0, 2, 1, 3);
        auto * scores = ggml_mul_mat(ctx.get(), keys, queries);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        auto * probs = ggml_soft_max_ext(ctx.get(), scores, mask, 1.0f/std::sqrt(float(D)), 0.0f);
        auto * attn = ggml_mul_mat(ctx.get(), vals, probs);
        ggml_prec_set_acc(attn, GGML_PREC_F32);
        attn = ggml_cont_2d(ctx.get(), ggml_permute(ctx.get(), attn, 0, 2, 1, 3), D*HQ, U);
        auto * logits = ggml_mul_mat(ctx.get(), weights, attn);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        ggml_build_forward_expand(gf, logits);

        buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu), ggml_backend_buffer_free);
        require(bool(buf), "CPU graph tensor allocation failed");
        for (int i = 0; i < D*HQ; ++i) {
            static_cast<float *>(q->data)[i] = value(pos, variant, i, 31u);
        }
        for (int i = 0; i < D*HKV; ++i) {
            static_cast<float *>(k->data)[i] = value(pos, variant, i, 71u);
            static_cast<float *>(v->data)[i] = value(pos, variant, i, 137u);
        }
        for (int t = 0; t < VOCAB; ++t) {
            for (int i = 0; i < D*HQ; ++i) {
                static_cast<float *>(weights->data)[t*D*HQ + i] = projection(i, t);
            }
        }
        kv->set_input_k_idxs(ki, &input.ub, sinfo);
        kv->set_input_v_idxs(vi, &input.ub, sinfo);
        kv->set_input_kq_mask(mask, &input.ub, true);

        // Do not call is_masked_swa here: a common mask bug must not make both caches pass.
        const auto & cells = kv->get_cells(0);
        for (uint32_t i = 0; i < n_kv; ++i) {
            const bool keep = !cells.is_empty(i) && cells.seq_has(i, 0) &&
                cells.pos_get(i) <= pos && (window == 0 || pos - cells.pos_get(i) < int(window));
            const float actual = static_cast<const float *>(mask->data)[i];
            require(keep ? actual == 0.0f : std::isinf(actual) && actual < 0.0f,
                    "mask disagrees with absolute-position W at " + std::to_string(pos));
        }
        require(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "CPU graph compute failed");
        result out{std::vector<float>(D*HQ), std::vector<float>(VOCAB)};
        ggml_backend_tensor_get(attn, out.attention.data(), 0, out.attention.size()*sizeof(float));
        ggml_backend_tensor_get(logits, out.logits.data(), 0, out.logits.size()*sizeof(float));
        return out;
    }

    void remove_from(llama_pos pos) {
        require(kv->seq_rm(0, pos, -1), "seq_rm failed");
        const auto & cells = kv->get_cells(0);
        for (uint32_t i = 0; i < cells.size(); ++i) {
            require(cells.is_empty(i) || cells.pos_get(i) < pos, "seq_rm left a future cell");
        }
        require(kv->seq_pos_max(0) == pos - 1, "rollback changed the absolute prefix end");
    }

    bool has_history_for(llama_pos query) const {
        const auto & cells = kv->get_cells(0);
        std::vector<bool> found(W, false);
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, 0)) {
                const auto distance = query - cells.pos_get(i);
                if (distance > 0 && distance < int(W)) {
                    found[distance] = true;
                }
            }
        }
        for (uint32_t distance = 1; distance < W && int(distance) <= query; ++distance) {
            if (!found[distance]) {
                return false;
            }
        }
        return true;
    }
};

result scalar(const std::map<llama_pos, int> & history, llama_pos query, int variant, uint32_t window = W) {
    result out{std::vector<float>(D*HQ, 0.0f), std::vector<float>(VOCAB, 0.0f)};
    for (int h = 0; h < HQ; ++h) {
        std::vector<double> scores;
        std::vector<std::pair<llama_pos, int>> visible;
        for (const auto & row : history) {
            if (row.first <= query && (window == 0 || query - row.first < int(window))) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += double(value(query, variant, h*D + d, 31u))*value(row.first, row.second, d, 71u);
                }
                scores.push_back(dot/std::sqrt(double(D)));
                visible.push_back(row);
            }
        }
        require(!scores.empty(), "empty scalar attention");
        const double max_score = *std::max_element(scores.begin(), scores.end());
        double denom = 0.0;
        for (auto & s : scores) {
            s = std::exp(s - max_score);
            denom += s;
        }
        for (int d = 0; d < D; ++d) {
            double sum = 0.0;
            for (size_t i = 0; i < visible.size(); ++i) {
                sum += scores[i]/denom*value(visible[i].first, visible[i].second, d, 137u);
            }
            out.attention[h*D + d] = float(sum);
        }
    }
    for (int t = 0; t < VOCAB; ++t) {
        double sum = 0.0;
        for (int i = 0; i < D*HQ; ++i) {
            sum += double(out.attention[i])*projection(i, t);
        }
        out.logits[t] = float(sum);
    }
    return out;
}

double max_error(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "result shape mismatch");
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite attention/logits");
        error = std::max(error, std::abs(double(a[i]) - b[i]));
    }
    return error;
}

bool close(const result & a, const result & b, double & worst) {
    const double error = std::max(max_error(a.attention, b.attention), max_error(a.logits, b.logits));
    worst = std::max(worst, error);
    return error <= 2e-5;
}

// Local in-memory adapters for the cache's existing serialization interface.
struct vector_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;

    void write(const void * src, size_t size) override {
        const size_t start = bytes.size();
        bytes.resize(start + size);
        if (size) {
            std::memcpy(bytes.data() + start, src, size);
        }
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t start = bytes.size();
        bytes.resize(start + size);
        if (size) {
            ggml_backend_tensor_get(tensor, bytes.data() + start, offset, size);
        }
    }
    size_t n_bytes() override { return bytes.size(); }
};

struct vector_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t cursor = 0;

    explicit vector_reader(const std::vector<uint8_t> & data) : bytes(data) {}
    void read(void * dst, size_t size) override {
        require(size <= bytes.size() - cursor, "truncated cache metadata");
        if (size) {
            std::memcpy(dst, bytes.data() + cursor, size);
        }
        cursor += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        require(size <= bytes.size() - cursor, "truncated cache tensor");
        if (size) {
            ggml_backend_tensor_set(tensor, bytes.data() + cursor, offset, size);
        }
        cursor += size;
    }
    size_t n_bytes() override { return cursor; }
};

void run_partial_roundtrip(ggml_backend_t cpu, const llama_model & model, bool trans,
        const cache & source, const cache & reference, const cache & old_policy,
        std::map<llama_pos, int> history) {
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    vector_writer saved, ref_saved, old_saved;
    source.kv->state_write(saved, 0, flags);
    reference.kv->state_write(ref_saved, 0, flags);
    old_policy.kv->state_write(old_saved, 0, flags);
    require(saved.n_bytes() > 0, "PARTIAL_ONLY unexpectedly skipped windowed cache");

    cache restored(model, C, W + R, trans, "restored_");
    cache ref_restored(model, REF_C, REF_C, trans, "ref_restored_");
    // Restore the old-policy snapshot into the SAME correct-retention configuration.
    // Thus this negative control isolates state_write filtering, not subsequent eviction.
    cache old_restored(model, C, W + R, trans, "old_snapshot_");
    vector_reader reader(saved.bytes), ref_reader(ref_saved.bytes), old_reader(old_saved.bytes);
    restored.kv->state_read(reader, 0, flags);
    ref_restored.kv->state_read(ref_reader, 0, flags);
    old_restored.kv->state_read(old_reader, 0, flags);
    require(reader.n_bytes() == saved.n_bytes() && ref_reader.n_bytes() == ref_saved.n_bytes() &&
            old_reader.n_bytes() == old_saved.n_bytes(), "cache restore did not consume its snapshot");
    require(restored.kv->seq_pos_max(0) == ANCHOR + llama_pos(R) - 1,
            "PARTIAL_ONLY restore lost the absolute trial endpoint");
    require(source.has_history_for(ANCHOR) && old_policy.has_history_for(ANCHOR),
            "serialization control already lost history before saving");

    restored.remove_from(ANCHOR);
    ref_restored.remove_from(ANCHOR);
    old_restored.remove_from(ANCHOR);
    require(restored.has_history_for(ANCHOR), "PARTIAL_ONLY snapshot lost W+R rollback protection");
    require(ref_restored.has_history_for(ANCHOR), "restored full-capacity reference lost history");
    require(!old_restored.has_history_for(ANCHOR), "old-policy snapshot did not expose missing R protection");
    history.erase(history.lower_bound(ANCHOR), history.end());
    history[ANCHOR] = 3;
    const auto expected = scalar(history, ANCHOR, 3);
    const auto ref = ref_restored.append(cpu, ANCHOR, 3);
    const auto actual = restored.append(cpu, ANCHOR, 3);
    const auto negative = old_restored.append(cpu, ANCHOR, 3);
    double worst = 0.0;
    double negative_error = 0.0;
    require(close(ref, expected, worst), "PARTIAL_ONLY reference/scalar mismatch after rollback");
    require(close(actual, ref, worst) && close(actual, expected, worst),
            "PARTIAL_ONLY restored attention/logits mismatch after rollback");
    require(!close(negative, ref, negative_error), "old-policy snapshot did not produce numerical divergence");
    std::printf("PASS PARTIAL_ONLY v_trans=%d saved_bytes=%zu max_abs=%.9g; "
                "EXPECTED_DIVERGENCE old-retention-snapshot max_abs=%.9g\n",
                int(trans), saved.n_bytes(), worst, negative_error);
}

void run(ggml_backend_t cpu, const llama_model & model, bool trans) {
    cache ring(model, C, W + R, trans, "ring_");
    cache reference(model, REF_C, REF_C, trans, "reference_");
    cache old_policy(model, C, W, trans, "without_R_");
    cache default_policy(model, C, W, trans, "default_retention_", W, true);
    std::map<llama_pos, int> history;
    double worst = 0.0;
    double negative_error = 0.0;
    const auto step = [&](llama_pos pos, int variant, const char * stage, bool check_old) {
        history[pos] = variant;
        const auto expected = scalar(history, pos, variant);
        const auto ref = reference.append(cpu, pos, variant);
        const auto actual = ring.append(cpu, pos, variant);
        const auto negative = old_policy.append(cpu, pos, variant);
        const auto default_result = default_policy.append(cpu, pos, variant);
        const std::string where = std::string(stage) + " pos=" + std::to_string(pos);
        require(default_result.attention == negative.attention && default_result.logits == negative.logits,
                "omitted retention differs from explicit W: " + where);
        require(close(ref, expected, worst), "reference/scalar mismatch: " + where);
        require(close(actual, ref, worst), "ring/reference mismatch: " + where);
        require(close(actual, expected, worst), "ring/scalar mismatch: " + where);
        if (check_old) {
            require(close(negative, ref, worst), "negative diverged before the second rollback: " + where);
        } else {
            close(negative, ref, negative_error);
        }
        require(reference.kv->seq_pos_min(0) == 0, "reference evicted history");
        require(reference.kv->get_cells(0).get_used() == history.size(), "reference is not a full-history cache");
    };
    const auto rollback = [&](llama_pos pos) {
        ring.remove_from(pos);
        reference.remove_from(pos);
        old_policy.remove_from(pos);
        default_policy.remove_from(pos);
        history.erase(history.lower_bound(pos), history.end());
    };

    // ANCHOR=761: two complete physical wraps, then seven tail slots before the next wrap.
    // R=8 crosses the next wrap. U=1 prevents padding/large batches from hiding the bug.
    for (llama_pos p = 0; p < ANCHOR; ++p) {
        step(p, 0, "cache oracle", true);
    }
    require(ring.wraps >= 2, "oracle did not actually wrap the physical ring twice");
    require(old_policy.wraps >= 2, "negative control did not wrap twice");
    for (uint32_t i = 0; i < R; ++i) {
        step(ANCHOR + i, 1, "speculate R8", true);
    }
    run_partial_roundtrip(cpu, model, trans, ring, reference, old_policy, history);
    rollback(ANCHOR); // discard trial KV before replaying target verify inputs
    require(ring.has_history_for(ANCHOR), "first rollback lost the accepted prefix");
    require(old_policy.has_history_for(ANCHOR), "negative control failed too early");
    for (uint32_t i = 0; i < R; ++i) {
        step(ANCHOR + i, 2, "reprocess verify", true);
    }
    rollback(ANCHOR); // zero accepted draft tokens: reject verify tail a second time
    require(ring.has_history_for(ANCHOR), "second rollback lost W history despite retention W+R");
    require(!old_policy.has_history_for(ANCHOR), "negative control did not expose missing rollback history");
    // With the old policy the replay consumes early physical cells again and purges
    // absolute positions 515..519; query 761 still needs them (761 - W + 1 = 515).

    // Replace rejected rows with different values; stale rows must neither remain visible nor win.
    step(ANCHOR, 3, "after second rollback", false);
    require(negative_error > 2e-5, "negative control lost cells but did not produce numerical divergence");
    std::printf("EXPECTED_DIVERGENCE v_trans=%d stage=after-second-rollback pos=%d retention=%u "
                "max_abs=%.9g (same C=%u; retention W+R matches reference)\n",
                int(trans), int(ANCHOR), W, negative_error, C);
    for (llama_pos p = ANCHOR + 1; p < ANCHOR + 17; ++p) {
        step(p, 0, "continue accepted prefix", false);
    }
    std::printf("PASS v_trans=%d W=%u R=%u U=%u C=%u wraps=%d max_abs=%.9g; "
                "expected divergence retention=W max_abs=%.9g\n",
                int(trans), W, R, U, C, ring.wraps, worst, negative_error);
}

void run_unwindowed(ggml_backend_t cpu, const llama_model & model, bool trans) {
    cache explicit_zero(model, C, 0, trans, "zero_explicit_", 0);
    cache default_zero(model, C, 0, trans, "zero_default_", 0, true);
    std::map<llama_pos, int> history;
    double worst = 0.0;
    // Exceed W=247 to expose accidental windowing, but fill only C cells (no eviction).
    for (llama_pos p = 0; p < llama_pos(C); ++p) {
        history[p] = 0;
        const auto expected = scalar(history, p, 0, 0);
        const auto actual = explicit_zero.append(cpu, p, 0);
        const auto omitted = default_zero.append(cpu, p, 0);
        require(close(actual, expected, worst), "W=0 dense causal attention differs from scalar oracle");
        require(actual.attention == omitted.attention && actual.logits == omitted.logits,
                "W=0 omitted retention differs from explicit zero");
    }
    std::printf("PASS W=0 v_trans=%d omitted/explicit retention and dense scalar oracle max_abs=%.9g\n",
                int(trans), worst);
}
} // namespace

int main() {
    try {
        const auto cp = llama_context_default_params();
        require(cp.kv_window == 0 && cp.kv_rollback == 0, "public window/rollback defaults must remain zero");
        // Explicit CPU init only. In particular, never load_all or enumerate accelerator devices.
        backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(cpu), "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        ggml_backend_dev_t cpu_only[] = {ggml_backend_get_device(cpu.get()), nullptr};
        mp.devices = cpu_only;
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, mp));
        require(bool(model), "failed to create tensor-free Qwen35 model shell");
        model->hparams = llama_hparams{};
        auto & hp = model->hparams;
        hp.n_ctx_train = REF_C;
        hp.n_embd = D*HQ;
        hp.n_layer_all = 1;
        hp.n_layer_nextn = 1;
        hp.n_head_arr[0] = HQ;
        hp.n_head_kv_arr[0] = HKV;
        hp.n_embd_head_k_full = D;
        hp.n_embd_head_v_full = D;
        // Keep shared model hparams dense: W belongs to each cache, not to the model.
        require(hp.n_swa == 0 && hp.swa_type == LLAMA_SWA_TYPE_NONE, "unexpected model SWA defaults");
        run(cpu.get(), *model, false);
        run(cpu.get(), *model, true);
        run_unwindowed(cpu.get(), *model, false);
        run_unwindowed(cpu.get(), *model, true);
        require(hp.n_swa == 0 && hp.swa_type == LLAMA_SWA_TYPE_NONE, "cache modified shared model SWA");
        return EXIT_SUCCESS;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL test-draft-kv-window: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
