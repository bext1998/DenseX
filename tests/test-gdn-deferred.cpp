// DenseX: semantic check for GGML_OP_GATED_DELTA_NET_DEFERRED (CPU backend only).
//
// Proves that the deferred op equals sequential processing:
//   A: the plain op ggml_gated_delta_net processes [pending rows 0..j-1 | this batch's T tokens] at once;
//      its last T attention rows and (for the state check) the final state of a run over only the first
//      j pending rows are the references.
//   B: ggml_gated_delta_net_deferred with (state = s0, pend, j, batch).
// A and B must match; bitwise equality and the maximum absolute error are reported, never hidden by a
// relaxed tolerance.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

struct gdn_dims {
    int64_t S  = 128;   // head size
    int64_t Hk = 16;    // compact k/q heads
    int64_t Hv = 48;    // v heads
    int64_t B  = 1;     // sequences
    int64_t T;          // tokens of this batch
    int64_t C  = 17;    // pending capacity

    int64_t R() const { return S*Hk + S*Hv + Hv + Hv; }
    int64_t attn_elems() const { return S*Hv*T*B; }
    int64_t state_elems() const { return S*S*Hv*B; }
    int64_t pend_elems() const { return R()*C*B; }
};

// fills [0, n) with values in [min, max)
static void fill_random(float * dst, int64_t n, float min, float max, std::mt19937 & gen) {
    std::uniform_real_distribution<float> dis(min, max);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = dis(gen);
    }
}

static void fill_pend_row(float * row, const gdn_dims & d, std::mt19937 & gen) {
    fill_random(row, d.S*d.Hk, -0.3f, 5.0f, gen);                                  // compact k
    fill_random(row + d.S*d.Hk, d.S*d.Hv, -0.3f, 5.0f, gen);                       // v
    fill_random(row + d.S*(d.Hk + d.Hv), d.Hv, -20.0f, -1e-4f, gen);               // scalar log gate (negative)
    fill_random(row + d.S*(d.Hk + d.Hv) + d.Hv, d.Hv, 1e-3f, 1.0f - 1e-3f, gen);   // beta in (0, 1)
}

// computes `out` of `graph` on the CPU backend; the graph inputs are plain tensors with host data
static void compute_graph(ggml_context * ctx, ggml_backend_t backend, ggml_cgraph * graph) {
    static ggml_gallocr_t allocr = nullptr;
    if (!allocr) {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_gallocr_alloc_graph(allocr, graph);
    ggml_backend_graph_compute(backend, graph);
    GGML_UNUSED(ctx);
}

struct compare_result {
    bool     bitwise;
    int64_t  n_diff_bits;
    double   max_abs;
    int64_t  first_diff;
    int64_t  n;
};

static compare_result compare(const float * a, const float * b, int64_t n) {
    compare_result res { true, 0, 0.0, -1, n };
    for (int64_t i = 0; i < n; ++i) {
        uint32_t ba, bb;
        memcpy(&ba, a + i, 4);
        memcpy(&bb, b + i, 4);
        if (ba != bb) {
            res.bitwise = false;
            ++res.n_diff_bits;
            if (res.first_diff < 0) {
                res.first_diff = i;
            }
        }
        const double diff = std::abs((double) a[i] - (double) b[i]);
        if (diff > res.max_abs) {
            res.max_abs = diff;
        }
    }
    return res;
}

// builds the reference graph A: the plain op over [pend rows 0..j-1 | the first n_batch batch tokens], K = 1.
// the final state is the reference for the deferred op's state output with n_batch = p = max(0, T - C);
// with n_batch = T the last T attention rows are the reference for the deferred op's attention output.
static ggml_tensor * build_ref_old_op(ggml_context * ctx, const gdn_dims & d,
        const float * s0, const float * q_b, const float * k_b, const float * v_b,
        const float * g_b, const float * beta_b, const float * pend,
        int64_t j, int64_t n_batch,
        ggml_tensor ** out_attn, ggml_tensor ** out_state) {
    const int64_t n_tok = j + n_batch;
    if (n_tok == 0) {
        *out_attn  = nullptr;
        *out_state = nullptr;
        return nullptr;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hk, n_tok, d.B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hk, n_tok, d.B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hv, n_tok, d.B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, d.Hv, n_tok, d.B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, d.Hv, n_tok, d.B);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.S, d.Hv, d.B);

    memcpy(state->data, s0, ggml_nbytes(state));

    // q is not part of the state recurrence; the pending rows reuse the batch's first q row
    memcpy((char *) q->data + 0*ggml_row_size(GGML_TYPE_F32, d.S), q_b, ggml_row_size(GGML_TYPE_F32, d.S)*d.Hk);
    for (int64_t t = 0; t < j; ++t) {
        const float * p_row = pend + t*d.R();
        memcpy((char *) k->data + t*k->nb[2], p_row, d.S*d.Hk*sizeof(float));
        memcpy((char *) v->data + t*v->nb[2], p_row + d.S*d.Hk, d.S*d.Hv*sizeof(float));
        memcpy((char *) g->data + t*g->nb[2], p_row + d.S*(d.Hk + d.Hv), d.Hv*sizeof(float));
        memcpy((char *) beta->data + t*beta->nb[2], p_row + d.S*(d.Hk + d.Hv) + d.Hv, d.Hv*sizeof(float));
    }
    if (n_batch > 0) {
        memcpy((char *) q->data + j*q->nb[2], q_b, ggml_nbytes(q) - j*q->nb[2]);
        memcpy((char *) k->data + j*k->nb[2], k_b, ggml_nbytes(k) - j*k->nb[2]);
        memcpy((char *) v->data + j*v->nb[2], v_b, ggml_nbytes(v) - j*v->nb[2]);
        memcpy((char *) g->data + j*g->nb[2], g_b, ggml_nbytes(g) - j*g->nb[2]);
        memcpy((char *) beta->data + j*beta->nb[2], beta_b, ggml_nbytes(beta) - j*beta->nb[2]);
    }

    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);

    // output layout: attn rows first (one S*Hv row per token), then the K = 1 state snapshot
    *out_attn  = ggml_view_1d(ctx, out, d.S*d.Hv*n_tok, 0);
    *out_state = ggml_view_1d(ctx, out, d.S*d.S*d.Hv*d.B, d.S*d.Hv*n_tok*sizeof(float));
    return out;
}

static bool run_case(int64_t T, int64_t C, const std::set<int64_t> & js) {
    gdn_dims d;
    d.T = T;
    d.C = C;
    // committed prefix length: the state output covers [pending | the first p batch tokens]
    const int64_t p0 = T > C ? T - C : 0;

    std::mt19937 gen(1234 + (int) T);

    const size_t sz_state = d.state_elems()*sizeof(float);
    const size_t sz_q     = d.S*d.Hk*d.T*sizeof(float);
    const size_t sz_v     = d.S*d.Hv*d.T*sizeof(float);

    std::vector<float> s0(d.state_elems());
    std::vector<float> q_b(d.S*d.Hk*d.T), k_b(d.S*d.Hk*d.T), v_b(d.S*d.Hv*d.T);
    std::vector<float> g_b(d.Hv*d.T), beta_b(d.Hv*d.T);
    std::vector<float> pend(d.pend_elems());

    // base state: random but bounded so the recurrence stays sane
    fill_random(s0.data(), d.state_elems(), -1.0f, 1.0f, gen);
    fill_random(q_b.data(), d.S*d.Hk*d.T, -1.0f, 1.0f, gen);
    fill_random(k_b.data(), d.S*d.Hk*d.T, -0.3f, 5.0f, gen);
    fill_random(v_b.data(), d.S*d.Hv*d.T, -0.3f, 5.0f, gen);
    fill_random(g_b.data(), d.Hv*d.T, -20.0f, -1e-4f, gen);
    fill_random(beta_b.data(), d.Hv*d.T, 1e-3f, 1.0f - 1e-3f, gen);
    for (int64_t c = 0; c < d.C; ++c) {
        fill_pend_row(pend.data() + c*d.R(), d, gen);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();

    bool all_ok = true;
    for (int64_t j : js) {
        // graph B: the deferred op
        struct ggml_init_params ip = { 256ull*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hk, d.T, d.B);
        ggml_tensor * k_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hk, d.T, d.B);
        ggml_tensor * v_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.Hv, d.T, d.B);
        ggml_tensor * g_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, d.Hv, d.T, d.B);
        ggml_tensor * beta_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, d.Hv, d.T, d.B);
        ggml_tensor * state_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d.S, d.S, d.Hv, d.B);
        ggml_tensor * pend_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.R(), d.C, d.B);
        ggml_tensor * n_pend_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, d.B);

        memcpy(q_t->data, q_b.data(), sz_q);
        memcpy(k_t->data, k_b.data(), d.S*d.Hk*d.T*sizeof(float));
        memcpy(v_t->data, v_b.data(), sz_v);
        memcpy(g_t->data, g_b.data(), d.Hv*d.T*sizeof(float));
        memcpy(beta_t->data, beta_b.data(), d.Hv*d.T*sizeof(float));
        memcpy(state_t->data, s0.data(), sz_state);
        memcpy(pend_t->data, pend.data(), d.pend_elems()*sizeof(float));
        ((int32_t *) n_pend_t->data)[0] = (int32_t) j;
        GGML_ASSERT(j <= C);

        ggml_tensor * out_b = ggml_gated_delta_net_deferred(ctx, q_t, k_t, v_t, g_t, beta_t, state_t, pend_t, n_pend_t, d.C);
        ggml_tensor * b_attn  = ggml_view_1d(ctx, out_b, d.attn_elems(), 0);
        ggml_tensor * b_state = ggml_view_1d(ctx, out_b, d.state_elems(), d.attn_elems()*sizeof(float));

        // graph A1: plain op over [pend 0..j-1 | the whole batch]; references the last T attention rows
        ggml_tensor * a1_attn = nullptr;
        ggml_tensor * a1_state = nullptr;
        ggml_tensor * out_a1 = build_ref_old_op(ctx, d, s0.data(), q_b.data(), k_b.data(), v_b.data(),
                                                g_b.data(), beta_b.data(), pend.data(), j, d.T, &a1_attn, &a1_state);

        // graph A2: plain op over [pend 0..j-1 | the first p batch tokens]; the final state is the
        // state output reference
        ggml_tensor * a2_attn = nullptr;
        ggml_tensor * a2_state = nullptr;
        ggml_tensor * out_a2 = build_ref_old_op(ctx, d, s0.data(), q_b.data(), k_b.data(), v_b.data(),
                                                g_b.data(), beta_b.data(), pend.data(), j, p0, &a2_attn, &a2_state);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out_b);
        if (out_a1) { ggml_build_forward_expand(graph, out_a1); }
        if (out_a2) { ggml_build_forward_expand(graph, out_a2); }
        compute_graph(ctx, backend, graph);

        const float * b_attn_d  = (const float *) b_attn->data;
        const float * a_attn_d  = (const float *) a1_attn->data + (d.S*d.Hv)*j; // last T rows
        const float * b_state_d = (const float *) b_state->data;

        const compare_result attn_cmp = compare(a_attn_d, b_attn_d, d.attn_elems());
        // j = 0: no pending rows to replay, state_j must equal s0 exactly; the reference run is empty
        compare_result state_cmp { true, 0, 0.0, d.state_elems() };
        if (a2_state) {
            state_cmp = compare((const float *) a2_state->data, b_state_d, d.state_elems());
        } else {
            state_cmp = compare(s0.data(), b_state_d, d.state_elems());
        }

        printf("T=%2lld j=%2lld  attn: bitwise=%s max_abs=%.3e | state_j: bitwise=%s max_abs=%.3e\n",
               (long long) T, (long long) j,
               attn_cmp.bitwise ? "yes" : "NO ", attn_cmp.max_abs,
               state_cmp.bitwise ? "yes" : "NO ", state_cmp.max_abs);
        if (!state_cmp.bitwise && state_cmp.first_diff >= 0) {
            const int64_t i = state_cmp.first_diff;
            const float * av = a2_state ? (const float *) a2_state->data : s0.data();
            printf("    state first diff idx=%lld: ref=%.6e got=%.6e | idx-1 ref=%.6e got=%.6e\n",
                   (long long) i, av[i], b_state_d[i],
                   i > 0 ? av[i-1] : 0.0f, i > 0 ? b_state_d[i-1] : 0.0f);
        }

        // verify pend_new: rows 0..min(T,C)-1 hold the batch's LAST min(T,C) token inputs
        {
            const int64_t n_rows = d.T < d.C ? d.T : d.C;
            const ggml_tensor * out_t = out_b; // 1-D packed: attn | state | pend_new
            const float * pn = (const float *) out_t->data + d.attn_elems() + d.state_elems();
            const int64_t R = d.R();
            int64_t pn_diff = 0;
            double pn_max = 0.0;
            for (int64_t r = 0; r < n_rows; ++r) {
                const int64_t tok = d.T - n_rows + r; // batch token index
                const float * row = pn + r*R;
                for (int64_t h = 0; h < d.Hk; ++h) {
                    for (int64_t i = 0; i < d.S; ++i) {
                        const float ref = k_b[tok*d.Hk*d.S + h*d.S + i];
                        const float got = row[h*d.S + i];
                        const double diff = std::abs((double) ref - (double) got);
                        if (diff > pn_max) pn_max = diff;
                        if (diff != 0.0) ++pn_diff;
                    }
                }
                for (int64_t h = 0; h < d.Hv; ++h) {
                    for (int64_t i = 0; i < d.S; ++i) {
                        const float ref = v_b[tok*d.Hv*d.S + h*d.S + i];
                        const float got = row[d.Hk*d.S + h*d.S + i];
                        const double diff = std::abs((double) ref - (double) got);
                        if (diff > pn_max) pn_max = diff;
                        if (diff != 0.0) ++pn_diff;
                    }
                    {
                        const float ref = g_b[tok*d.Hv + h];
                        const float got = row[d.Hk*d.S + d.Hv*d.S + h];
                        const double diff = std::abs((double) ref - (double) got);
                        if (diff > pn_max) pn_max = diff;
                        if (diff != 0.0) ++pn_diff;
                    }
                    {
                        const float ref = beta_b[tok*d.Hv + h];
                        const float got = row[d.Hk*d.S + d.Hv*d.S + d.Hv + h];
                        const double diff = std::abs((double) ref - (double) got);
                        if (diff > pn_max) pn_max = diff;
                        if (diff != 0.0) ++pn_diff;
                    }
                }
            }
            printf("    pend_new: diffs=%lld max_abs=%.3e\n", (long long) pn_diff, pn_max);
            all_ok = all_ok && (pn_diff == 0);
        }
        all_ok = all_ok && attn_cmp.bitwise && state_cmp.bitwise;

        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return all_ok;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    setbuf(stdout, NULL); // unbuffered: report progress even if a later step crashes

    bool ok = true;
    for (auto [T, C] : { std::pair<int64_t, int64_t>{1, 17}, {2, 17}, {4, 17}, {9, 17}, {17, 17},
                         {18, 17}, {33, 17}, {17, 9}, {33, 9} }) {
        std::set<int64_t> js = { 0, 1, 2, T - 1, T, C };
        for (auto it = js.begin(); it != js.end(); ) {
            it = *it > C ? js.erase(it) : std::next(it); // n_pend cannot exceed the capacity
        }
        ok = run_case(T, C, js) && ok;
    }
    // full sweep of j for one case
    {
        std::set<int64_t> js;
        for (int64_t j = 0; j <= 17; ++j) {
            js.insert(j);
        }
        ok = run_case(4, 17, js) && ok;
    }

    printf("%s\n", ok ? "GDN_DEFERRED_SEMANTICS_OK" : "GDN_DEFERRED_SEMANTICS_FAILED");
    return ok ? 0 : 1;
}
