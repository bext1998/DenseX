// DenseX (issue #5): a CPU-only oracle for what one KV execution is allowed to touch.
//
// llama_kv_cache::get_access() and get_access_shift() tell llama_kv_storage which cells of which streams an
// execution reads and writes; a lazily mapped backing (E3) maps exactly those ranges before the graph runs. The
// test does not reuse the formulas that fill those lists. It builds the real graph pieces - the get_k / get_v views
// and the cpy_k / cpy_v set_rows writes - and reads the touched (stream, cell) pairs back out of the tensors
// themselves: ne, nb, view_offs, element size, the values the idxs inputs were filled with, and where the marker
// rows the ubatch stored actually ended up inside the K/V buffers. Every one of them has to fall inside a declared
// range of the matching kind (read or write).
//
// What the parts pin down:
//   - the graph views read cells [0, n_kv) of the streams the ubatch spans - padded, and with the free holes in the
//     middle included, not just the cells the ubatch occupies;
//   - the set_rows writes stay inside the cells the list declares for that stream, in every layer, even when the
//     cells of a stream are not contiguous;
//   - K and V keep their own strides: n_embd_head_k != n_embd_head_v and a different n_head_kv per layer, so a V
//     row index computed from the K layout lands in another cell and fails the marker check;
//   - a transposed V also rewrites the embedding padding of a narrower layer, up to v->ne[0];
//   - the K-shift declares the whole capacity of every stream, and the rotated view it runs on lies inside it.
//
// Like test-draft-kv-window.cpp: a tensor-free Qwen35 model shell plus an explicit CPU backend. No model files, no
// device discovery, no GPU.
#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-kv-storage.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t N_LAYER = 2;
constexpr uint32_t HKV[N_LAYER] = { 1, 2 };  // variable width per layer: a transposed V has to pad the narrow one
constexpr uint32_t HQ   = 2;                 // query heads; the cache never reads them
constexpr uint32_t DK   = 8;                 // n_embd_head_k
constexpr uint32_t DV   = 6;                 // n_embd_head_v != DK, so a swapped stride changes the layout
constexpr uint32_t ROT  = 6;                 // n_rot_full, a strict subset of a K head
constexpr uint32_t NPAD = 256;               // n_pad; get_n_kv() never reports less than this
constexpr uint32_t CAP  = 512;               // cells per stream
constexpr uint32_t T1   = 32;                // tokens per stream in the first ubatch
constexpr uint32_t T2   = 2;                 // tokens per stream in the one that lands in the holes

static_assert(CAP % NPAD == 0, "the capacity has to be a whole number of padded windows");
static_assert(ROT < DK, "the rotated part of a K head has to be a strict subset of it");
static_assert(T2 < T1, "the second ubatch has to fit in the holes the first one leaves");

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// integer arithmetic, reproducible across standard libraries
float marker(int seed, int token, int lane) {
    uint32_t x = (uint32_t) seed*2654435761u + (uint32_t) token*40503u + (uint32_t) lane*2246822509u + 374761393u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return float(int(x % 200003u) - 100001)/100001.0f;
}

// a ubatch of n_seq sequences with per_seq tokens each, one sequence per stream
struct batch {
    llama_ubatch              ub{};
    std::vector<llama_pos>    pos;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   seq_id_unq;

    batch(uint32_t n_seq, uint32_t per_seq, llama_pos first) {
        const uint32_t n = n_seq*per_seq;

        pos.resize(n);
        n_seq_id.assign(n, 1);
        seq_id_data.resize(n);
        seq_id.resize(n);
        seq_id_unq.resize(n_seq);

        for (uint32_t s = 0; s < n_seq; ++s) {
            seq_id_unq[s] = (llama_seq_id) s;
            for (uint32_t i = 0; i < per_seq; ++i) {
                const uint32_t t = s*per_seq + i;
                pos[t]         = first + (llama_pos) i;
                seq_id_data[t] = (llama_seq_id) s;
                seq_id[t]      = &seq_id_data[t];
            }
        }

        ub.b_equal_seqs = n_seq == 1 ? 1 : 0;
        ub.n_tokens     = n;
        ub.n_seq_tokens = per_seq;
        ub.n_seqs       = n_seq;
        ub.n_seqs_unq   = n_seq;
        ub.n_pos        = 1;
        ub.pos          = pos.data();
        ub.n_seq_id     = n_seq_id.data();
        ub.seq_id       = seq_id.data();
        ub.seq_id_unq   = seq_id_unq.data();
    }
};

// Which cell a byte of one of the storage's K / V tensors belongs to, read only off the tensor:
//   K and non-transposed V: [n_embd_k_gqa | n_embd_v_gqa, kv_size, n_stream] - a cell is one contiguous row
//   transposed V:           [n_embd_v_gqa_max, kv_size, n_stream]            - inside a stream, element e of a row
//                           j is cell e % kv_size, so the cells of one j are contiguous
// kv_size and the stream count are the tensor's own dims 1 and 2; nothing here repeats the formulas that built the
// access list.
struct layout {
    const ggml_tensor * base;
    const bool          trans;
    const size_t        el;
    const size_t        row;         // bytes from one cell to the next, non-transposed
    const size_t        per_stream;
    const uint32_t      kv_size;
    const uint32_t      n_stream;

    layout(const ggml_tensor * t, bool trans_) :
        base(t->view_src ? t->view_src : t),
        trans(trans_),
        el(ggml_element_size(base)),
        row(base->nb[1]),
        per_stream(base->nb[2]),
        kv_size((uint32_t) base->ne[1]),
        n_stream((uint32_t) base->ne[2]) {
        require(ggml_backend_buffer_is_host(base->buffer), "the K/V buffer is expected to be host memory");
        require(base->data != nullptr, "the K/V tensor has no data");
        require(base->nb[0] == el, "unexpected element stride in the K/V tensor");
        require(row == ggml_row_size(base->type, base->ne[0]), "the rows of the K/V tensor are not contiguous");
        require(per_stream == row*base->ne[1], "the streams of the K/V tensor are not contiguous blocks");
    }

    void map(size_t byte, uint32_t & strm, uint32_t & cell) const {
        strm = (uint32_t) (byte / per_stream);
        const size_t rest = byte % per_stream;

        cell = trans ? (uint32_t) ((rest / el) % kv_size) : (uint32_t) ((rest / row) % kv_size);

        require(strm < n_stream && cell < kv_size, "byte outside the K/V buffer");
    }

    // `lane` is the element inside the cell (a merged head element, or the embedding index j of a transposed V)
    size_t byte_of(uint32_t strm, uint32_t cell, uint32_t lane) const {
        return (size_t) strm*per_stream
            + (trans ? ((size_t) lane*kv_size + cell)*el : (size_t) cell*row + (size_t) lane*el);
    }
};

// every element byte of a tensor, from its own ne / nb / view_offs
template <typename F>
void for_each_element(const ggml_tensor * t, F && f) {
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    f((size_t) t->view_offs + (size_t) i0*t->nb[0] + (size_t) i1*t->nb[1]
                          + (size_t) i2*t->nb[2] + (size_t) i3*t->nb[3]);
                }
            }
        }
    }
}

bool covered(const std::vector<llama_kv_storage::access> & acc, bool write, uint32_t strm, uint32_t cell) {
    for (const auto & a : acc) {
        if (a.write == write && a.strm == strm && cell >= a.cells.first && cell < a.cells.second) {
            return true;
        }
    }

    return false;
}

// counts the cells a set of graph pieces touches, and remembers the first one the list failed to declare
struct tally {
    std::vector<std::vector<bool>> seen;
    size_t      cells  = 0;
    size_t      bytes  = 0;
    std::string miss;

    explicit tally(uint32_t n_stream) : seen(n_stream, std::vector<bool>(CAP, false)) {}

    void add(const std::vector<llama_kv_storage::access> & acc, bool write, const layout & l, size_t byte,
            const char * what) {
        uint32_t strm = 0;
        uint32_t cell = 0;

        l.map(byte, strm, cell);

        bytes += l.el;
        if (!seen[strm][cell]) {
            seen[strm][cell] = true;
            ++cells;
        }
        if (miss.empty() && !covered(acc, write, strm, cell)) {
            miss = std::string(what) + ": cell " + std::to_string(cell) + " of stream " + std::to_string(strm)
                + " is outside every declared " + (write ? "write" : "read") + " range";
        }
    }

    std::string describe() const {
        return std::to_string(bytes) + " bytes / " + std::to_string(cells) + " cells"
            + (miss.empty() ? "" : "; first miss: " + miss);
    }
};

void check_bytes(tally & t, const std::vector<llama_kv_storage::access> & acc, bool write, const ggml_tensor * t_,
        bool trans, const char * what) {
    const layout l(t_, trans);

    for_each_element(t_, [&](size_t byte) { t.add(acc, write, l, byte, what); });
}

// the cell whose stored values match this row of markers; the position is derived from the base tensor's dims
struct place {
    uint32_t strm = 0;
    uint32_t cell = 0;
    int      hits = 0;
};

place locate(const ggml_tensor * base, bool trans, const std::vector<float> & row) {
    const layout l(base, trans);
    place        res;

    for (uint32_t s = 0; s < l.n_stream; ++s) {
        for (uint32_t c = 0; c < l.kv_size; ++c) {
            bool same = true;

            for (size_t m = 0; m < row.size() && same; ++m) {
                float got = 0.0f;
                std::memcpy(&got, (const char *) base->data + l.byte_of(s, c, (uint32_t) m), sizeof(float));
                same = got == row[m];
            }

            if (same) {
                ++res.hits;
                res.strm = s;
                res.cell = c;
            }
        }
    }

    return res;
}

struct execution {
    llama_kv_cache &       kv;
    const llama_hparams &  hp;
    ggml_backend_t         cpu;
    const bool             v_trans;
    const uint32_t         n_stream;
    std::vector<uint32_t>  layers;
    int                    gt = 0;   // ubatch token id, counted across the whole config

    execution(llama_kv_cache & kv, const llama_hparams & hp, ggml_backend_t cpu, bool v_trans, uint32_t n_stream) :
        kv(kv), hp(hp), cpu(cpu), v_trans(v_trans), n_stream(n_stream) {
        layers = kv.get_layer_ids();
        require(layers.size() == N_LAYER, "the cache does not hold one entry per model layer");
        require(hp.is_n_embd_v_gqa_variable(), "the model shell did not make the V width vary per layer");
    }

    // the rows an idxs input addresses, mapped through the destination it writes
    void check_idxs(tally & t, const std::vector<llama_kv_storage::access> & acc, const ggml_tensor * wr,
            const ggml_tensor * idxs, bool trans, const char * what) {
        require(wr->op == GGML_OP_SET_ROWS, "cpy_k / cpy_v did not produce a set_rows op");
        require(wr->src[1] == idxs, "the set_rows index input is not the idxs tensor that was filled in");
        require(ggml_backend_buffer_is_host(idxs->buffer), "the idxs tensor is expected to be host memory");
        require(ggml_element_size(idxs) == sizeof(int64_t), "the idxs are not int64");

        // src[0] is the source rows, src[1] the indices and src[2] the destination (llama.cpp #16063)
        const ggml_tensor * dst = wr->src[2];
        const layout        l(dst, trans);

        const int64_t * vals = (const int64_t *) idxs->data;
        const int64_t   n    = ggml_nelements(idxs);

        for (int64_t e = 0; e < n; ++e) {
            const int64_t row = vals[e];

            require(row >= 0 && row < dst->ne[1], "an idxs value is outside the rows of the destination");

            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                t.add(acc, true, l, (size_t) dst->view_offs + (size_t) row*dst->nb[1] + (size_t) i0*dst->nb[0], what);
            }
        }
    }

    // place the ubatch, run the KV writes, and check every byte this execution reads or writes against the list
    void step(const llama_ubatch & ub, const char * stage, bool expect_gaps) {
        const auto sinfo = kv.find_slot(ub, false);
        require(!sinfo.empty(), "find_slot found no slot");

        kv.apply_ubatch(sinfo, ub);

        const uint32_t n_kv = kv.get_n_kv(sinfo);
        const auto     acc  = kv.get_access(sinfo, n_kv);

        kv.prepare_access(sinfo, n_kv);   // the cache hands the same list to the storage (E3 would map it here)

        const auto & idxs0 = sinfo.idxs;

        require(n_kv >= NPAD && n_kv <= CAP, "n_kv is not a padded part of the capacity");

        bool gapped = false;
        for (const auto & cells : idxs0) {
            for (size_t i = 1; i < cells.size(); ++i) {
                if (cells[i] != cells[i - 1] + 1) {
                    gapped = true;
                }
            }
        }
        require(gapped == expect_gaps, std::string(stage) + ": the placement is "
                + (gapped ? "not contiguous" : "contiguous") + " but should not be");

        const uint32_t n_tokens = ub.n_tokens;
        const int      gtbase   = gt;
        gt += (int) n_tokens;

        const ggml_init_params params{ 2*1024*1024, nullptr, true };
        context_ptr ctx(ggml_init(params), ggml_free);
        require(bool(ctx), "ggml_init failed");

        auto * gf = ggml_new_graph_custom(ctx.get(), 1024, false);

        std::vector<ggml_tensor *> k_cur(layers.size()), v_cur(layers.size()), k_idx(layers.size()), v_idx(layers.size());
        std::vector<ggml_tensor *> k_view(layers.size()), v_view(layers.size());
        std::vector<ggml_tensor *> set_k(layers.size()), set_v(layers.size());

        for (size_t i = 0; i < layers.size(); ++i) {
            const uint32_t il = layers[i];

            k_cur[i] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.n_embd_head_k(il), hp.n_head_kv(il), n_tokens);
            v_cur[i] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hp.n_embd_head_v(il), hp.n_head_kv(il), n_tokens);
            k_idx[i] = kv.build_input_k_idxs(ctx.get(), ub);
            v_idx[i] = kv.build_input_v_idxs(ctx.get(), ub);

            // made before the allocation, so that ggml-alloc wires them onto the K/V buffer
            k_view[i] = kv.get_k(ctx.get(), (int32_t) il, n_kv, sinfo);
            v_view[i] = kv.get_v(ctx.get(), (int32_t) il, n_kv, sinfo);
        }

        // the writes: same ops the attention block runs (cpy_k / cpy_v -> storage.set_rows_k / set_rows_v)
        for (size_t i = 0; i < layers.size(); ++i) {
            const uint32_t il = layers[i];

            set_k[i] = kv.cpy_k(ctx.get(), k_cur[i], k_idx[i], (int32_t) il, sinfo);
            set_v[i] = kv.cpy_v(ctx.get(), v_cur[i], v_idx[i], (int32_t) il, sinfo);

            ggml_build_forward_expand(gf, set_k[i]);
            ggml_build_forward_expand(gf, set_v[i]);
        }

        // allocate only once the graph is complete: ggml-alloc has to see the intermediate tensors that
        // cpy_v creates for a narrow layer (ggml_pad), exactly like in a real graph build
        buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu), ggml_backend_buffer_free);
        require(bool(buf), "allocating the graph tensors on the CPU backend failed");

        for (size_t i = 0; i < layers.size(); ++i) {
            const uint32_t il   = layers[i];
            const size_t   n_kg = hp.n_embd_k_gqa(il);
            const size_t   n_vg = hp.n_embd_v_gqa(il);

            require(ggml_nbytes(k_cur[i]) == n_tokens*n_kg*sizeof(float), "unexpected K input layout");

            auto * kd = (float *) k_cur[i]->data;
            auto * vd = (float *) v_cur[i]->data;

            for (uint32_t t = 0; t < n_tokens; ++t) {
                for (size_t m = 0; m < n_kg; ++m) {
                    kd[t*n_kg + m] = marker(1000 + (int) il, gtbase + (int) t, (int) m);
                }
                for (size_t m = 0; m < n_vg; ++m) {
                    vd[t*n_vg + m] = marker(2000 + (int) il, gtbase + (int) t, (int) m);
                }
            }

            kv.set_input_k_idxs(k_idx[i], &ub, sinfo);
            kv.set_input_v_idxs(v_idx[i], &ub, sinfo);
        }

        // poison the embedding padding columns of the narrow transposed-V layer: storing a wider row has to
        // overwrite them, in the cells this ubatch writes
        const ggml_tensor * v_pad   = nullptr;
        uint32_t            v_pad_il = 0;
        std::vector<size_t> v_pad_bytes;

        if (v_trans) {
            const uint32_t wide = hp.n_embd_v_gqa_max();

            for (size_t i = 0; i < layers.size(); ++i) {
                const uint32_t il = layers[i];
                if (hp.n_embd_v_gqa(il) >= wide) {
                    continue;
                }

                const ggml_tensor * vb = v_view[i]->view_src;
                const layout        l(vb, true);

                v_pad_il = il;
                v_pad    = vb;

                for (size_t s = 0; s < idxs0.size(); ++s) {
                    for (const uint32_t cell : idxs0[s]) {
                        for (uint32_t j = hp.n_embd_v_gqa(il); j < wide; ++j) {
                            v_pad_bytes.push_back(l.byte_of((uint32_t) sinfo.strm[s], cell, j));
                        }
                    }
                }

                break;
            }

            require(v_pad && !v_pad_bytes.empty(), "no narrow transposed-V layer to check the padding on");

            for (const size_t byte : v_pad_bytes) {
                const float poison = -12345.0f;
                std::memcpy((char *) v_pad->data + byte, &poison, sizeof(float));
            }
        }

        require(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "CPU graph compute failed");

        // 1. reads: the views the attention graph runs on
        tally reads(n_stream);
        for (size_t i = 0; i < layers.size(); ++i) {
            const uint32_t    il = layers[i];
            const ggml_tensor * kb = k_view[i]->view_src;
            const ggml_tensor * vb = v_view[i]->view_src;

            require(kb && vb, "a graph view is not based on a K/V tensor");
            require((const char *) k_view[i]->data == (const char *) kb->data + k_view[i]->view_offs,
                    "the K view is not wired onto the K buffer");
            require((const char *) v_view[i]->data == (const char *) vb->data + v_view[i]->view_offs,
                    "the V view is not wired onto the V buffer");

            // the strides have to come from V's own dims, not from K's: DK != DV and n_head_kv varies per layer,
            // so a K-derived V stride would not match these numbers
            require(k_view[i]->nb[2] == ggml_row_size(kb->type, hp.n_embd_k_gqa(il)),
                    "the K view rows are not one row per cell of K");
            if (!v_trans) {
                require(v_view[i]->nb[2] == ggml_row_size(vb->type, vb->ne[0]),
                        "the V view rows are not one row per cell of V");
                require(v_view[i]->nb[1] == ggml_row_size(vb->type, hp.n_embd_head_v(il)),
                        "the V view strides are not built from n_embd_head_v");
            } else {
                require(v_view[i]->nb[2] == ggml_row_size(vb->type, vb->ne[1]),
                        "the transposed V view does not step cells by kv_size elements");
                require(v_view[i]->nb[1] == ggml_row_size(vb->type, vb->ne[1]*hp.n_embd_head_v(il)),
                        "the transposed V view strides are not built from n_embd_head_v");
            }

            check_bytes(reads, acc, false, k_view[i], false, "get_k");
            check_bytes(reads, acc, false, v_view[i], v_trans, "get_v");
        }

        require(reads.miss.empty(), "the graph reads cells it did not declare: " + reads.describe());
        require(reads.cells == (size_t) n_kv*n_stream,
                "the graph views do not span the padded window of every stream (" + reads.describe() + ")");

        // 2. writes: the rows the idxs inputs address
        tally writes(n_stream);
        for (size_t i = 0; i < layers.size(); ++i) {
            check_idxs(writes, acc, set_k[i], k_idx[i], false, "set_rows_k");
            check_idxs(writes, acc, set_v[i], v_idx[i], v_trans, "set_rows_v");
        }

        require(writes.miss.empty(), "the graph writes cells it did not declare: " + writes.describe());

        size_t slot_cells = 0;
        for (const auto & cells : idxs0) {
            std::vector<uint32_t> sorted = cells;
            std::sort(sorted.begin(), sorted.end());
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
            slot_cells += sorted.size();
        }
        require(writes.cells == slot_cells, "the idxs do not name exactly the cells of this slot ("
                + writes.describe() + ")");

        // 3. writes: where the marker rows actually landed
        for (size_t i = 0; i < layers.size(); ++i) {
            const uint32_t    il   = layers[i];
            const size_t      n_kg = hp.n_embd_k_gqa(il);
            const size_t      n_vg = hp.n_embd_v_gqa(il);
            const ggml_tensor * kb = k_view[i]->view_src;
            const ggml_tensor * vb = v_view[i]->view_src;

            for (size_t s = 0; s < idxs0.size(); ++s) {
                const uint32_t strm = (uint32_t) sinfo.strm[s];

                for (size_t ii = 0; ii < idxs0[s].size(); ++ii) {
                    // the ubatch walks its streams one after another: stream s, token ii
                    const int   gtid = gtbase + (int) (s*idxs0[s].size() + ii);
                    const auto  where = std::string(stage) + ": layer " + std::to_string(il) + ", stream "
                        + std::to_string(strm) + ", token " + std::to_string(ii);

                    std::vector<float> krow(n_kg), vrow(n_vg);
                    for (size_t m = 0; m < n_kg; ++m) {
                        krow[m] = marker(1000 + (int) il, gtid, (int) m);
                    }
                    for (size_t m = 0; m < n_vg; ++m) {
                        vrow[m] = marker(2000 + (int) il, gtid, (int) m);
                    }

                    const place kp = locate(kb, false, krow);
                    const place vp = locate(vb, v_trans, vrow);

                    require(kp.hits == 1, "the K row of a token matched " + std::to_string(kp.hits) + " cells: " + where);
                    require(vp.hits == 1, "the V row of a token matched " + std::to_string(vp.hits) + " cells: " + where);
                    require(kp.strm == strm && vp.strm == strm, "a stored row landed in another stream: " + where);
                    require(kp.cell == idxs0[s][ii] && vp.cell == idxs0[s][ii],
                            "a stored row landed in another cell than the slot: " + where);
                    require(covered(acc, true, kp.strm, kp.cell), "the K row landed in an undeclared cell: " + where);
                    require(covered(acc, true, vp.strm, vp.cell), "the V row landed in an undeclared cell: " + where);
                }
            }
        }

        // 4. transposed V: the padding columns of the written cells were written as well
        if (v_trans) {
            size_t overwritten = 0;

            for (const size_t byte : v_pad_bytes) {
                float got = 0.0f;
                std::memcpy(&got, (const char *) v_pad->data + byte, sizeof(float));
                if (got != -12345.0f) {
                    ++overwritten;
                }
            }

            require(overwritten == v_pad_bytes.size(), "a transposed V write skipped the embedding padding of layer "
                    + std::to_string(v_pad_il) + " (" + std::to_string(overwritten) + "/"
                    + std::to_string(v_pad_bytes.size()) + " columns left untouched)");

            std::printf("      V padding of layer %u: %zu columns rewritten\n", v_pad_il, v_pad_bytes.size());
        }

        std::printf("    %-10s n_kv=%3u  reads %s  writes %s\n", stage, n_kv, reads.describe().c_str(),
                writes.describe().c_str());

        // negative controls: a list that leaves a cell out has to be caught
        auto less_read = acc;
        bool found = false;
        for (auto & a : less_read) {
            if (!a.write && a.cells.second == n_kv) {
                --a.cells.second;
                found = true;
                break;
            }
        }
        require(found, "no read range ended at n_kv");

        tally neg_read(n_stream);
        for (size_t i = 0; i < layers.size(); ++i) {
            check_bytes(neg_read, less_read, false, k_view[i], false, "get_k");
        }
        require(!neg_read.miss.empty(), "dropping the last read cell went unnoticed");

        auto less_write = acc;
        found = false;
        for (auto it = less_write.rbegin(); it != less_write.rend(); ++it) {
            if (it->write) {
                --it->cells.second;
                found = true;
                break;
            }
        }
        require(found, "no write range to shrink");

        tally neg_write(n_stream);
        for (size_t i = 0; i < layers.size(); ++i) {
            check_idxs(neg_write, less_write, set_k[i], k_idx[i], false, "set_rows_k");
            check_idxs(neg_write, less_write, set_v[i], v_idx[i], v_trans, "set_rows_v");
        }
        require(!neg_write.miss.empty(), "dropping the last written cell went unnoticed");

        std::printf("    %-10s negative controls: %s / %s\n", "control", neg_read.miss.c_str(), neg_write.miss.c_str());
    }

    // the K-shift graph reads and rewrites every cell of every stream; update() hands the list to the storage
    // before building it, and build_graph_shift runs view_k_rope over each layer that has rope
    void shift() {
        const auto acc = kv.get_access_shift();

        kv.prepare_access_shift();

        tally touched(n_stream);

        context_ptr ctx(ggml_init({ 1 * 1024 * 1024, nullptr, true }), ggml_free);
        require(bool(ctx), "ggml_init failed");

        uint32_t ropes = 0;

        for (const uint32_t il : layers) {
            if (!hp.has_rope(il)) {
                continue;
            }

            ++ropes;

            const uint32_t n_rot = hp.n_rot(il);
            const uint32_t nope  = hp.n_lora_kv > 0 ? hp.n_embd_head_k(il) - n_rot : 0;

            require(n_rot > 0 && n_rot <= hp.n_embd_head_k(il), "the rotated part of K is empty or oversized");

            // the region build_graph_shift asks the storage for, rebuilt here from the K tensor and the hparams
            ggml_tensor * k = kv.get_k_storage((int32_t) il);
            ggml_tensor * v = ggml_view_3d(ctx.get(), k,
                    n_rot, hp.n_head_kv(il), (int64_t) CAP*kv.get_n_stream(),
                    ggml_row_size(k->type, hp.n_embd_head_k(il)),
                    ggml_row_size(k->type, hp.n_embd_k_gqa(il)),
                    ggml_row_size(k->type, nope));

            check_bytes(touched, acc, true, v, false, "view_k_rope");
        }

        require(ropes == N_LAYER, "the model shell gave the layers no rope pattern");
        require(touched.miss.empty(), "the K-shift touches cells it did not declare: " + touched.describe());
        require(touched.cells == (size_t) CAP*kv.get_n_stream(),
                "the K-shift does not span the whole capacity of every stream (" + touched.describe() + ")");

        auto less = acc;
        bool found = false;
        for (auto it = less.rbegin(); it != less.rend(); ++it) {
            if (it->write && it->cells.second == CAP) {
                --it->cells.second;
                found = true;
                break;
            }
        }
        require(found, "the shift list does not cover the whole capacity of the last stream");

        const uint32_t last = layers.back();
        ggml_tensor *  k    = kv.get_k_storage((int32_t) last);
        ggml_tensor *  v    = ggml_view_3d(ctx.get(), k,
                hp.n_rot(last), hp.n_head_kv(last), (int64_t) CAP*kv.get_n_stream(),
                ggml_row_size(k->type, hp.n_embd_head_k(last)),
                ggml_row_size(k->type, hp.n_embd_k_gqa(last)),
                ggml_row_size(k->type, 0));

        tally neg(n_stream);
        check_bytes(neg, less, true, v, false, "view_k_rope");
        require(!neg.miss.empty(), "the shift list can miss the last cell of a stream without being caught");

        std::printf("    %-10s shift %s\n", "K-shift", touched.describe().c_str());
    }
};

void run_config(ggml_backend_t cpu, const llama_model & model, bool v_trans, uint32_t n_stream) {
    const std::string tag = v_trans ? (n_stream > 1 ? "vt2_" : "vt1_") : (n_stream > 1 ? "fv2_" : "fv1_");

    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            v_trans, false, false, CAP, n_stream, NPAD, 0, LLAMA_SWA_TYPE_NONE, nullptr, {}, {}, {}, tag.c_str());

    require(kv.get_size() == CAP, "the cache did not keep the requested capacity");
    require(kv.get_n_stream() == n_stream, "the cache did not keep the requested stream count");

    execution ex(kv, model.hparams, cpu, v_trans, n_stream);

    std::printf("config %s: kv_size=%u n_pad=%u v_trans=%d n_stream=%u  K %u x { %u, %u }  V %u x { %u, %u }"
            "  padded V to %u\n",
            tag.c_str(), kv.get_size(), NPAD, int(v_trans), n_stream, DK, HKV[0], HKV[1], DV, HKV[0], HKV[1],
            model.hparams.n_embd_v_gqa_max());

    {
        const batch ub(n_stream, T1, 0);
        ex.step(ub.ub, "window", false);
    }

    // free every other cell of the window: the head moved past them, so the next small ubatch starts over at cell 0
    // and is placed in scattered cells (seq_rm takes a half-open position range)
    for (uint32_t s = 0; s < n_stream; ++s) {
        for (uint32_t i = 1; i < T1; i += 2) {
            require(kv.seq_rm((llama_seq_id) s, (llama_pos) i, (llama_pos) (i + 1)), "seq_rm failed");
        }
    }

    size_t kept = 0;
    for (uint32_t i = 0; i < CAP; ++i) {
        if (!kv.get_cells(0).is_empty(i)) {
            ++kept;
        }
    }
    require(kept == T1/2, "the rollback did not leave the expected holes");

    {
        const batch ub(n_stream, T2, (llama_pos) T1);
        ex.step(ub.ub, "holes", true);
    }

    ex.shift();
}

} // namespace

int main() {
    try {
        backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(cpu), "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);

        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        ggml_backend_dev_t cpu_only[] = { ggml_backend_get_device(cpu.get()), nullptr };
        mp.devices = cpu_only;

        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, mp));
        require(bool(model), "failed to create the tensor-free Qwen35 model shell");

        model->hparams = llama_hparams{};
        auto & hp = model->hparams;

        hp.n_ctx_train        = CAP;
        hp.n_embd             = HQ*DK;
        hp.n_layer_all        = N_LAYER;
        hp.n_layer_nextn      = 0;
        hp.n_embd_head_k_full = DK;
        hp.n_embd_head_v_full = DV;
        hp.n_rot_full         = ROT;

        for (uint32_t il = 0; il < N_LAYER; ++il) {
            hp.n_head_arr[il]    = HQ;
            hp.n_head_kv_arr[il] = HKV[il];
            hp.rope_pattern[il]  = 1;   // build_graph_shift only shifts layers with rope_pattern != 0
        }

        require(hp.is_n_embd_k_gqa_variable(), "the model shell did not make the K width vary per layer");
        require(hp.n_embd_v_gqa_max() == HKV[N_LAYER - 1]*DV, "the model shell did not pad V to the widest layer");
        require(hp.n_swa == 0 && hp.swa_type == LLAMA_SWA_TYPE_NONE, "unexpected SWA defaults");

        run_config(cpu.get(), *model, false, 1);
        run_config(cpu.get(), *model, true,  1);
        run_config(cpu.get(), *model, false, 2);
        run_config(cpu.get(), *model, true,  2);

        require(hp.n_swa == 0 && hp.swa_type == LLAMA_SWA_TYPE_NONE, "a cache modified the model SWA settings");

        std::printf("PASS test-kv-storage-contract: every byte this execution reads or writes is inside a"
                " declared range\n");
        return EXIT_SUCCESS;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL test-kv-storage-contract: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
