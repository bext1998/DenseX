#include "llama-kv-storage.h"

#include "llama-densex-env.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-io.h"

#include <algorithm>

llama_kv_storage::llama_kv_storage(const llama_hparams & hparams, bool v_trans, uint32_t kv_size) :
    hparams(hparams), v_trans(v_trans), kv_size(kv_size) {
}

void llama_kv_storage::add_buffer(ggml_context_ptr ctx, ggml_backend_buffer_t buf) {
    ggml_backend_buffer_clear(buf, 0);
    ctxs_bufs.emplace_back(std::move(ctx), buf);

    // DenseX (issue #8): note whether this buffer maps its pages on demand - see lazy_attach()
    lazy_attach(buf);
}

void llama_kv_storage::clear() {
    for (auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_storage::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

void llama_kv_storage::copy_stream(uint32_t ssrc, uint32_t sdst) const {
    for (const auto & layer : layers) {
        ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

        if (layer.v_stream[ssrc]) {
            ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
        }
    }
}

size_t llama_kv_storage::k_row_size(uint32_t ikv) const {
    const auto & layer = layers[ikv];
    return ggml_row_size(layer.k->type, hparams.n_embd_k_gqa(layer.il));
}

size_t llama_kv_storage::v_row_size(uint32_t ikv) const {
    GGML_ASSERT(!v_trans);
    const auto & layer = layers[ikv];
    return ggml_row_size(layer.v->type, hparams.n_embd_v_gqa(layer.il));
}

size_t llama_kv_storage::v_el_size(uint32_t ikv) const {
    GGML_ASSERT(v_trans);
    return ggml_type_size(layers[ikv].v->type);
}

// K and non-transposed V: one row per cell, a range of cells is one contiguous block.
// Transposed V: one row per embedding element, a range of cells is a block in every row.

void llama_kv_storage::write_k(llama_io_write_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * k = layers[ikv].k_stream[strm];
    const size_t row = k_row_size(ikv);
    for (const auto & r : ranges) {
        io.write_tensor(k, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
    }
}

void llama_kv_storage::write_v(llama_io_write_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * v = layers[ikv].v_stream[strm];
    if (!v_trans) {
        const size_t row = v_row_size(ikv);
        for (const auto & r : ranges) {
            io.write_tensor(v, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
        }
    } else {
        const size_t   el     = v_el_size(ikv);
        const uint32_t n_embd = hparams.n_embd_v_gqa(layers[ikv].il);
        for (uint32_t j = 0; j < n_embd; ++j) {
            for (const auto & r : ranges) {
                io.write_tensor(v, ((size_t) r.first + (size_t) j * kv_size) * el, (size_t) (r.second - r.first) * el);
            }
        }
    }
}

void llama_kv_storage::read_k(llama_io_read_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * k = layers[ikv].k_stream[strm];
    const size_t row = k_row_size(ikv);
    for (const auto & r : ranges) {
        io.read_tensor(k, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
    }
}

void llama_kv_storage::read_v(llama_io_read_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * v = layers[ikv].v_stream[strm];
    if (!v_trans) {
        const size_t row = v_row_size(ikv);
        for (const auto & r : ranges) {
            io.read_tensor(v, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
        }
    } else {
        const size_t   el     = v_el_size(ikv);
        const uint32_t n_embd = hparams.n_embd_v_gqa(layers[ikv].il);
        for (uint32_t j = 0; j < n_embd; ++j) {
            for (const auto & r : ranges) {
                io.read_tensor(v, ((size_t) r.first + (size_t) j * kv_size) * el, (size_t) (r.second - r.first) * el);
            }
        }
    }
}

void llama_kv_storage::clear_k(uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * k = layers[ikv].k_stream[strm];
    const size_t row = k_row_size(ikv);
    for (const auto & r : ranges) {
        llama_clear_tensor_data(k, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
    }
}

void llama_kv_storage::clear_v(uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const {
    ggml_tensor * v = layers[ikv].v_stream[strm];
    if (!v_trans) {
        const size_t row = v_row_size(ikv);
        for (const auto & r : ranges) {
            llama_clear_tensor_data(v, (size_t) r.first * row, (size_t) (r.second - r.first) * row);
        }
    } else {
        const size_t   el     = v_el_size(ikv);
        const uint32_t n_embd = hparams.n_embd_v_gqa(layers[ikv].il);
        for (uint32_t j = 0; j < n_embd; ++j) {
            for (const auto & r : ranges) {
                llama_clear_tensor_data(v, ((size_t) r.first + (size_t) j * kv_size) * el, (size_t) (r.second - r.first) * el);
            }
        }
    }
}

// graph views and writes (issue #5): the strides and offsets are the ones llama_kv_cache used, unchanged

ggml_tensor * llama_kv_storage::view_k(ggml_context * ctx, uint32_t ikv, uint32_t n_kv, uint32_t s0, uint32_t ns) const {
    const auto & layer = layers[ikv];
    const uint32_t il = layer.il;

    auto * k = layer.k;

    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*s0);
}

ggml_tensor * llama_kv_storage::view_v(ggml_context * ctx, uint32_t ikv, uint32_t n_kv, uint32_t s0, uint32_t ns) const {
    const auto & layer = layers[ikv];
    const uint32_t il = layer.il;

    auto * v = layer.v;

    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),      // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                            // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),               // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*s0);
}

ggml_tensor * llama_kv_storage::set_rows_k(ggml_context * ctx, uint32_t ikv, ggml_tensor * k_cur, ggml_tensor * k_idxs) const {
    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, (int64_t) kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_storage::set_rows_v(ggml_context * ctx, uint32_t ikv, ggml_tensor * v_cur, ggml_tensor * v_idxs) const {
    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, (int64_t) kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_storage::view_k_rope(ggml_context * ctx, uint32_t ikv, uint32_t n_rot, uint32_t n_embd_nope) const {
    const auto & layer = layers[ikv];
    const uint32_t il = layer.il;

    const int64_t n_stream = layer.k->ne[2];

    return ggml_view_3d(ctx, layer.k,
            n_rot, hparams.n_head_kv(il), (int64_t) kv_size*n_stream,
            ggml_row_size(layer.k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(layer.k->type, hparams.n_embd_k_gqa(il)),
            ggml_row_size(layer.k->type, n_embd_nope));
}

// DenseX (issue #8): DENSEX_KV_LAZY=1 puts the offloaded KV cache on a CUDA buffer that keeps the address
// space of the whole cache but holds no physical memory yet - the driver hands out its 2 MiB pages as the
// KV uses them, and the backend tracks which pages it has (see ggml_backend_cuda_vmm_map and
// vmm_ensure_mapped() in ggml/src/ggml-cuda/ggml-cuda.cu).
//
// The two halves only work together: the backend cannot tell which cells the graph of a ubatch writes, and
// this class cannot tell where a page boundary falls inside a row. So the backend exports the mapping
// through its registry, and prepare() - which is told what every execution touches - asks for the pages of
// those cells before the execution runs. Planning a graph never calls prepare(), so a reserve or a dummy
// graph maps nothing.
//
// K and a non-transposed V hold one row per cell, so cells [a, b) of a stream are one contiguous byte range
// of that stream's view, and one mapping call per tensor is enough. A transposed V holds one row per
// embedding element, so a range of cells reaches into every row: there the whole V is backed once and the
// switch saves nothing on it, which the log says once.

void llama_kv_storage::lazy_attach(ggml_backend_buffer_t buf) {
    const llama_densex_env & env = llama_densex_env_get();

    // DENSEX_KV_LAZY without DENSEX_KV_VMM leaves the KV cache on a plain cudaMalloc buffer; llama-densex-env
    // already warns about that combination on the log
    if (!env.kv_lazy || !env.kv_vmm) {
        return;
    }

    if (!lazy.resolved) {
        // asked once, even when the answer is no: a backend without the mapping never gets it later
        lazy.resolved = true;

        const ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
        const ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;

        if (reg == nullptr) {
            return;
        }

        lazy.map  = (vmm_map_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_map");
        lazy.info = (vmm_info_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_buffer_lazy_info");

        // a backend that does not offer the mapping (not CUDA, or built with GGML_CUDA_NO_VMM)
        if (lazy.map == nullptr || lazy.info == nullptr) {
            return;
        }
    }

    size_t reserved = 0;
    size_t backed   = 0;

    if (!lazy.info(buf, &reserved, &backed)) {
        return; // a buffer that is fully backed already (issue #7 maps its pages eagerly)
    }

    const bool first = !lazy.active;

    lazy.active = true;

    if (lazy.step == 0) {
        lazy.step = (uint32_t) std::min<int64_t>(env.kv_lazy_cells, kv_size);
    }

    if (first) {
        LLAMA_LOG_INFO("%s: DENSEX_KV_LAZY=1: buffer '%s' reserves %.2f MiB of device address space and backs "
                "nothing yet; the pages are taken in steps of %u cell(s) per stream "
                "(DENSEX_KV_LAZY_CELLS=%lld, valid values: 1 .. the KV size)\n", __func__,
                ggml_backend_buffer_name(buf), reserved / 1024.0 / 1024.0, lazy.step,
                (long long) env.kv_lazy_cells);
    } else {
        LLAMA_LOG_INFO("%s: DENSEX_KV_LAZY=1: buffer '%s': %.2f MiB reserved, %.2f MiB backed\n", __func__,
                ggml_backend_buffer_name(buf), reserved / 1024.0 / 1024.0, backed / 1024.0 / 1024.0);
    }
}

bool llama_kv_storage::lazy_mapped_mib(double * backed, double * reserved) const {
    *backed   = 0.0;
    *reserved = 0.0;

    if (lazy.info == nullptr) {
        return false;
    }

    for (const auto & [_, buf] : ctxs_bufs) {
        size_t b = 0;
        size_t r = 0;

        if (lazy.info(buf.get(), &r, &b)) {
            *backed   += b;
            *reserved += r;
        }
    }

    return true;
}

// back the pages of the cells an execution touches, in the K of every layer and - unless the access does
// not touch it, or the V is transposed - in the V too. K and V are tracked apart: a K-shift maps the whole K
// of a stream and leaves its V where it was, and the two then grow on their own. false when the driver could
// not hand out the pages (the reason is already on the log); a failed call leaves the buffer as it was
// because the backend rolls its own mapping back.
bool llama_kv_storage::lazy_map(uint32_t strm, uint32_t cells, bool k_only) const {
    if (!lazy.active || lazy.map == nullptr || layers.empty()) {
        return true;
    }

    const uint32_t n_stream = layers[0].k_stream.size();

    if (strm >= n_stream) {
        return false; // prepare() checks the accesses; keep this class's own table in bounds
    }

    if (lazy.mapped_k.size() < n_stream) {
        lazy.mapped_k.resize(n_stream, 0);
        lazy.mapped_v.resize(n_stream, 0);
    }

    // the k-pool of the idx memory variant reaches cells through graph operators this class is never told
    // about (src/llama-memory-hybrid-idx.cpp), so there is no per-ubatch mapping to rely on there
    const uint32_t want = hparams.indexer_kpool != 0 ? kv_size : std::min(cells, kv_size);

    // grow in whole steps of DENSEX_KV_LAZY_CELLS, so that a long run of ubatches does not ask the driver
    // for a page range on every token; the step also covers the padding the views of the graph read
    const uint32_t step = lazy.step > 0 ? lazy.step : 1;

    auto next = [&](uint32_t from) {
        return want <= from ? from : (uint32_t) std::min<int64_t>(kv_size,
                (int64_t) from + (int64_t) step*(((int64_t) want - from + step - 1)/step));
    };

    const uint32_t from_k = lazy.mapped_k[strm];
    const uint32_t to_k   = next(from_k);

    const uint32_t from_v = lazy.mapped_v[strm];
    const uint32_t to_v   = k_only ? from_v : (v_trans ? kv_size : next(from_v));

    if (to_k == from_k && to_v == from_v) {
        return true;
    }

    const int64_t t0    = ggml_time_us();
    size_t        asked = 0;
    size_t        backed = 0;
    bool          ok    = true;

    for (const auto & layer : layers) {
        if (to_k > from_k && layer.k_stream[strm] != nullptr) {
            // the cells of a stream view step by its row stride, which for K includes the padding of
            // the row and for a transposed V is the row of the widest layer (see add_layer)
            const size_t  stride = layer.k_stream[strm]->nb[1];
            const size_t  off    = (size_t) from_k*stride;
            const size_t  size   = (size_t) (to_k - from_k)*stride;
            size_t        newly  = 0;

            asked += size;

            if (!lazy.map(layer.k_stream[strm], off, size, &newly)) {
                ok = false;
                break;
            }

            backed += newly;
        }

        if (to_v > from_v && layer.v_stream[strm] != nullptr) {
            const size_t  stride = layer.v_stream[strm]->nb[1];
            const size_t  off    = (size_t) from_v*stride;
            const size_t  size   = (size_t) (to_v - from_v)*stride;
            size_t        newly  = 0;

            asked += size;

            if (!lazy.map(layer.v_stream[strm], off, size, &newly)) {
                ok = false;
                break;
            }

            backed += newly;
        }
    }

    if (!ok) {
        return false;
    }

    lazy.mapped_k[strm] = to_k;
    lazy.mapped_v[strm] = to_v;

    if (v_trans && from_v == 0 && to_v > 0 && !lazy.transposed_v_backed) {
        lazy.transposed_v_backed = true;

        LLAMA_LOG_WARN("%s: DENSEX_KV_LAZY=1: this cache stores V transposed, so any range of cells reaches "
                "every row of V - V is backed whole (%zu cell(s) of %zu layer(s)) and the switch saves "
                "nothing on it\n", __func__, to_v, layers.size());
    }

    double backed_total   = 0.0;
    double reserved_total = 0.0;

    lazy_mapped_mib(&backed_total, &reserved_total);

    LLAMA_LOG_INFO("%s: lazy KV: stream %u: K cells [%u, %u), V cells [%u, %u) of %zu layer(s) - %.2f MiB "
            "asked for, %.2f MiB newly backed (whole pages), %.2f / %.2f MiB of the KV backing in use, "
            "%.2f ms\n", __func__, strm, from_k, to_k, from_v, to_v, layers.size(),
            asked / 1024.0 / 1024.0, backed / 1024.0 / 1024.0,
            backed_total / 1024.0 / 1024.0, reserved_total / 1024.0 / 1024.0,
            (ggml_time_us() - t0) / 1000.0);

    return true;
}

void llama_kv_storage::prepare(const std::vector<access> & acc) const {
    if (layers.empty()) {
        return;
    }

    const uint32_t n_stream = layers[0].k_stream.size();

    for (const auto & a : acc) {
        GGML_ASSERT(a.strm < n_stream);
        GGML_ASSERT(a.cells.first < a.cells.second && a.cells.second <= kv_size);
    }

    if (!lazy.active) {
        return;
    }

    // the first map of a stream is an ordinary one: the cells this execution reads and writes, rounded up
    // to a whole step - so a stream that is never used costs nothing
    for (const auto & a : acc) {
        if (!lazy_map(a.strm, a.cells.second, a.k_only)) {
            GGML_ABORT("the lazily mapped KV cache could not back stream %u up to cell %u - the device is out"
                    " of memory (see the error above)", a.strm, a.cells.second);
        }
    }
}
