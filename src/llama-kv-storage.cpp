#include "llama-kv-storage.h"

#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-io.h"

llama_kv_storage::llama_kv_storage(const llama_hparams & hparams, bool v_trans, uint32_t kv_size) :
    hparams(hparams), v_trans(v_trans), kv_size(kv_size) {
}

void llama_kv_storage::add_buffer(ggml_context_ptr ctx, ggml_backend_buffer_t buf) {
    ggml_backend_buffer_clear(buf, 0);
    ctxs_bufs.emplace_back(std::move(ctx), buf);
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

void llama_kv_storage::prepare(const std::vector<access> & acc) const {
    if (layers.empty()) {
        return;
    }

    const uint32_t n_stream = layers[0].k_stream.size();

    for (const auto & a : acc) {
        GGML_ASSERT(a.strm < n_stream);
        GGML_ASSERT(a.cells.first < a.cells.second && a.cells.second <= kv_size);
    }
}
