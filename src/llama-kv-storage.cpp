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
