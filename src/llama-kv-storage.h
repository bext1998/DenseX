#pragma once

#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

struct llama_hparams;
class llama_io_write_i;
class llama_io_read_i;

// DenseX (issue #4, R1a): the physical side of the attention KV cache.
//
// llama_kv_cache keeps the logical side - cell allocation, sequence sharing, rollback, the per-stream rings and the
// graph views. This class keeps what the cells are stored in: the K/V tensors of every layer and stream, the backend
// buffers that own them, and the byte layout of a cell inside a tensor. In this first slice the cell -> byte math of
// the session state data, the failed-restore clear and the stream copy is done here; the graph views and writes
// (get_k/get_v, cpy_k/cpy_v, the K-shift view) still compute their strides in llama_kv_cache (B5 moves them here).
//
// Contract (what each operation touches; first slice: the backing is always the full, originally allocated buffer):
//   - allocation:  the backend init_tensor runs before add_buffer() (CUDA may zero the alloc padding of quantized
//                  tensors there), then add_buffer() zero-fills each owned buffer, padding included, so it never
//                  holds NaNs. A lazily mapped backing must have the init writes mapped too.
//   - clear:       clear() zero-fills every owned buffer (layers shared from another cache are not owned and are
//                  left alone).
//   - state I/O:   write_k/v, read_k/v and clear_k/v work on cell ranges of one stream, in the session order and
//                  with its byte ranges. Offsets are size_t: the transposed-V offset used to be computed in 32 bits
//                  and wrapped past 4 GiB in a layer (never reached by the supported configs).
//   - stream copy: copy_stream() copies whole streams (all layers) - the cross-stream copy of seq_cp.
//   - graph reads: the FA and non-FA views read cells [0, n_kv) of the streams s0..s1 they span, padded, holes
//                  included: a free cell can be read even if the mask hides it. K-shift reads and rewrites the
//                  rotated part of K over the whole capacity. Reserve / dummy graphs are planned at full capacity but
//                  do not touch the data.
//   - graph writes: cpy_k/cpy_v (set_rows) write the destination cells of the ubatch, per stream; a transposed V
//                  also writes the embedding padding up to v->ne[0]. The cells are known only when the ubatch is
//                  placed, so a lazy backing must map them per execution, not when the graph is built or reused.
// With the full backing all of these are always valid; a lazily mapped backing (E3) must map what an execution
// touches before it runs, and never in planning.
class llama_kv_storage {
public:
    struct layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    // a range of cells of one stream, from inclusive, to exclusive
    using cell_range = std::pair<uint32_t, uint32_t>;

    llama_kv_storage(const llama_hparams & hparams, bool v_trans, uint32_t kv_size);

    // every layer of the cache, including layers shared from another cache (their tensors are not owned here)
    std::vector<layer> layers;

    // takes a buffer holding K/V tensors of this cache and zero-fills it
    void add_buffer(ggml_context_ptr ctx, ggml_backend_buffer_t buf);

    // zero-fill every owned buffer
    void clear();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // copy all layers of stream ssrc to stream sdst
    void copy_stream(uint32_t ssrc, uint32_t sdst) const;

    // session state data of layer ikv, stream strm, the given cells
    void write_k(llama_io_write_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;
    void write_v(llama_io_write_i & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;
    void read_k (llama_io_read_i  & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;
    void read_v (llama_io_read_i  & io, uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;

    // zero the data of the given cells (failed restore)
    void clear_k(uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;
    void clear_v(uint32_t ikv, uint32_t strm, const std::vector<cell_range> & ranges) const;

    size_t k_row_size(uint32_t ikv) const;
    size_t v_row_size(uint32_t ikv) const; // !v_trans only
    size_t v_el_size (uint32_t ikv) const; // v_trans only

private:
    const llama_hparams & hparams;

    const bool     v_trans;
    const uint32_t kv_size;

    // ggml contexts for the KV cache along with the allocated backend buffers
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
};
