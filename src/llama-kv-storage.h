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
// llama_kv_cache keeps the logical side - cell allocation, sequence sharing, rollback, the per-stream rings. This
// class keeps what the cells are stored in: the K/V tensors of every layer and stream, the backend buffers that own
// them, the byte layout of a cell inside a tensor, and the views and writes the compute graph runs on them. Callers
// name layers, streams and cells; only this class turns them into byte ranges and strides.
//
// Contract (what each operation touches; with the full backing the buffer is the originally allocated
// one throughout):
//   - allocation:  ggml_backend_alloc_ctx_tensors() runs the backend's init_tensor hook for each tensor before this
//                  class sees the buffer - CUDA may zero the alloc padding of quantized tensors there - and only
//                  then does add_buffer() take that buffer and zero-fill it, padding included, so it never holds
//                  NaNs. A lazily mapped backing clears only the pages it already has, and leaves the alloc padding
//                  unmapped instead of writing zeros into it: an unmapped page reads as zero, which is what that
//                  zero-fill is for. Planning a graph - the reserve and dummy graphs - allocates no access and maps
//                  nothing.
//   - clear:       clear() zero-fills every owned buffer (layers shared from another cache are not owned and are
//                  left alone).
//   - state I/O:   write_k/v, read_k/v and clear_k/v work on cell ranges of one stream, in the session order and
//                  with its byte ranges. Offsets are size_t: the transposed-V offset used to be computed in 32 bits
//                  and wrapped past 4 GiB in a layer (never reached by the supported configs).
//   - stream copy: copy_stream() copies whole streams (all layers) - the cross-stream copy of seq_cp.
//   - graph reads: view_k and view_v hand out the FA and non-FA views. They read cells [0, n_kv) of the streams
//                  s0..s1 the ubatch spans, padded, holes included: a free cell can be read even if the mask hides
//                  it. view_k_rope is the K-shift view: it reads and rewrites the rotated part of K over the whole
//                  capacity. Reserve / dummy graphs are planned at full capacity but do not touch the data.
//   - graph writes: set_rows_k and set_rows_v store the rows of the ubatch in the cells named by the idxs filled
//                  with k_row / v_row, per stream; a transposed V also writes the embedding padding up to v->ne[0].
//                  The cells are known only when the ubatch is placed, so a lazy backing must map them per
//                  execution, not when the graph is built or reused.
//   - accesses:    prepare() is told what each execution touches, as cells of every stream in the same order for
//                  every layer of the cache: llama_kv_cache::get_access for a ubatch (cells [0, n_kv) of each stream
//                  the ubatch spans are read, the cells of sinfo.idxs are written), and get_access_shift for the
//                  K-shift (every cell of every stream, read and rewritten). llama_kv_cache_context::apply() calls
//                  it for every ubatch, including when the graph is reused, and update() calls it before running the
//                  shift graph; the planning path never calls it. With the full backing prepare() only checks the
//                  bounds; a lazily mapped backing (issue #8) maps the ranges here, and skips the V tensors of a
//                  K-shift because that operation only touches K.
//   - not owned:   the MTP draft cache is a separate llama_kv_cache, so it has its own storage instance; the layers
//                  it shares with the target cache (the share callback) are pushed into its layers without being
//                  owned by it, and their buffers stay with the target cache.
//   - outside this contract for now (raw tensors, other architectures - the model DenseX trains on never reaches
//     them): src/llama-kv-cache-dsv4.cpp does host-side state write, read and clear of the whole K of one stream
//     through get_k_storage, and src/llama-memory-hybrid-idx.cpp has a k-pool that views full K across streams. A
//     lazily mapped backing handles the first through the backend's own mapping hooks, which map what a state read
//     or write touches (correct, but it backs the whole K); the k-pool reaches cells through graph operators that
//     this class is never told about, so with hparams.indexer_kpool set the whole K is backed up front instead of
//     growing per ubatch.
// With the full backing all of these are always valid; a lazily mapped backing (issue #8) maps what an execution
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

    // graph reads: cells [0, n_kv) of the streams [s0, s0 + ns) of layer ikv, padded, holes included
    ggml_tensor * view_k(ggml_context * ctx, uint32_t ikv, uint32_t n_kv, uint32_t s0, uint32_t ns) const;
    ggml_tensor * view_v(ggml_context * ctx, uint32_t ikv, uint32_t n_kv, uint32_t s0, uint32_t ns) const;

    // graph writes: store the rows of k_cur / v_cur in the rows that k_idxs / v_idxs name (filled with k_row / v_row)
    ggml_tensor * set_rows_k(ggml_context * ctx, uint32_t ikv, ggml_tensor * k_cur, ggml_tensor * k_idxs) const;
    ggml_tensor * set_rows_v(ggml_context * ctx, uint32_t ikv, ggml_tensor * v_cur, ggml_tensor * v_idxs) const;

    // the set_rows row of cell `cell` of stream strm; a transposed V has one row per embedding element j, and writes
    // every j < n_embd_v_gqa_max (the embedding padding of a layer with a smaller V included)
    int64_t k_row(uint32_t strm, uint32_t cell) const {
        return (int64_t) strm*kv_size + cell;
    }
    int64_t v_row(uint32_t strm, uint32_t cell, uint32_t j, uint32_t n_embd_v_gqa_max) const {
        return v_trans ? (int64_t) strm*kv_size*n_embd_v_gqa_max + (int64_t) j*kv_size + cell : (int64_t) strm*kv_size + cell;
    }

    // K-shift: the rotated part [n_embd_nope, n_embd_nope + n_rot) of every head, every cell of every stream; read and
    // rewritten in place
    ggml_tensor * view_k_rope(ggml_context * ctx, uint32_t ikv, uint32_t n_rot, uint32_t n_embd_nope) const;

    // what one execution touches, the same cells in every layer of the cache
    struct access {
        uint32_t   strm;
        cell_range cells;
        bool       write;

        // true when the access only touches K, so a lazily mapped backing does not have to back V for it
        // (the K-shift rewrites the rotated part of K over the whole capacity; issue #8)
        bool k_only = false;
    };

    // called before every execution that touches the K/V data (each ubatch, also when the graph is reused, and the
    // K-shift), never while planning. With the full backing there is nothing to map: it checks that the accesses are
    // in bounds. A lazily mapped backing (issue #8) maps them here.
    void prepare(const std::vector<access> & acc) const;

    uint32_t get_kv_size() const { return kv_size; }

private:
    const llama_hparams & hparams;

    const bool     v_trans;
    const uint32_t kv_size;

    // ggml contexts for the KV cache along with the allocated backend buffers
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // DenseX (issue #8): the page-granular backing of DENSEX_KV_LAZY. The backend owns the pages of the buffer and
    // maps them; this class decides how far into each stream an execution goes, because that is what neither side
    // can know alone: the backend cannot tell which cells a graph of set_rows writes, and this class cannot tell
    // where a page boundary falls inside a row. See llama-kv-storage.cpp for the logging and the failure path.
    using vmm_map_fn  = bool (*)(ggml_tensor * tensor, size_t offset, size_t size, size_t * newly_mapped);
    using vmm_info_fn = bool (*)(ggml_backend_buffer_t buffer, size_t * va_size, size_t * mapped_size);

    struct lazy_backing {
        bool       resolved = false;  // the backend registry has been asked for the two hooks below
        vmm_map_fn  map      = nullptr;
        vmm_info_fn info     = nullptr;

        bool     active  = false;  // at least one owned buffer of this cache is lazily mapped
        uint32_t step    = 0;      // cells backed at a time, the first map and every expansion (issue #8)
        bool     transposed_v_backed = false; // a transposed V is backed a whole stream at a time (see lazy_map)

        // cells [0, mapped_k[strm]) backed in the K, and [0, mapped_v[strm]) backed in the V, of every layer;
        // kept apart because an access can touch K only, as the K-shift does
        std::vector<uint32_t> mapped_k;
        std::vector<uint32_t> mapped_v;
    };

    mutable lazy_backing lazy;

    void lazy_attach(ggml_backend_buffer_t buf);
    bool lazy_map(uint32_t strm, uint32_t cells, bool k_only) const;
    bool lazy_mapped_mib(double * backed, double * reserved) const;
};
