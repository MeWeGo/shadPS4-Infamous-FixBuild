// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_compute_queue.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Marks buffer memory a game thread is about to write as CPU modified ahead of it.
    void InvalidateBuffersAhead(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Notes memory written past its page protection, like fence values.
    void OnBackingWritten(VAddr addr, u64 size);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    void BindComputePipeline(const ComputePipeline& pipeline);

    /// Whether a compute dispatch may run on the async compute queue: ready
    /// pipeline, graphics ring, no images/samplers, push descriptors, no DMA
    /// and no special buffers outside stream-backed ones.
    bool IsAsyncComputeEligible(const ComputePipeline& pipeline, const Shader::Info& cs);
    /// Records a dispatch on the async compute queue instead of the primary
    /// command buffer. Batched: collected in one command buffer through the
    /// frame, submitted at its end or when a readback needs the results.
    void RouteComputeDispatch(const ComputePipeline* pipeline);
    /// Submits the open compute batch, noting its writes. Called from the
    /// submit callback (frame end) and from the readback path (game thread
    /// faulting on batch-written memory needs the batch submitted first).
    void SubmitComputeBatch();
    /// Clears bindings after a routed dispatch without touching the runtime
    /// barrier tracking (visibility comes from timeline waits instead).
    void ResetComputeBindings();
    /// Submits open graphics work (recorded draws/dispatches or uploads) so an
    /// async compute submission can wait for an already-submitted tick.
    void FlushGraphicsForCompute();
    /// Waits a compute submission needs: all submitted graphics work plus all
    /// submitted sparse binds, so it never touches unmapped or stale memory.
    std::vector<std::pair<vk::Semaphore, u64>> ComputeSubmitWaits();

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

        const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    /// Second queue for compute work. Batched: compute-ring dispatches are
    /// collected in one command buffer through the frame and submitted at the
    /// frame's submit, overlapping with the next frame's draws.
    ComputeQueue compute_queue;
    /// Command buffer of the open compute batch; null when none.
    vk::CommandBuffer compute_batch_cb{};
    /// Graphics tick the batch waits for when submitted: the last submitted one
    /// when the batch began, so it doesn't wait for work recorded after it.
    u64 compute_batch_graphics_tick{};
    /// Guest ranges the batch writes, noted for graphics consumers and the
    /// readback queue once the batch is submitted and its tick is known.
    struct ComputeBatchWrite {
        VAddr start;
        VAddr end;
    };
    boost::container::small_vector<ComputeBatchWrite, 64> compute_batch_writes;    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;

    /// Memoizes GetSampler (XXH3 + map + LRU touch per sampler per draw).
    /// Entries are only valid for the sampler generation they were filled in;
    /// any map change (insert or GC erase) bumps it and invalidates them all.
    struct SamplerMemo {
        AmdGpu::Sampler sharp{};
        bool is_depth{};
        u64 generation{};
        vk::Sampler handle{};
        bool valid{};
    };
    static constexpr size_t NumSamplerMemos = 8;
    std::array<SamplerMemo, NumSamplerMemos> sampler_memos{};
    size_t next_sampler_memo{};

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
    };
    /// Shader buffers, vertex buffers and the index buffer.
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS + MaxVertexBufferCount + 1>
        bound_buffers;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    /// The vertex input last set in the command buffer, valid unless the dynamic state of the
    /// scheduler says it is dirty.
    VertexInputs<vk::VertexInputAttributeDescription2EXT> last_vertex_attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> last_vertex_bindings;

    /// Everything the dynamic state of a draw is built from besides the registers covered by
    /// their version, as of the last draw that built it.
    struct DynamicStateInputs {
        u64 regs_version{};
        std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS> write_masks{};
        bool feedback_loop{};
        bool is_indexed{};

        bool operator==(const DynamicStateInputs&) const = default;
    };
    mutable DynamicStateInputs last_dynamic_inputs{};

    /// Everything the render targets of a draw are found from besides the registers covered by
    /// their version, as of the last draw that found them.
    struct RenderTargetInputs {
        u64 regs_version{};
        u64 image_generation{};
        std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> cb_extents{};
        u32 db_extent{};
        u32 mrt_mask{};

        bool operator==(const RenderTargetInputs&) const = default;
    };
    RenderTargetInputs last_render_target_inputs{};

    bool attachment_feedback_loop{};
    bool needs_barrier{};
    /// Whether the draw being made reads memory in ways no accesses are kept for.
    bool untracked_access{};
    /// Whether the last work recorded was a dispatch, to count switches between draws and
    /// dispatches.
    bool last_work_compute{};
};

} // namespace Vulkan
