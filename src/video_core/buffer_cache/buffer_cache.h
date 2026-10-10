// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class ComputeQueue;
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;
    /// Up to this much of memory the game keeps writing is copied for each use.
    static constexpr u64 REWRITE_STREAM_THRESHOLD = 512_KB;
    /// At most this much of it is copied a frame, the rest is uploaded.
    static constexpr u64 MaxRewriteCopyBytes = 32_MB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Waits for the upload worker to finish every job, finishing the ones it left.
    /// A submission calls this first: the staging the jobs copy into must be complete
    /// before the command buffer using it reaches the driver.
    void FlushUploadJobs();

    /// Gives the buffer cache a second Vulkan queue to run readback copies on,
    /// so a game thread waiting for its data waits only for the submission that
    /// wrote it (signalled on that queue's timeline), not for the whole graphics
    /// timeline up to the submission the copy was recorded in.
    void SetReadbackQueue(Vulkan::ComputeQueue* queue) noexcept;

    /// Notes the tick of the submission that last wrote to a buffer the GPU
    /// bound. The readback copy waits for this tick, not for the latest one.
    void NoteGpuWriteTick() noexcept;

    /// Notes the tick of the async compute batch that last wrote, so the
    /// readback copy waits on the compute timeline instead of the graphics one.
    void NoteComputeWriteTick(u64 tick) noexcept;

    /// Registers a callback that submits the open compute batch. Called before
    /// a readback that may cover memory the batch writes: its results aren't
    /// available to a copy until it is submitted. GPU thread.
    void SetComputeBatchFlusher(std::function<void()> flusher) noexcept {
        compute_batch_flusher = std::move(flusher);
    }

    /// Copies back GPU modified memory that game threads read back recently, before they read
    /// it again. Called when the game is signalled that GPU work is done.
    void PrefetchReadbacks();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Notes memory written straight to its backing, past the page protection, so the GPU copy
    /// of it is uploaded again before it is used.
    void OnBackingWritten(VAddr device_addr, u64 size);

    /// Notes memory a dispatch of one of the game's compute rings writes, to tell how many of the
    /// copies back game threads wait for are of what those wrote.
    void NoteComputeRingWrite(VAddr device_addr, u64 size) {
        compute_ring_writes[compute_ring_write_index++ % compute_ring_writes.size()] = {
            device_addr, device_addr + size};
    }

    /// Notes ranges an async compute submission wrote. Prunes completed ticks.
    void NoteComputeWrites(VAddr start, VAddr end, u64 tick);
    /// Returns and clears the compute tick graphics must wait for, 0 if none.
    u64 ConsumePendingComputeWait();
    /// Tells whether an async-compute timeline tick completed, for pruning.
    void SetComputeIsFree(std::function<bool(u64)> is_free) {
        compute_is_free = std::move(is_free);
    }
    /// Provides the async-compute fence (semaphore + last submitted tick) that
    /// sparse binds must wait for, so remaps don't yank memory out from under
    /// in-flight compute work. Empty tick means nothing to wait for.
    void SetComputeFence(std::function<std::pair<vk::Semaphore, u64>()> fence) {
        compute_fence = std::move(fence);
    }
    /// Fence covering all sparse binds submitted so far (semaphore + tick,
    /// tick 0 when none). Compute submissions wait for it before touching
    /// arena memory.
    std::pair<vk::Semaphore, u64> MemoryFence() const noexcept;

    /// Finds a buffer for the specified region. is_read_tracked tells that the caller reports
    /// its accesses to the runtime, which lets small reads use the cached copy in place.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false,
                                                             bool is_read_tracked = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    using DownloadCopies = boost::container::small_vector<vk::BufferCopy, 8>;

    /// A copy of GPU modified memory back to the game, recorded and submitted by the GPU thread
    /// and waited for by the game thread that touched the memory.
    struct Readback {
        Vulkan::StagingBufferRef staging;
        /// Source offsets are into the arena, destination offsets into the staging buffer.
        DownloadCopies copies;
        VAddr arena_base{};
        VAddr start{};
        VAddr end{};
        u64 tick{};
        /// True when the copy runs on the readback queue and the tick is on its
        /// timeline; false for the graphics queue's timeline.
        bool on_readback_queue{};
        /// Set by the GPU thread when it writes the memory again, so the copy is outdated.
        std::atomic<bool> stale{};
        std::atomic<bool> applied{};
        /// The copy won't be applied, and its ranges are GPU modified again.
        std::atomic<bool> recovered{};
        /// Made ahead of a game thread touching the memory.
        bool prefetched{};
        /// Whether it was counted for or against copying its window ahead. GPU thread.
        bool rated{};
        /// Written back by the thread writing back copies made ahead, rather than by a game
        /// thread waiting for it. Set before applied.
        bool applied_ahead{};
        std::mutex mutex;

        bool Done() const noexcept {
            return applied.load(std::memory_order_acquire) ||
                   recovered.load(std::memory_order_acquire);
        }
    };

    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    /// Returns device memory and an offset into it to back size bytes of arena blocks.
    std::pair<vk::DeviceMemory, u64> AllocateResidency(u64 size);

    /// Returns the arena and the window around a range that is read back with it.
    std::tuple<const Buffer*, VAddr, VAddr> GetReadbackWindow(VAddr device_addr, u64 size);

    /// Records and submits a copy back of the GPU modified memory around a range. GPU thread.
    std::shared_ptr<Readback> StartReadback(VAddr device_addr, u64 size);

    /// Waits for a copy back and writes it to the game's memory. Returns false if it can't be
    /// used as the GPU wrote the memory again. Any thread. Ahead is for the thread writing back
    /// copies made ahead, which waits for the GPU itself and gets false for copies written back
    /// already.
    bool FinishReadback(Readback& readback, bool ahead = false);

    /// Makes the memory of a copy back that won't be used GPU modified again. GPU thread.
    void RecoverReadback(Readback& readback);

    /// Finishes or recovers the copies back overlapping a range. GPU thread.
    void SettleReadbacks(VAddr start, VAddr end);

    /// Frees the staging memory of copies back that are done. GPU thread.
    void PruneReadbacks();

    /// Returns true if memory the game keeps writing should be copied for a draw: once a frame for
    /// each address, and within a budget.
    bool TakeRewriteCopy(VAddr device_addr, u64 size);

    /// Recovers the copies the GPU wrote over again and frees those that are done. GPU thread.
    void ApplyFinishedReadbacks();

    /// Drops async-compute write records whose ticks completed. GPU thread.
    void PruneComputeWrites();

    /// Writes back copies made ahead as soon as the GPU is done with them.
    void ReadbackThread(std::stop_token token);

    /// Counts a copy made ahead for or against copying its window ahead again. GPU thread.
    void RatePrefetch(Readback& readback, bool useful);

    /// Records a copy back of the GPU modified memory in a window, or returns null if there is
    /// none. GPU thread.
    std::shared_ptr<Readback> RecordReadback(const Buffer* arena, VAddr start, VAddr end,
                                             bool record_copy = true);

    /// Takes the GPU modified ranges in a range out of the tracked ones, adding copies of them.
    u64 CollectDownloads(const Buffer* arena, VAddr device_addr, u64 size, DownloadCopies& copies);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    std::vector<std::shared_ptr<Readback>> readbacks;

    /// Windows game threads read back recently, which are copied back ahead.
    struct HotWindow {
        VAddr start;
        VAddr end;
        std::chrono::steady_clock::time_point last_fault;
        /// Chances to copy it ahead skipped after copies of it went stale, and still to skip.
        u8 backoff{};
        u8 skip{};
    };
    std::vector<HotWindow> hot_windows;
    struct ReadbackStats {
        u64 on_fault{};
        u64 joined{};
        u64 prefetched{};
        u64 written_ahead{};
        u64 skipped{};
        /// Time from game threads asking for copies back to the GPU thread taking them up.
        u64 pickup_ns{};
        u64 pickups{};
        /// Command buffers submitted and not done yet when game threads asked, summed.
        u64 in_flight{};
        /// Copies asked for of memory the command buffer being recorded hadn't touched.
        u64 untouched{};
        /// Copies asked for of memory the game's compute rings wrote lately.
        u64 compute_ring{};
    } readback_stats;
    /// The last memory ranges the game's compute rings' dispatches wrote.
    std::array<std::pair<VAddr, VAddr>, 256> compute_ring_writes{};
    size_t compute_ring_write_index{};
    std::chrono::steady_clock::time_point last_readback_report{};

    /// Ranges written by dispatches routed to the async compute queue, with the
    /// compute-timeline tick that wrote them. Graphics consumers overlapping them
    /// wait for that tick at the next submit instead of serializing behind the
    /// dispatches. GPU thread only.
    struct ComputeWrite {
        VAddr start{};
        VAddr end{};
        u64 tick{};
    };
    std::vector<ComputeWrite> compute_writes;
    /// Latest compute tick graphics was told to wait for, consumed at submit.
    u64 pending_compute_wait{};
    /// Guards compute_writes and pending_compute_wait: the submit callback
    /// also runs on the presenter thread, while notes/checks run on the GPU
    /// thread. An untorn tick is load-bearing here: a garbage tick waits
    /// forever, so this must never race.
    std::mutex compute_writes_mutex;
    /// True when a compute-timeline tick is complete. Set once when async compute exists.
    std::function<bool(u64)> compute_is_free{};
    /// Fence of the async-compute timeline for sparse binds. Set with SetComputeFence.
    std::function<std::pair<vk::Semaphore, u64>()> compute_fence{};
    /// Marks graphics to wait for compute work overlapping a range. GPU thread.
    void CheckComputeOverlap(VAddr addr, u64 size);

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};
    struct RewriteCopy {
        VAddr address{};
        u64 frame{};
    };
    /// Addresses of memory the game keeps writing that were copied for a draw, and in which frame.
    std::array<RewriteCopy, 1024> rewrite_copies{};
    u64 rewrite_copy_frame{};
    u64 rewrite_copy_bytes{};
    /// The CPU modified generation all memory in use was last uploaded at for such shaders.
    u64 dma_synced_generation{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;
    vk::DeviceMemory residency_chunk{};
    u64 residency_chunk_used{};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};

    /// Copies made ahead, in the order they were recorded, to be written back once done.
    std::deque<std::shared_ptr<Readback>> finished_readbacks;
    std::mutex finished_readbacks_mutex;
    std::condition_variable_any finished_readbacks_cv;
    /// Declared last so it stops before anything it uses goes away.
    std::jthread readback_thread;

    /// Copies guest memory into staging and applies the page protections of the uploads
    /// on a thread of its own, off the command processor. A submission flushes every job
    /// first: the staging a job copies into must be complete before the command buffer
    /// using it is submitted, which is before the GPU reads it.
    class UploadWorker {
    public:
        explicit UploadWorker(PageManager& page_manager_);
        ~UploadWorker();

        UploadWorker(const UploadWorker&) = delete;
        UploadWorker& operator=(const UploadWorker&) = delete;

        struct Copy {
            /// Guest address to read.
            VAddr src;
            /// Host mapping to write.
            u8* dst;
            u64 size;
        };
        struct Job {
            /// Page ranges [first_page, end_page) whose protection to apply first, so a
            /// guest write after it faults and one before is caught by the copies.
            std::vector<std::pair<u64, u64>> pages;
            boost::container::small_vector<Copy, 8> copies;
            Vulkan::StagingBufferRef staging;
            u64 bytes{};
        };

        /// Queues a job. Single producer: the command processor thread.
        void Enqueue(Job&& job);

        /// Completes every queued job: waits for the ones the worker took, finishing
        /// what it left in the queue.
        void Flush();

        [[nodiscard]] bool Idle() noexcept;

    private:
        void Run(std::stop_token stoken);
        void ExecuteQueued();

        PageManager& page_manager;
        std::jthread thread;
        std::mutex queue_mutex;
        std::condition_variable_any queue_cv;
        std::vector<Job> queue;
        /// Held while the worker executes: Flush waits on it, so a job is either
        /// queued, executing or done, never lost in between.
        std::mutex exec_mutex;
        std::vector<Job> batch;
        std::vector<std::pair<u64, u64>> ranges;
        std::chrono::steady_clock::time_point last_report{};
        u64 stat_jobs{};
        u64 stat_bytes{};
        u64 stat_pages{};
        u64 stat_busy_ns{};
    };

    /// Null when the upload worker is off (SHADPS4_UPLOAD_WORKER=0).
    std::unique_ptr<UploadWorker> upload_worker;
    /// Page ranges whose protection the upload being prepared defers to the worker.
    /// Command processor thread only.
    std::vector<std::pair<u64, u64>> deferred_pages;
    /// Second queue for readback copies, set by the rasterizer. Null disables it.
    Vulkan::ComputeQueue* readback_queue{};
    /// Tick of the submission that last wrote to a GPU-bound buffer, so the readback
    /// copy waits for it rather than for the latest submission. GPU thread only.
    u64 last_write_tick{};
    /// True when the last write went to the compute queue's timeline (async
    /// compute batch), so the readback copy waits on that semaphore.
    bool last_write_on_compute{};
    /// Submits the open compute batch before a readback that covers its writes.
    std::function<void()> compute_batch_flusher{};
    /// Declared after the staging pool and the tracker so it stops before they go away.
};

} // namespace VideoCore
