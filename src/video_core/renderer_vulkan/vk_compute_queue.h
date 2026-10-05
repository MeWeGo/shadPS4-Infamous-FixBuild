// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace Vulkan {

class Instance;

/// A second queue beside the graphics queue for compute work, with its own
/// timeline. Same queue family, so no ownership transfers are needed.
/// Calls are serialized by the caller (the GPU thread for now).
class ComputeQueue {
public:
    explicit ComputeQueue(const Instance& instance);
    ~ComputeQueue();

    ComputeQueue(const ComputeQueue&) = delete;
    ComputeQueue& operator=(const ComputeQueue&) = delete;

    /// Whether a second queue exists. Without one everything stays serialized.
    [[nodiscard]] bool IsAvailable() const noexcept {
        return !!queue;
    }

    [[nodiscard]] Semaphore& GetSemaphore() noexcept {
        return semaphore;
    }

    /// Begins a compute command buffer. Only one may be open at a time.
    vk::CommandBuffer Begin();

    /// Ends and submits the open command buffer, signalling the timeline.
    /// Waits are (semaphore, value) pairs the submission waits for first.
    /// Returns the signalled tick.
    u64 Submit(std::span<const std::pair<vk::Semaphore, u64>> waits = {});

    /// Host-side wait for a tick.
    void Wait(u64 tick) {
        semaphore.Wait(tick);
    }

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return semaphore.CurrentTick();
    }

private:
    const Instance& instance;
    vk::Queue queue{};
    Semaphore semaphore;
    CommandPool pool;
    vk::CommandBuffer current{};
    bool open{};
    std::mutex submit_mutex;
};

} // namespace Vulkan
