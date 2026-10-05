// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_compute_queue.h"

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

ComputeQueue::ComputeQueue(const Instance& instance_)
    : instance{instance_}, semaphore{instance_}, pool{instance_, &semaphore} {
    if (instance.HasComputeQueue()) {
        queue = instance.GetComputeQueue();
    }
}

ComputeQueue::~ComputeQueue() = default;

vk::CommandBuffer ComputeQueue::Begin() {
    ASSERT_MSG(!open, "Compute command buffer already open");
    current = pool.Commit();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    const auto result = current.begin(begin_info);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to begin compute command buffer: {}",
               vk::to_string(result));
    open = true;
    return current;
}

u64 ComputeQueue::Submit(std::span<const std::pair<vk::Semaphore, u64>> waits) {
    ASSERT_MSG(open, "No open compute command buffer to submit");
    const auto end_result = current.end();
    ASSERT_MSG(end_result == vk::Result::eSuccess, "Failed to end compute command buffer: {}",
               vk::to_string(end_result));
    open = false;

    const u64 signal_value = semaphore.NextTick();
    const vk::Semaphore signal_sema = semaphore.Handle();
    std::vector<vk::Semaphore> wait_semas;
    std::vector<u64> wait_values;
    std::vector<vk::PipelineStageFlags> wait_masks;
    wait_semas.reserve(waits.size());
    wait_values.reserve(waits.size());
    wait_masks.reserve(waits.size());
    for (const auto& [sema, value] : waits) {
        wait_semas.push_back(sema);
        wait_values.push_back(value);
        wait_masks.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = static_cast<u32>(wait_semas.size()),
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_value,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = static_cast<u32>(wait_semas.size()),
        .pWaitSemaphores = wait_semas.data(),
        .pWaitDstStageMask = wait_masks.data(),
        .commandBufferCount = 1u,
        .pCommandBuffers = &current,
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    std::scoped_lock lock{submit_mutex};
    const auto submit_result = queue.submit(submit_info, nullptr);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during compute submit");
    current = vk::CommandBuffer{};
    return signal_value;
}

} // namespace Vulkan
