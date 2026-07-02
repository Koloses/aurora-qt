/**
 * @file src/pyrowave/pyrowave_vk.h
 * @brief Minimal Vulkan context owner for the PyroWave GPU codec.
 *
 * PyroWave needs a live Vulkan device plus exactly one VMA `vk_allocator`
 * singleton (see third-party/pyrowave/README.md "Integration contract"). Sunshine
 * has no general-purpose Vulkan device exposed to the encoder layer, so we create
 * and own a self-contained one here. The encoder (and, by analogy, the
 * moonlight-qt decoder) drives PyroWave on this context.
 *
 * Scope note: this is the foundation for Milestone 2. It deliberately contains
 * no Sunshine capture/avcodec coupling. Built only when SUNSHINE_ENABLE_PYROWAVE.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

// Public headers from the vendored pyrowave static library (PUBLIC include dirs
// propagate through the linked `pyrowave` target).
#include <vulkan/vulkan_raii.hpp>
#include "vk/vk_allocator.h"

namespace pyrowave_vk {

  /**
   * @brief Required/optional Vulkan device features the codec relies on.
   */
  struct device_caps_t {
    bool shader_float16 = false;  ///< Enables the faster *_fp16 shader paths.
    bool timeline_semaphore = false;
    bool ycbcr_conversion = false;
    // VK_KHR_present_id + VK_KHR_present_wait are enabled (swapchain contexts
    // only). The decoder uses them to timestamp actual scanout of each frame
    // from a side thread and phase-lock the host capture clock to the client
    // display (smoothness without added buffering latency).
    bool present_wait = false;
    uint32_t compute_queue_family = UINT32_MAX;
  };

  /**
   * @brief Owns a Vulkan instance/device dedicated to PyroWave plus the VMA
   *        singleton the codec requires.
   *
   * Lifetime: construct once before any PyroWave::Encoder/Decoder and keep alive
   * for their entire lifetime. The `vk_allocator` is a process-global singleton,
   * so at most one `context` may exist per process.
   */
  class context {
  public:
    /**
     * @brief Create the Vulkan context.
     * @param prefer_luid Optional adapter LUID/UUID to match the capture GPU
     *        (so encoder input can later be imported without a cross-GPU copy).
     * @param instance_exts Extra instance extensions to enable (e.g. the
     *        surface extensions SDL requires to present to a window).
     * @param want_swapchain Enable VK_KHR_swapchain on the device so the caller
     *        can create a swapchain on a window surface and present.
     * @return nullptr on failure (no suitable device, missing extensions).
     */
    static std::unique_ptr<context> create(const uint8_t *prefer_uuid = nullptr,
                                            const std::vector<const char *> &instance_exts = {},
                                            bool want_swapchain = false);

    ~context();

    context(const context &) = delete;
    context &operator=(const context &) = delete;

    /// The Vulkan instance (used to create a window surface for presentation).
    vk::raii::Instance &instance() {
      return inst;
    }

    vk::raii::PhysicalDevice &physical_device() {
      return phys_dev;
    }

    vk::raii::Device &device() {
      return dev;
    }

    const device_caps_t &caps() const {
      return caps_;
    }

    /// The compute queue PyroWave command buffers are submitted on.
    vk::raii::Queue &queue() {
      return compute_queue;
    }

  private:
    context() = default;

    vk::raii::Context ctx {};
    vk::raii::Instance inst = nullptr;
    vk::raii::PhysicalDevice phys_dev = nullptr;
    vk::raii::Device dev = nullptr;
    vk::raii::Queue compute_queue = nullptr;
    device_caps_t caps_;

    // The VMA singleton required by the vendored allocation layer. Must outlive
    // every PyroWave object; destroyed last.
    std::optional<vk_allocator> allocator;
  };

}  // namespace pyrowave_vk
