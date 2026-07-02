/**
 * @file pyrowavedec.h
 * @brief Self-contained IVideoDecoder for the PyroWave GPU wavelet codec.
 *
 * Modelled on SLVideoDecoder (a self-contained decoder that owns its own
 * presentation). Decode is driven by PyroWave::Decoder on a private Vulkan
 * device; the decoded YCbCr planes are converted to RGBA by a compute shader
 * (yuv2rgba) and presented via a Vulkan swapchain created on the (Vulkan)
 * SDL_Window. No CPU readback and no SDL_Renderer.
 *
 * Because pyrowave owns presentation (bypassing moonlight's ffmpeg renderers),
 * it also collects VIDEO_STATS and composites the performance/status overlay
 * onto its own frames - it implements IOverlayRenderer for that.
 */
#pragma once

#include "decoder.h"
#include "overlaymanager.h"

#include <SDL.h>
#include <array>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <vulkan/vulkan_raii.hpp>
#include "pyrowave/pyrowave_decoder.h"
#include "vk/allocation.h"

#include "pyrowave/pyrowave_vk.h"  // app/streaming/video/pyrowave/pyrowave_vk.h

class PyroWaveVideoDecoder : public IVideoDecoder, public Overlay::IOverlayRenderer {
public:
    explicit PyroWaveVideoDecoder(bool testOnly);
    virtual ~PyroWaveVideoDecoder() override;

    virtual bool initialize(PDECODER_PARAMETERS params) override;
    virtual bool isHardwareAccelerated() override { return true; }
    virtual bool isAlwaysFullScreen() override { return false; }
    // HDR10 is supported when the surface offers an ST 2084 colorspace;
    // initialize() verifies that per stream and rejects the HDR profile
    // otherwise (so the session falls back to SDR PyroWave).
    virtual bool isHdrSupported() override { return true; }
    virtual int getDecoderCapabilities() override { return 0; }
    virtual int getDecoderColorspace() override { return COLORSPACE_REC_709; }
    virtual int getDecoderColorRange() override { return COLOR_RANGE_LIMITED; }
    virtual QSize getDecoderMaxResolution() override;
    virtual int submitDecodeUnit(PDECODE_UNIT du) override;
    virtual void renderFrameOnMainThread() override {}  // present happens in submitDecodeUnit
    virtual void setHdrMode(bool) override {}
    virtual bool notifyWindowChanged(PWINDOW_STATE_CHANGE_INFO) override { return false; }

    // IOverlayRenderer: pyrowave polls getUpdatedOverlaySurface() in its present
    // path, so the notification is only a hint (no work needed here).
    virtual void notifyOverlayUpdated(Overlay::OverlayType) override {}

private:
    bool initVulkanResources();         // decode planes + command/fence/semaphores
    bool initSwapchain();               // surface + swapchain on the SDL window
    bool initConvertPipeline();         // yuv2rgba compute pipeline + descriptors
    bool initOverlayPipeline();         // overlay_blend compute pipeline + descriptors
    bool decodeAndPresent();            // decode -> yuv2rgba -> overlay -> present

    // Stats / performance overlay (mirrors FFmpegVideoDecoder).
    void addVideoStats(VIDEO_STATS& src, VIDEO_STATS& dst);
    void stringifyVideoStats(VIDEO_STATS& stats, char* output, int length);
    void updateStatsAndOverlay(PDECODE_UNIT du, uint64_t decodeTimeUs, bool decoded);

    // Overlay compositing.
    void maybeUploadOverlayTexture();   // pull a fresh overlay surface -> GPU texture
    void recordOverlayBlend(vk::ImageView targetView, uint32_t targetW, uint32_t targetH);

    bool m_TestOnly;
    int m_Width = 0;
    int m_Height = 0;
    int m_VideoFormat = 0;
    SDL_Window* m_Window = nullptr;

    std::shared_ptr<pyrowave_vk::context> m_Ctx;
    std::unique_ptr<PyroWave::Decoder> m_Decoder;
    std::unique_ptr<PyroWave::DecoderInput> m_Input;

    // Decoded YCbCr planes (separate R8 images: Y full-res, Cb/Cr half-res).
    image_allocation m_ImgY, m_ImgCb, m_ImgCr;
    vk::raii::ImageView m_ViewY = nullptr, m_ViewCb = nullptr, m_ViewCr = nullptr;
    bool m_ImagesInitialized = false;
    bool m_FirstFramePresented = false;                 // gate the cold-start acquire out of phase pacing

    vk::raii::CommandPool m_CmdPool = nullptr;
    vk::raii::CommandBuffer m_Cmd = nullptr;
    vk::raii::Fence m_Fence = nullptr;
    // Ping-pong present sync (parity-indexed) so one frame can be in GPU
    // flight while the receive thread depacketizes the next. Mirrors the
    // Android decoder's pipelining.
    vk::raii::Semaphore m_AcquireSem[2] = {nullptr, nullptr};
    vk::raii::Semaphore m_PresentSem[2] = {nullptr, nullptr};
    bool m_HavePending = false;  // a submitted-but-not-yet-waited frame exists
    uint32_t m_Parity = 0;

    // --- Phase-lock feedback via VK_KHR_present_wait -----------------------
    // A side thread waits on present IDs to timestamp each frame's actual
    // scanout, measures the submit->scanout margin, and the decoder thread
    // forwards (margin - target) to the host via LiSendPhaseOffset. The host's
    // existing pacing controller then shifts its capture phase so frames
    // arrive just-in-time for the client display: smoothness with no jitter
    // buffer. Falls back to the FIFO acquire-block signal when unsupported.
    bool m_UsePresentWait = false;
    bool m_SwapchainStale = false;  // out-of-date/suboptimal seen; recreate before next frame
    bool m_Chroma444 = false;       // stream uses the PyroWave 4:4:4 profile
    bool m_Hdr = false;             // stream uses a PyroWave HDR10 profile (10-bit BT.2020 + PQ)

    bool createSwapchain();     // (re)creates the swapchain for the existing surface
    bool recreateSwapchain();   // teardown + createSwapchain, called on resize/out-of-date
    uint64_t m_NextPresentId = 0;   // last ID handed to presentKHR (0 = none)
    std::thread m_PresentWaitThread;
    std::mutex m_PresentWaitMutex;
    std::condition_variable m_PresentWaitCv;
    std::deque<std::pair<uint64_t, uint64_t>> m_PendingPresents;  // (id, submit time us)
    bool m_PresentWaitStop = false;
    std::atomic<int> m_PhaseErrUs {INT_MIN};  // INT_MIN = no fresh measurement

    void presentWaitThreadProc();
    void stopPresentWaitThread();

    // Wait for the previously submitted frame's GPU work to finish. Must be
    // called before touching the shared decode resources or the CPU-written
    // input buffers for the next frame.
    void waitPreviousFrame() {
        if (m_HavePending) {
            (void) m_Ctx->device().waitForFences(*m_Fence, true, UINT64_MAX);
            m_HavePending = false;
        }
    }

    // Swapchain presentation on the SDL_WINDOW_VULKAN window.
    vk::raii::SurfaceKHR m_Surface = nullptr;
    vk::raii::SwapchainKHR m_Swapchain = nullptr;
    std::vector<vk::Image> m_SwapImages;
    vk::Format m_SwapFormat = vk::Format::eUndefined;
    vk::Extent2D m_SwapExtent {};
    bool m_DirectPresent = false;                       // yuv2rgba writes the swapchain image directly
    std::vector<vk::raii::ImageView> m_SwapStorageViews;  // per-image, direct path only

    // yuv2rgba compute pipeline (Y/Cb/Cr sampled images + sampler -> RGBA storage image).
    image_allocation m_ImgRgba;                         // offscreen RGBA target (blit path)
    vk::raii::ImageView m_RgbaView = nullptr;
    vk::raii::Sampler m_Sampler = nullptr;
    vk::raii::DescriptorSetLayout m_Dsl = nullptr;
    vk::raii::PipelineLayout m_Pl = nullptr;
    vk::raii::Pipeline m_Pipe = nullptr;
    vk::raii::DescriptorPool m_Dpool = nullptr;
    vk::raii::DescriptorSet m_Dset = nullptr;

    // overlay_blend compute pipeline (overlay texture + sampler -> RGBA storage target).
    image_allocation m_OverlayImg;
    vk::raii::ImageView m_OverlayView = nullptr;
    int m_OverlayW = 0, m_OverlayH = 0;
    bool m_OverlayValid = false;
    buffer_allocation m_OverlayStaging;
    vk::raii::DescriptorSetLayout m_OverlayDsl = nullptr;
    vk::raii::PipelineLayout m_OverlayPl = nullptr;
    vk::raii::Pipeline m_OverlayPipe = nullptr;
    vk::raii::DescriptorPool m_OverlayDpool = nullptr;
    vk::raii::DescriptorSet m_OverlayDset = nullptr;

    // Performance-overlay stats windows (mirrors FFmpegVideoDecoder).
    VIDEO_STATS m_ActiveWndVideoStats {};
    VIDEO_STATS m_LastWndVideoStats {};
    VIDEO_STATS m_GlobalVideoStats {};
    uint32_t m_LastFrameNumber = 0;
};
