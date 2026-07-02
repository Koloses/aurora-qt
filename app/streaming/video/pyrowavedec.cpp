/**
 * @file pyrowavedec.cpp
 * @brief PyroWave IVideoDecoder implementation. See header.
 *
 * Decode runs on a private Vulkan device (PyroWave::Decoder, compute IDWT path).
 * The decoded Y/Cb/Cr plane images are converted to RGBA by the yuv2rgba compute
 * shader and presented through a Vulkan swapchain created on moonlight's
 * SDL_WINDOW_VULKAN window - no CPU readback, no SDL_Renderer.
 */
#include "pyrowavedec.h"
#include "streaming/session.h"         // Session::get()->getOverlayManager()

#include "pyrowave/pyrowave_common.h"  // PyroWave::load_shader

#include <Limelight.h>                 // LiGetMicroseconds / LiGetEstimatedRttInfo / FRAME_TYPE_IDR
#include <SDL_vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
  // Push constants for yuv2rgba.comp: output (decode/surface) resolution + recip.
  struct ConvPush { int32_t w, h; float inv_w, inv_h; };

  // Push constants for overlay_blend.comp: overlay rect (top-left + size) in the target.
  struct OverlayPush { int32_t ox, oy, w, h; };

  image_allocation make_plane_image(vk::raii::Device &device, uint32_t w, uint32_t h, const char *name) {
    vk::ImageCreateInfo info {
      .imageType = vk::ImageType::e2D,
      .format = vk::Format::eR8Unorm,
      .extent = {.width = w, .height = h, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
               vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };
    return image_allocation(device, info, {.usage = VMA_MEMORY_USAGE_AUTO}, name);
  }

  vk::raii::ImageView make_plane_view(vk::raii::Device &device, vk::Image img) {
    return device.createImageView(vk::ImageViewCreateInfo {
      .image = img,
      .viewType = vk::ImageViewType::e2D,
      .format = vk::Format::eR8Unorm,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
    });
  }
}  // namespace

PyroWaveVideoDecoder::PyroWaveVideoDecoder(bool testOnly):
    m_TestOnly(testOnly) {
}

PyroWaveVideoDecoder::~PyroWaveVideoDecoder() {
    // Stop the overlay manager from calling back into us as we tear down.
    if (!m_TestOnly) {
        Session::get()->getOverlayManager().setOverlayRenderer(nullptr);
    }
    if (m_Ctx) {
        try { waitPreviousFrame(); } catch (...) {}
        try { (void) m_Ctx->device().waitIdle(); } catch (...) {}
    }
}

QSize PyroWaveVideoDecoder::getDecoderMaxResolution() {
    return QSize(7680, 4320);
}

bool PyroWaveVideoDecoder::initialize(PDECODER_PARAMETERS params) {
    if (params->videoFormat != VIDEO_FORMAT_PYROWAVE) {
        return false;  // not ours; let chooseDecoder try the next decoder
    }

    m_Width = params->width;
    m_Height = params->height;
    m_VideoFormat = params->videoFormat;
    m_Window = params->window;

    // Instance extensions SDL needs to present to this window (VK_KHR_surface +
    // the platform surface extension). Required so context::create() builds an
    // instance we can make a swapchain surface on.
    std::vector<const char *> instExts;
    unsigned int nExt = 0;
    if (SDL_Vulkan_GetInstanceExtensions(m_Window, &nExt, nullptr) && nExt) {
        instExts.resize(nExt);
        SDL_Vulkan_GetInstanceExtensions(m_Window, &nExt, instExts.data());
    }

    m_Ctx = pyrowave_vk::context::create(nullptr, instExts, /*want_swapchain=*/true);
    if (!m_Ctx) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: Vulkan context unavailable");
        return false;
    }

    try {
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: creating PyroWave::Decoder %dx%d", m_Width, m_Height);
        // Compute IDWT path (fragment_path=false): the fragment path builds fp16
        // graphics pipelines that fault some desktop drivers during pipeline
        // creation; the compute path is the same route the Adreno/Tegra client uses.
        m_Decoder = std::make_unique<PyroWave::Decoder>(
            m_Ctx->physical_device(), m_Ctx->device(), m_Width, m_Height,
            PyroWave::ChromaSubsampling::Chroma420, /*fragment_path=*/false);
        m_Input = std::make_unique<PyroWave::DecoderInput>(*m_Decoder);
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: PyroWave::Decoder + input created");
    } catch (const std::exception &e) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: decoder init failed: %s", e.what());
        return false;
    }

    if (!initVulkanResources()) {
        return false;
    }
    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: Vulkan decode resources ready");

    // In test mode we only verify the codec can be set up; don't touch the window.
    if (m_TestOnly) {
        return true;
    }

    if (!initSwapchain() || !initConvertPipeline()) {
        return false;
    }

    // Overlay compositing is best-effort: if it fails to set up, we still stream
    // (just without the on-screen performance/status overlay).
    if (initOverlayPipeline()) {
        Session::get()->getOverlayManager().setOverlayRenderer(this);
    }

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: decoder initialized %dx%d (swapchain present)", m_Width, m_Height);
    return true;
}

bool PyroWaveVideoDecoder::initVulkanResources() {
    try {
        auto &device = m_Ctx->device();
        uint32_t cw = (uint32_t(m_Width) + 1) / 2;
        uint32_t ch = (uint32_t(m_Height) + 1) / 2;

        m_ImgY = make_plane_image(device, m_Width, m_Height, "pyrowave dec Y");
        m_ImgCb = make_plane_image(device, cw, ch, "pyrowave dec Cb");
        m_ImgCr = make_plane_image(device, cw, ch, "pyrowave dec Cr");
        m_ViewY = make_plane_view(device, m_ImgY);
        m_ViewCb = make_plane_view(device, m_ImgCb);
        m_ViewCr = make_plane_view(device, m_ImgCr);

        m_CmdPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_Ctx->caps().compute_queue_family});
        m_Cmd = std::move(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo {
            .commandPool = *m_CmdPool, .commandBufferCount = 1})[0]);
        m_Fence = vk::raii::Fence(device, vk::FenceCreateInfo {});
        for (int i = 0; i < 2; i++) {
            m_AcquireSem[i] = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo {});
            m_PresentSem[i] = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo {});
        }
        return true;
    } catch (const std::exception &e) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: vk resource init failed: %s", e.what());
        return false;
    }
}

bool PyroWaveVideoDecoder::initSwapchain() {
    try {
        auto &device = m_Ctx->device();
        auto &phys = m_Ctx->physical_device();

        // Surface on the SDL window using our own instance.
        VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(m_Window, static_cast<VkInstance>(*m_Ctx->instance()), &rawSurface)) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
            return false;
        }
        m_Surface = vk::raii::SurfaceKHR(m_Ctx->instance(), rawSurface);

        if (!phys.getSurfaceSupportKHR(m_Ctx->caps().compute_queue_family, *m_Surface)) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: queue family does not support present");
            return false;
        }

        auto caps = phys.getSurfaceCapabilitiesKHR(*m_Surface);
        auto formats = phys.getSurfaceFormatsKHR(*m_Surface);
        vk::ColorSpaceKHR colorspace = formats.empty() ? vk::ColorSpaceKHR::eSrgbNonlinear : formats[0].colorSpace;
        m_SwapFormat = formats.empty() ? vk::Format::eB8G8R8A8Unorm : formats[0].format;
        for (auto &f : formats) {
            if (f.format == vk::Format::eR8G8B8A8Unorm || f.format == vk::Format::eB8G8R8A8Unorm) {
                m_SwapFormat = f.format; colorspace = f.colorSpace; break;
            }
        }
        m_SwapExtent = (caps.currentExtent.width != 0xFFFFFFFFu)
            ? caps.currentExtent : vk::Extent2D {(uint32_t) m_Width, (uint32_t) m_Height};
        uint32_t imgCount = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) imgCount = std::min(imgCount, caps.maxImageCount);

        // Direct present (yuv2rgba writes the swapchain image) is only safe when the
        // swapchain image is a storage image AND format R8G8B8A8 - the shader stores
        // rgba8 by component, so B8G8R8A8 would swap R/B (the blit path converts).
        m_DirectPresent = (m_SwapFormat == vk::Format::eR8G8B8A8Unorm) &&
                          (bool) (caps.supportedUsageFlags & vk::ImageUsageFlagBits::eStorage);

        auto pretransform = (caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
            ? vk::SurfaceTransformFlagBitsKHR::eIdentity : caps.currentTransform;

        // Lowest-latency present: prefer IMMEDIATE, then MAILBOX, then FIFO.
        auto presentModes = phys.getSurfacePresentModesKHR(*m_Surface);
        auto hasMode = [&](vk::PresentModeKHR m) {
            return std::find(presentModes.begin(), presentModes.end(), m) != presentModes.end();
        };
        vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
        if (hasMode(vk::PresentModeKHR::eImmediate)) presentMode = vk::PresentModeKHR::eImmediate;
        else if (hasMode(vk::PresentModeKHR::eMailbox)) presentMode = vk::PresentModeKHR::eMailbox;

        m_Swapchain = vk::raii::SwapchainKHR(device, vk::SwapchainCreateInfoKHR {
            .surface = *m_Surface, .minImageCount = imgCount,
            .imageFormat = m_SwapFormat, .imageColorSpace = colorspace,
            .imageExtent = m_SwapExtent, .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment |
                          (m_DirectPresent ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits{}),
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = pretransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = presentMode,
            .clipped = true});
        m_SwapImages = m_Swapchain.getImages();
        if (m_DirectPresent) {
            for (auto img : m_SwapImages) {
                m_SwapStorageViews.push_back(device.createImageView(vk::ImageViewCreateInfo {
                    .image = img, .viewType = vk::ImageViewType::e2D, .format = m_SwapFormat,
                    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}}));
            }
        }
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: swapchain %ux%u present_mode=%d direct_present=%d",
                    m_SwapExtent.width, m_SwapExtent.height, (int) presentMode, (int) m_DirectPresent);
        return true;
    } catch (const std::exception &e) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: swapchain init failed: %s", e.what());
        return false;
    }
}

bool PyroWaveVideoDecoder::initConvertPipeline() {
    try {
        auto &device = m_Ctx->device();

        // Offscreen R8G8B8A8 compute target (blit path; matches the shader's rgba8).
        vk::ImageCreateInfo ri {
            .imageType = vk::ImageType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {.width = (uint32_t) m_Width, .height = (uint32_t) m_Height, .depth = 1},
            .mipLevels = 1, .arrayLayers = 1,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc};
        m_ImgRgba = image_allocation(device, ri, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave rgba");
        m_RgbaView = device.createImageView(vk::ImageViewCreateInfo {
            .image = m_ImgRgba, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});

        m_Sampler = vk::raii::Sampler(device, vk::SamplerCreateInfo {
            .magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge});

        std::array<vk::DescriptorSetLayoutBinding, 5> binds {{
            {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3, .descriptorType = vk::DescriptorType::eSampler,      .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        m_Dsl = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo {.bindingCount = (uint32_t) binds.size(), .pBindings = binds.data()});
        vk::PushConstantRange pcr {.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(ConvPush)};
        m_Pl = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo {.setLayoutCount = 1, .pSetLayouts = &*m_Dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr});
        auto convSh = PyroWave::load_shader(device, "yuv2rgba");
        vk::PipelineShaderStageCreateInfo convSt {.stage = vk::ShaderStageFlagBits::eCompute, .module = *convSh, .pName = "main"};
        m_Pipe = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo {.stage = convSt, .layout = *m_Pl});

        std::array<vk::DescriptorPoolSize, 3> psz {{
            {.type = vk::DescriptorType::eSampledImage, .descriptorCount = 3},
            {.type = vk::DescriptorType::eSampler,      .descriptorCount = 1},
            {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
        }};
        m_Dpool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo {.maxSets = 1, .poolSizeCount = (uint32_t) psz.size(), .pPoolSizes = psz.data()});
        m_Dset = std::move(vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo {.descriptorPool = *m_Dpool, .descriptorSetCount = 1, .pSetLayouts = &*m_Dsl}).front());

        vk::DescriptorImageInfo yi {.imageView = *m_ViewY,  .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo cbi {.imageView = *m_ViewCb, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo cri {.imageView = *m_ViewCr, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo si {.sampler = *m_Sampler};
        vk::DescriptorImageInfo oi {.imageView = *m_RgbaView, .imageLayout = vk::ImageLayout::eGeneral};
        std::array<vk::WriteDescriptorSet, 5> ws {{
            {.dstSet = *m_Dset, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &yi},
            {.dstSet = *m_Dset, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &cbi},
            {.dstSet = *m_Dset, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &cri},
            {.dstSet = *m_Dset, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler,      .pImageInfo = &si},
            {.dstSet = *m_Dset, .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &oi},
        }};
        device.updateDescriptorSets(ws, {});
        return true;
    } catch (const std::exception &e) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: convert pipeline init failed: %s", e.what());
        return false;
    }
}

bool PyroWaveVideoDecoder::initOverlayPipeline() {
    try {
        auto &device = m_Ctx->device();
        // binding 0: overlay texture (sampled), 1: sampler, 2: target RGBA storage image.
        std::array<vk::DescriptorSetLayoutBinding, 3> binds {{
            {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eSampler,      .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        m_OverlayDsl = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo {.bindingCount = (uint32_t) binds.size(), .pBindings = binds.data()});
        vk::PushConstantRange pcr {.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(OverlayPush)};
        m_OverlayPl = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo {.setLayoutCount = 1, .pSetLayouts = &*m_OverlayDsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr});
        auto sh = PyroWave::load_shader(device, "overlay_blend");
        vk::PipelineShaderStageCreateInfo st {.stage = vk::ShaderStageFlagBits::eCompute, .module = *sh, .pName = "main"};
        m_OverlayPipe = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo {.stage = st, .layout = *m_OverlayPl});
        std::array<vk::DescriptorPoolSize, 3> psz {{
            {.type = vk::DescriptorType::eSampledImage, .descriptorCount = 1},
            {.type = vk::DescriptorType::eSampler,      .descriptorCount = 1},
            {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
        }};
        m_OverlayDpool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo {.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = 1, .poolSizeCount = (uint32_t) psz.size(), .pPoolSizes = psz.data()});
        m_OverlayDset = std::move(vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo {.descriptorPool = *m_OverlayDpool, .descriptorSetCount = 1, .pSetLayouts = &*m_OverlayDsl}).front());
        // Sampler (binding 1) is shared with the yuv2rgba pipeline.
        vk::DescriptorImageInfo si {.sampler = *m_Sampler};
        vk::WriteDescriptorSet w {.dstSet = *m_OverlayDset, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &si};
        device.updateDescriptorSets(w, {});
        return true;
    } catch (const std::exception &e) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: overlay pipeline init failed (overlay disabled): %s", e.what());
        return false;
    }
}

// Poll OverlayManager for a fresh overlay surface and upload it to the GPU
// texture (only when it actually changed, ~1/sec). Runs on the decoder thread.
void PyroWaveVideoDecoder::maybeUploadOverlayTexture() {
    if (!*m_OverlayPipe) {
        return;
    }
    SDL_Surface *surface = Session::get()->getOverlayManager().getUpdatedOverlaySurface(Overlay::OverlayDebug);
    if (!surface) {
        return;  // no change; keep the existing texture
    }
    SDL_Surface *conv = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_ABGR8888, 0);  // RGBA byte order
    SDL_FreeSurface(surface);
    if (!conv) {
        return;
    }

    try {
        auto &device = m_Ctx->device();
        uint32_t w = (uint32_t) conv->w, h = (uint32_t) conv->h;
        if ((int) w != m_OverlayW || (int) h != m_OverlayH || !m_OverlayValid) {
            (void) device.waitIdle();  // texture may be in use by a prior frame
            vk::ImageCreateInfo ci {
                .imageType = vk::ImageType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
                .extent = {.width = w, .height = h, .depth = 1}, .mipLevels = 1, .arrayLayers = 1,
                .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst};
            m_OverlayImg = image_allocation(device, ci, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave overlay");
            m_OverlayView = device.createImageView(vk::ImageViewCreateInfo {
                .image = m_OverlayImg, .viewType = vk::ImageViewType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
                .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
            m_OverlayStaging = buffer_allocation(device,
                {.size = vk::DeviceSize(w) * h * 4, .usage = vk::BufferUsageFlagBits::eTransferSrc},
                {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, .usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave overlay staging");
            m_OverlayW = (int) w; m_OverlayH = (int) h;
            // Point binding 0 at the new texture.
            vk::DescriptorImageInfo oi {.imageView = *m_OverlayView, .imageLayout = vk::ImageLayout::eGeneral};
            vk::WriteDescriptorSet ws0 {.dstSet = *m_OverlayDset, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &oi};
            device.updateDescriptorSets(ws0, {});
        }

        // Copy pixels (respecting source pitch) into the staging buffer.
        auto *dst = static_cast<uint8_t *>(m_OverlayStaging.map());
        for (uint32_t y = 0; y < h; ++y) {
            std::memcpy(dst + size_t(y) * w * 4,
                        static_cast<const uint8_t *>(conv->pixels) + size_t(y) * conv->pitch,
                        size_t(w) * 4);
        }

        // One-shot upload (staging -> image), ~1/sec.
        auto up = std::move(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo {.commandPool = *m_CmdPool, .commandBufferCount = 1})[0]);
        up.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        up.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier {
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite, .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = m_OverlayImg, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
        up.copyBufferToImage(m_OverlayStaging, m_OverlayImg, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy {
            .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .imageExtent = {.width = w, .height = h, .depth = 1}});
        up.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, vk::ImageMemoryBarrier {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal, .newLayout = vk::ImageLayout::eGeneral,
            .image = m_OverlayImg, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
        up.end();
        vk::raii::Fence f(device, vk::FenceCreateInfo {});
        m_Ctx->queue().submit(vk::SubmitInfo {.commandBufferCount = 1, .pCommandBuffers = &*up}, *f);
        (void) device.waitForFences(*f, true, UINT64_MAX);
        m_OverlayValid = true;
    } catch (const std::exception &e) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: overlay upload failed: %s", e.what());
        m_OverlayValid = false;
    }
    SDL_FreeSurface(conv);
}

// Blend the overlay texture (top-left) onto the target RGBA storage image.
// The target must already be in eGeneral with prior writes made visible.
void PyroWaveVideoDecoder::recordOverlayBlend(vk::ImageView targetView, uint32_t targetW, uint32_t targetH) {
    int ow = std::min(m_OverlayW, (int) targetW);
    int oh = std::min(m_OverlayH, (int) targetH);
    if (ow <= 0 || oh <= 0) {
        return;
    }
    vk::DescriptorImageInfo ti {.imageView = targetView, .imageLayout = vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet w {.dstSet = *m_OverlayDset, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &ti};
    m_Ctx->device().updateDescriptorSets(w, {});

    m_Cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_OverlayPipe);
    m_Cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_OverlayPl, 0, *m_OverlayDset, {});
    OverlayPush push {0, 0, ow, oh};
    m_Cmd.pushConstants<OverlayPush>(*m_OverlayPl, vk::ShaderStageFlagBits::eCompute, 0, push);
    m_Cmd.dispatch((uint32_t(ow) + 7) / 8, (uint32_t(oh) + 7) / 8, 1);
}

void PyroWaveVideoDecoder::addVideoStats(VIDEO_STATS& src, VIDEO_STATS& dst) {
    dst.receivedFrames += src.receivedFrames;
    dst.decodedFrames += src.decodedFrames;
    dst.renderedFrames += src.renderedFrames;
    dst.totalFrames += src.totalFrames;
    dst.networkDroppedFrames += src.networkDroppedFrames;
    dst.totalDecodeTimeUs += src.totalDecodeTimeUs;
    if (dst.minHostProcessingLatency == 0) {
        dst.minHostProcessingLatency = src.minHostProcessingLatency;
    } else if (src.minHostProcessingLatency != 0) {
        dst.minHostProcessingLatency = std::min(dst.minHostProcessingLatency, src.minHostProcessingLatency);
    }
    dst.maxHostProcessingLatency = std::max(dst.maxHostProcessingLatency, src.maxHostProcessingLatency);
    dst.totalHostProcessingLatency += src.totalHostProcessingLatency;
    dst.framesWithHostProcessingLatency += src.framesWithHostProcessingLatency;

    if (!LiGetEstimatedRttInfo(&dst.lastRtt, &dst.lastRttVariance)) {
        dst.lastRtt = 0;
        dst.lastRttVariance = 0;
    }
    if (!dst.measurementStartUs) {
        dst.measurementStartUs = src.measurementStartUs;
    }
    double secs = (double)(LiGetMicroseconds() - dst.measurementStartUs) / 1000000.0;
    if (secs > 0) {
        dst.totalFps    = (double)dst.totalFrames / secs;
        dst.receivedFps = (double)dst.receivedFrames / secs;
        dst.decodedFps  = (double)dst.decodedFrames / secs;
        dst.renderedFps = (double)dst.renderedFrames / secs;
    }
}

void PyroWaveVideoDecoder::stringifyVideoStats(VIDEO_STATS& stats, char* output, int length) {
    int offset = 0;
    int ret;
    output[0] = 0;

    if (stats.receivedFps > 0) {
        ret = snprintf(&output[offset], length - offset,
                       "Video stream: %dx%d %.2f FPS (Codec: PyroWave)\n"
                       "Incoming frame rate from network: %.2f FPS\n"
                       "Decoding frame rate: %.2f FPS\n"
                       "Rendering frame rate: %.2f FPS\n",
                       m_Width, m_Height, stats.totalFps,
                       stats.receivedFps, stats.decodedFps, stats.renderedFps);
        if (ret < 0 || ret >= length - offset) return;
        offset += ret;
    }
    if (stats.framesWithHostProcessingLatency > 0) {
        ret = snprintf(&output[offset], length - offset,
                       "Host processing latency min/max/average: %.1f/%.1f/%.1f ms\n",
                       (float)stats.minHostProcessingLatency / 10,
                       (float)stats.maxHostProcessingLatency / 10,
                       (float)stats.totalHostProcessingLatency / 10 / stats.framesWithHostProcessingLatency);
        if (ret < 0 || ret >= length - offset) return;
        offset += ret;
    }
    if (stats.renderedFrames != 0) {
        char rttString[32];
        if (stats.lastRtt != 0) {
            snprintf(rttString, sizeof(rttString), "%u ms (variance: %u ms)", stats.lastRtt, stats.lastRttVariance);
        } else {
            snprintf(rttString, sizeof(rttString), "N/A");
        }
        ret = snprintf(&output[offset], length - offset,
                       "Frames dropped by your network connection: %.2f%%\n"
                       "Average network latency: %s\n"
                       "Average decoding+present time: %.2f ms\n",
                       stats.totalFrames ? (float)stats.networkDroppedFrames / stats.totalFrames * 100 : 0.f,
                       rttString,
                       stats.decodedFrames ? (double)(stats.totalDecodeTimeUs / 1000.0) / stats.decodedFrames : 0.0);
        if (ret < 0 || ret >= length - offset) return;
        offset += ret;
    }
}

void PyroWaveVideoDecoder::updateStatsAndOverlay(PDECODE_UNIT du, uint64_t decodeTimeUs, bool decoded) {
    if (!m_LastFrameNumber) {
        m_ActiveWndVideoStats.measurementStartUs = LiGetMicroseconds();
        m_LastFrameNumber = du->frameNumber;
    } else {
        // Frames numbered beyond the last+1 were dropped by the network.
        m_ActiveWndVideoStats.networkDroppedFrames += du->frameNumber - (m_LastFrameNumber + 1);
        m_ActiveWndVideoStats.totalFrames += du->frameNumber - (m_LastFrameNumber + 1);
        m_LastFrameNumber = du->frameNumber;
    }

    // Flip the stats window ~once per second and refresh the overlay text.
    if (LiGetMicroseconds() > m_ActiveWndVideoStats.measurementStartUs + 1000000) {
        if (Session::get()->getOverlayManager().isOverlayEnabled(Overlay::OverlayDebug)) {
            VIDEO_STATS lastTwoWndStats = {};
            addVideoStats(m_LastWndVideoStats, lastTwoWndStats);
            addVideoStats(m_ActiveWndVideoStats, lastTwoWndStats);
            stringifyVideoStats(lastTwoWndStats,
                                Session::get()->getOverlayManager().getOverlayText(Overlay::OverlayDebug),
                                Session::get()->getOverlayManager().getOverlayMaxTextLength());
            Session::get()->getOverlayManager().setOverlayTextUpdated(Overlay::OverlayDebug);
        }
        addVideoStats(m_ActiveWndVideoStats, m_GlobalVideoStats);
        SDL_memcpy(&m_LastWndVideoStats, &m_ActiveWndVideoStats, sizeof(m_ActiveWndVideoStats));
        SDL_zero(m_ActiveWndVideoStats);
        m_ActiveWndVideoStats.measurementStartUs = LiGetMicroseconds();
    }

    if (du->frameHostProcessingLatency != 0) {
        if (m_ActiveWndVideoStats.minHostProcessingLatency != 0) {
            m_ActiveWndVideoStats.minHostProcessingLatency = std::min(m_ActiveWndVideoStats.minHostProcessingLatency, du->frameHostProcessingLatency);
        } else {
            m_ActiveWndVideoStats.minHostProcessingLatency = du->frameHostProcessingLatency;
        }
        m_ActiveWndVideoStats.maxHostProcessingLatency = std::max(m_ActiveWndVideoStats.maxHostProcessingLatency, du->frameHostProcessingLatency);
        m_ActiveWndVideoStats.totalHostProcessingLatency += du->frameHostProcessingLatency;
        m_ActiveWndVideoStats.framesWithHostProcessingLatency++;
    }

    m_ActiveWndVideoStats.receivedFrames++;
    m_ActiveWndVideoStats.totalFrames++;
    if (decoded) {
        m_ActiveWndVideoStats.decodedFrames++;
        m_ActiveWndVideoStats.renderedFrames++;
        m_ActiveWndVideoStats.totalDecodeTimeUs += decodeTimeUs;
    }
}

bool PyroWaveVideoDecoder::decodeAndPresent() {
    auto &device = m_Ctx->device();

    uint32_t idx = 0;
    // Phase-offset pacing (pyrofling-style): the time blocked in acquireNextImage is the
    // display-clock backpressure - how long we waited for the compositor/scanout to free a
    // swapchain image. Under a vsync-gated present mode this grows when the host produces
    // frames faster than this client can display them; under IMMEDIATE it stays ~0. We feed
    // it to the host as the phase offset (positive => host should slow its capture cadence).
    // pyrofling's server controller integrates with a deadband and recenters toward the
    // requested rate, so a ~0 signal is inert and this can never push the host below its
    // base FPS - it only trims overproduction.
    uint64_t acquireStartUs = LiGetMicroseconds();
    try {
        auto [res, i] = m_Swapchain.acquireNextImage(UINT64_MAX, *m_AcquireSem[m_Parity], nullptr);
        if (res != vk::Result::eSuccess && res != vk::Result::eSuboptimalKHR) {
            return true;  // transient; skip this frame
        }
        idx = i;
    } catch (const vk::SystemError &e) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: acquireNextImage failed: %s", e.what());
        return true;
    }
    if (!m_FirstFramePresented) {
        // Skip the cold-start acquire (no steady state yet); arm for subsequent frames.
        m_FirstFramePresented = true;
    } else {
        int64_t blockUs = (int64_t) (LiGetMicroseconds() - acquireStartUs);
        if (blockUs < 0) blockUs = 0;
        if (blockUs > 50000) blockUs = 50000;  // clamp pathological stalls (~50 ms)
        LiSendPhaseOffset((int) blockUs);
    }

    // Refresh the overlay texture (own submit) before we start the frame's cmd buffer.
    bool overlayOn = Session::get()->getOverlayManager().isOverlayEnabled(Overlay::OverlayDebug);
    if (overlayOn) {
        maybeUploadOverlayTexture();
    }

    m_Cmd.reset();
    m_Cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    auto barrier = [&](vk::Image image, vk::ImageLayout oldL, vk::ImageLayout newL,
                       vk::AccessFlags src, vk::AccessFlags dst,
                       vk::PipelineStageFlags ss, vk::PipelineStageFlags ds) {
        m_Cmd.pipelineBarrier(ss, ds, {}, {}, {}, vk::ImageMemoryBarrier {
            .srcAccessMask = src, .dstAccessMask = dst, .oldLayout = oldL, .newLayout = newL,
            .image = image,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
    };

    // Planes -> General for the decode (compute writes them).
    // Copy-initialize to vk::Image (image_allocation has both vk::Image and
    // VkImage conversion operators; a static_cast/function-style cast is
    // ambiguous on MSVC, but plain copy-init only matches the direct one).
    vk::Image imgY = m_ImgY, imgCb = m_ImgCb, imgCr = m_ImgCr;
    const std::array<vk::Image, 3> planes { imgY, imgCb, imgCr };
    auto planeOld = m_ImagesInitialized ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined;
    for (vk::Image p : planes) {
        barrier(p, planeOld, vk::ImageLayout::eGeneral,
                m_ImagesInitialized ? vk::AccessFlagBits::eShaderRead : vk::AccessFlags{},
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eComputeShader);
    }
    m_ImagesInitialized = true;

    PyroWave::Decoder::ViewBuffers views {*m_ViewY, *m_ViewCb, *m_ViewCr};
    if (!m_Decoder->decode(m_Cmd, *m_Input, views)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: decode() recorded failure");
    }

    // Planes: decode-write -> compute-sample.
    for (vk::Image p : planes) {
        barrier(p, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader);
    }

    vk::PipelineStageFlags waitStage;
    if (m_DirectPresent) {
        // yuv2rgba writes the acquired swapchain image directly (scaling decode-res
        // planes -> surface-res). Repoint binding 4; previous frame already waited.
        vk::DescriptorImageInfo oi {.imageView = *m_SwapStorageViews[idx], .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet w {.dstSet = *m_Dset, .dstBinding = 4, .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &oi};
        device.updateDescriptorSets(w, {});

        barrier(m_SwapImages[idx], vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                {}, vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);

        m_Cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_Pipe);
        m_Cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_Pl, 0, *m_Dset, {});
        ConvPush push {(int32_t) m_SwapExtent.width, (int32_t) m_SwapExtent.height,
                       1.0f / float(m_SwapExtent.width), 1.0f / float(m_SwapExtent.height)};
        m_Cmd.pushConstants<ConvPush>(*m_Pl, vk::ShaderStageFlagBits::eCompute, 0, push);
        m_Cmd.dispatch((m_SwapExtent.width + 7) / 8, (m_SwapExtent.height + 7) / 8, 1);

        if (overlayOn && m_OverlayValid) {
            barrier(m_SwapImages[idx], vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader);
            recordOverlayBlend(*m_SwapStorageViews[idx], m_SwapExtent.width, m_SwapExtent.height);
        }

        barrier(m_SwapImages[idx], vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits::eShaderWrite, {},
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eBottomOfPipe);

        waitStage = vk::PipelineStageFlagBits::eComputeShader;
    } else {
        // Offscreen RGBA -> General for compute write (fully overwritten).
        barrier(m_ImgRgba, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                {}, vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);

        m_Cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_Pipe);
        m_Cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_Pl, 0, *m_Dset, {});
        ConvPush push {m_Width, m_Height, 1.0f / float(m_Width), 1.0f / float(m_Height)};
        m_Cmd.pushConstants<ConvPush>(*m_Pl, vk::ShaderStageFlagBits::eCompute, 0, push);
        m_Cmd.dispatch((uint32_t(m_Width) + 7) / 8, (uint32_t(m_Height) + 7) / 8, 1);

        if (overlayOn && m_OverlayValid) {
            barrier(m_ImgRgba, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader);
            recordOverlayBlend(*m_RgbaView, (uint32_t) m_Width, (uint32_t) m_Height);
        }

        barrier(m_ImgRgba, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer);
        barrier(m_SwapImages[idx], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                {}, vk::AccessFlagBits::eTransferWrite,
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);

        vk::ImageBlit blit {
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .srcOffsets = std::array<vk::Offset3D, 2> {vk::Offset3D {0, 0, 0}, vk::Offset3D {m_Width, m_Height, 1}},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstOffsets = std::array<vk::Offset3D, 2> {vk::Offset3D {0, 0, 0}, vk::Offset3D {(int32_t) m_SwapExtent.width, (int32_t) m_SwapExtent.height, 1}},
        };
        m_Cmd.blitImage(m_ImgRgba, vk::ImageLayout::eTransferSrcOptimal,
                        m_SwapImages[idx], vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier(m_SwapImages[idx], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
                vk::AccessFlagBits::eTransferWrite, {},
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe);

        waitStage = vk::PipelineStageFlagBits::eTransfer;
    }

    m_Cmd.end();
    device.resetFences(*m_Fence);
    m_Ctx->queue().submit(vk::SubmitInfo {
        .waitSemaphoreCount = 1, .pWaitSemaphores = &*m_AcquireSem[m_Parity], .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1, .pCommandBuffers = &*m_Cmd,
        .signalSemaphoreCount = 1, .pSignalSemaphores = &*m_PresentSem[m_Parity]}, *m_Fence);

    vk::SwapchainKHR sc = *m_Swapchain;
    try {
        (void) m_Ctx->queue().presentKHR(vk::PresentInfoKHR {
            .waitSemaphoreCount = 1, .pWaitSemaphores = &*m_PresentSem[m_Parity],
            .swapchainCount = 1, .pSwapchains = &sc, .pImageIndices = &idx});
    } catch (const vk::SystemError &) {
        // out-of-date / suboptimal: ignored in this version (no resize handling yet).
    }

    // Pipelined: do NOT wait here. The next frame's waitPreviousFrame() waits
    // this GPU work, so decode+present overlaps the network receive of the
    // next frame (mirrors the Android decoder).
    m_HavePending = true;
    m_Parity ^= 1;
    return true;
}

int PyroWaveVideoDecoder::submitDecodeUnit(PDECODE_UNIT du) {
    // Pipelining: wait for the previous frame's GPU work (which overlapped this
    // frame's network receive) before clearing/writing the input buffers the
    // GPU reads from.
    waitPreviousFrame();
    m_Input->clear();

    // Accumulate the frame's reassembled buffer chain into the decoder input.
    // Each entry is one RTP payload; the host aligns block packets to payload
    // boundaries, so resync at each entry (a mid-record state means the
    // previous payload was lost - partial frames are delivered on loss).
    for (PLENTRY entry = du->bufferList; entry != nullptr; entry = entry->next) {
        m_Input->resync_at_packet_boundary();
        std::span<const uint8_t> span(reinterpret_cast<const uint8_t *>(entry->data), (size_t) entry->length);
        if (!m_Input->push_data(span)) {
            SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: push_data rejected frame %d", du->frameNumber);
            m_Input->clear();
            return DR_NEED_IDR;
        }
    }

    // Make the CPU-written bitstream/offset buffers visible to the GPU before decode
    // (non-coherent HOST_CACHED memory needs an explicit flush; no-op on coherent).
    m_Input->flush();

    uint64_t decodeStartUs = LiGetMicroseconds();
    bool ok = decodeAndPresent();
    uint64_t decodeTimeUs = LiGetMicroseconds() - decodeStartUs;
    // NB: no clear() here - the GPU may still be reading the input buffers;
    // the next submit clears them after waitPreviousFrame().

    updateStatsAndOverlay(du, decodeTimeUs, ok);

    return ok ? DR_OK : DR_NEED_IDR;
}
