/**
 * @file pyrowave_vk.cpp
 * @brief Vulkan context for the PyroWave decoder (moonlight-qt copy).
 *
 * Mirrors Sunshine's src/pyrowave/pyrowave_vk.cpp but logs via SDL. Creates a
 * headless Vulkan device + the VMA vk_allocator singleton the codec requires.
 * NOT yet compiled (no Vulkan SDK in the authoring environment); validate on a
 * real host with `qmake CONFIG+=pyrowave`. TODO markers note spots to revisit.
 */
#include "pyrowave_vk.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <SDL.h>

#include <vk_mem_alloc.h>

namespace pyrowave_vk {

  namespace {

    int select_physical_device(const std::vector<vk::raii::PhysicalDevice> &devices, const uint8_t *prefer_uuid) {
      int fallback = -1;
      for (int i = 0; i < (int) devices.size(); ++i) {
        auto props = devices[i].getProperties2();
        const auto &p = props.properties;
        if (prefer_uuid && std::memcmp(p.pipelineCacheUUID.data(), prefer_uuid, VK_UUID_SIZE) == 0) {
          return i;
        }
        if (fallback < 0 || p.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
          fallback = i;
        }
      }
      return fallback;
    }

    uint32_t find_compute_family(vk::raii::PhysicalDevice &dev) {
      auto families = dev.getQueueFamilyProperties();
      for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
          return i;
        }
      }
      return UINT32_MAX;
    }

  }  // namespace

  std::unique_ptr<context> context::create(const uint8_t *prefer_uuid,
                                            const std::vector<const char *> &instance_exts,
                                            bool want_swapchain) {
    try {
      auto self = std::unique_ptr<context>(new context());

      // Highest instance version the loader offers (capped at 1.3) so the codec's
      // VkPhysicalDeviceVulkan13Properties subgroup query is populated; the device
      // is created with structs valid on any version (works on a 1.1 device too).
      uint32_t loader_version = self->ctx.enumerateInstanceVersion();
      uint32_t api_version = std::min(loader_version, (uint32_t) VK_API_VERSION_1_3);
      vk::ApplicationInfo app_info {
        .pApplicationName = "moonlight-pyrowave",
        .apiVersion = api_version,
      };
      vk::InstanceCreateInfo inst_info {.pApplicationInfo = &app_info};
      // Surface extensions (e.g. VK_KHR_surface + the platform one) requested by
      // the caller so a window swapchain can be created for presentation.
      if (!instance_exts.empty()) {
        inst_info.enabledExtensionCount = (uint32_t) instance_exts.size();
        inst_info.ppEnabledExtensionNames = instance_exts.data();
      }
      self->inst = vk::raii::Instance(self->ctx, inst_info);

      vk::raii::PhysicalDevices phys_devices(self->inst);
      if (phys_devices.empty()) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: no Vulkan physical devices");
        return nullptr;
      }
      int idx = select_physical_device(phys_devices, prefer_uuid);
      if (idx < 0) {
        return nullptr;
      }
      self->phys_dev = std::move(phys_devices[idx]);

      uint32_t cqf = find_compute_family(self->phys_dev);
      if (cqf == UINT32_MAX) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: no compute queue family");
        return nullptr;
      }
      self->caps_.compute_queue_family = cqf;

      // On a Vulkan 1.1 device the 1.2/1.3 feature aggregates are invalid in the
      // device-create chain, so enable the codec's needs via individual KHR/EXT
      // extensions. 16-bit storage and YCbCr conversion are core 1.1. Mirrors
      // WiVRn's "Lower Vulkan requirement to 1.1" device setup.
      static const char *wanted_exts[] = {
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        // The codec records vkCmdPipelineBarrier2 (Synchronization2); enable the
        // extension so it is available on 1.1 devices (null fn ptr -> crash otherwise).
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
      };
      std::vector<const char *> enabled_exts;
      bool have_present_id_ext = false, have_present_wait_ext = false;
      {
        auto avail = self->phys_dev.enumerateDeviceExtensionProperties();
        for (auto *w : wanted_exts) {
          for (auto &e : avail) {
            if (std::strcmp(e.extensionName, w) == 0) { enabled_exts.push_back(w); break; }
          }
        }
        // VK_KHR_swapchain is needed to present to a window surface.
        if (want_swapchain) {
          for (auto &e : avail) {
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
              enabled_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME); break;
            }
          }
          // Present-wait pair (phase-lock feedback); both or neither.
          for (auto &e : avail) {
            if (std::strcmp(e.extensionName, VK_KHR_PRESENT_ID_EXTENSION_NAME) == 0) have_present_id_ext = true;
            if (std::strcmp(e.extensionName, VK_KHR_PRESENT_WAIT_EXTENSION_NAME) == 0) have_present_wait_ext = true;
          }
        }
      }
      auto supported = self->phys_dev.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDevice8BitStorageFeatures,
        vk::PhysicalDeviceSubgroupSizeControlFeatures,
        vk::PhysicalDeviceShaderFloat16Int8Features,
        vk::PhysicalDeviceTimelineSemaphoreFeatures,
        vk::PhysicalDeviceSynchronization2Features>();
      auto &q11 = supported.get<vk::PhysicalDeviceVulkan11Features>();
      auto &q8 = supported.get<vk::PhysicalDevice8BitStorageFeatures>();
      auto &qsg = supported.get<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      auto &qf16 = supported.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
      auto &qts = supported.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
      auto &qsync = supported.get<vk::PhysicalDeviceSynchronization2Features>();

      self->caps_.shader_float16 = qf16.shaderFloat16;
      self->caps_.timeline_semaphore = qts.timelineSemaphore;
      self->caps_.ycbcr_conversion = q11.samplerYcbcrConversion;

      // Present-wait feature query, gated on the extensions being enumerable
      // (querying feature structs of unsupported extensions is invalid).
      vk::PhysicalDevicePresentIdFeaturesKHR qpid {};
      vk::PhysicalDevicePresentWaitFeaturesKHR qpw {};
      if (have_present_id_ext && have_present_wait_ext) {
        auto sup_pw = self->phys_dev.getFeatures2<
          vk::PhysicalDeviceFeatures2,
          vk::PhysicalDevicePresentIdFeaturesKHR,
          vk::PhysicalDevicePresentWaitFeaturesKHR>();
        qpid.presentId = sup_pw.get<vk::PhysicalDevicePresentIdFeaturesKHR>().presentId;
        qpw.presentWait = sup_pw.get<vk::PhysicalDevicePresentWaitFeaturesKHR>().presentWait;
      }
      self->caps_.present_wait = qpid.presentId && qpw.presentWait;
      if (self->caps_.present_wait) {
        enabled_exts.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
        enabled_exts.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
      }

      float prio = 1.0f;
      vk::DeviceQueueCreateInfo queue_info {
        .queueFamilyIndex = cqf, .queueCount = 1, .pQueuePriorities = &prio};

      vk::PhysicalDeviceFeatures base_features {};
      base_features.shaderStorageImageWriteWithoutFormat = true;
      base_features.shaderInt16 = supported.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16;

      vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDevice8BitStorageFeatures,
        vk::PhysicalDeviceSubgroupSizeControlFeatures,
        vk::PhysicalDeviceShaderFloat16Int8Features,
        vk::PhysicalDeviceTimelineSemaphoreFeatures,
        vk::PhysicalDeviceSynchronization2Features,
        vk::PhysicalDevicePresentIdFeaturesKHR,
        vk::PhysicalDevicePresentWaitFeaturesKHR>
        dev_chain;

      auto &dci = dev_chain.get<vk::DeviceCreateInfo>();
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &queue_info;
      dci.enabledExtensionCount = (uint32_t) enabled_exts.size();
      dci.ppEnabledExtensionNames = enabled_exts.data();
      dci.pEnabledFeatures = &base_features;

      auto &e11 = dev_chain.get<vk::PhysicalDeviceVulkan11Features>();
      e11.samplerYcbcrConversion = q11.samplerYcbcrConversion;
      e11.storageBuffer16BitAccess = q11.storageBuffer16BitAccess;

      dev_chain.get<vk::PhysicalDevice8BitStorageFeatures>().storageBuffer8BitAccess = q8.storageBuffer8BitAccess;
      auto &esg = dev_chain.get<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      esg.subgroupSizeControl = qsg.subgroupSizeControl;
      esg.computeFullSubgroups = qsg.computeFullSubgroups;
      dev_chain.get<vk::PhysicalDeviceShaderFloat16Int8Features>().shaderFloat16 = qf16.shaderFloat16;
      dev_chain.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>().timelineSemaphore = qts.timelineSemaphore;
      dev_chain.get<vk::PhysicalDeviceSynchronization2Features>().synchronization2 = qsync.synchronization2;
      dev_chain.get<vk::PhysicalDevicePresentIdFeaturesKHR>().presentId = qpid.presentId;
      dev_chain.get<vk::PhysicalDevicePresentWaitFeaturesKHR>().presentWait = qpw.presentWait;

      // Link each feature struct based on whether the FEATURE is supported (on
      // 1.2/1.3 devices these are core and not enumerable as extensions, but the
      // structs/queries stay valid). Extension NAMES were only enabled if enumerable.
      if (!q8.storageBuffer8BitAccess)
        dev_chain.unlink<vk::PhysicalDevice8BitStorageFeatures>();
      if (!(qsg.subgroupSizeControl && qsg.computeFullSubgroups))
        dev_chain.unlink<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
      if (!qf16.shaderFloat16)
        dev_chain.unlink<vk::PhysicalDeviceShaderFloat16Int8Features>();
      if (!qts.timelineSemaphore)
        dev_chain.unlink<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
      if (!qsync.synchronization2)
        dev_chain.unlink<vk::PhysicalDeviceSynchronization2Features>();
      if (!self->caps_.present_wait) {
        dev_chain.unlink<vk::PhysicalDevicePresentIdFeaturesKHR>();
        dev_chain.unlink<vk::PhysicalDevicePresentWaitFeaturesKHR>();
      }

      self->dev = vk::raii::Device(self->phys_dev, dev_chain.get<vk::DeviceCreateInfo>());
      self->compute_queue = vk::raii::Queue(self->dev, cqf, 0);

      // TODO(pyrowave): VMA needs explicit function pointers under the dynamic
      // dispatcher; verify these are correct on the target.
      VmaVulkanFunctions vma_fns {};
      vma_fns.vkGetInstanceProcAddr = self->ctx.getDispatcher()->vkGetInstanceProcAddr;
      vma_fns.vkGetDeviceProcAddr = self->dev.getDispatcher()->vkGetDeviceProcAddr;

      VmaAllocatorCreateInfo aci {};
      aci.physicalDevice = *self->phys_dev;
      aci.device = *self->dev;
      aci.instance = *self->inst;
      // VMA uses the version the DEVICE implements (instance may be 1.3 for the
      // codec's property query while the physical device is only 1.1; using 1.3
      // here would make VMA load 1.3-core functions the device lacks and assert).
      aci.vulkanApiVersion = std::min(api_version, self->phys_dev.getProperties().apiVersion);
      aci.pVulkanFunctions = &vma_fns;
      self->allocator.emplace(aci, /*has_debug_utils=*/false);

      SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: Vulkan context ready (fp16=%d present_wait=%d)",
                  (int) self->caps_.shader_float16, (int) self->caps_.present_wait);
      return self;
    } catch (const std::exception &e) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "pyrowave: failed to create Vulkan context: %s", e.what());
      return nullptr;
    }
  }

  context::~context() {
    allocator.reset();
  }

}  // namespace pyrowave_vk
