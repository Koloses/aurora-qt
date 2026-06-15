# Vendored PyroWave GPU wavelet codec - static library subproject for moonlight-qt.
#
# Built as a separate static lib (rather than injected into app.pro) so it can
# use C++20 and isolate its VMA implementation TU from the rest of the app.
# Enabled from the top-level project via `CONFIG += pyrowave`.

TEMPLATE = lib
CONFIG += staticlib c++20
CONFIG -= qt          # pure C++/Vulkan, no Qt dependency
TARGET = pyrowave

# CRITICAL (NDEBUG consistency): vk::raii / vulkan.hpp class layout and the
# DeviceDispatcher asserts are gated on NDEBUG. This codec shares vk::raii
# objects (vk::raii::Device and its dispatcher) with the app across the static
# library boundary, so it MUST be compiled with the SAME NDEBUG state as the
# app -- otherwise the dispatcher is misread at runtime and
# `getDispatcher()` aborts on `getVkHeaderVersion() == VK_HEADER_VERSION`
# (decoder init crash). The app is shipped/built as a release (NDEBUG) build,
# but this subproject otherwise builds in debug mode when driven from the
# top-level debug_and_release project, so force NDEBUG here to keep the codec on
# the same (release) vk::raii layout as the app. (Mirrors Sunshine, which sets
# NDEBUG on its vendored pyrowave target for the same reason.)
DEFINES += NDEBUG

# Vulkan headers (vulkan_raii.hpp, vk_mem_alloc.h needs vulkan/vulkan.h).
# Prefer the Vulkan SDK when present; otherwise rely on system headers.
VULKAN_SDK_ENV = $$(VULKAN_SDK)
!isEmpty(VULKAN_SDK_ENV) {
    win32: INCLUDEPATH += $$VULKAN_SDK_ENV/Include
    else:  INCLUDEPATH += $$VULKAN_SDK_ENV/include
}

include($$PWD/pyrowave.pri)
