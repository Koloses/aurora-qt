# Vendored PyroWave GPU wavelet codec - qmake include file for moonlight-qt.
#
# Usage (from app/app.pro or a decoder .pri):
#   include($$PWD/../pyrowave/pyrowave.pri)
#
# Build requirements:
#   * C++20 (CONFIG += c++20 in the including project)
#   * Vulkan headers reachable on the include path
#   * Python 3 and glslangValidator on PATH (or set PYROWAVE_GLSLANG / PYTHON)
#
# Integration requirement (see README.md): the host MUST construct exactly one
# `vk_allocator` (VMA) singleton before creating PyroWave::Encoder/Decoder.

PYROWAVE_ROOT = $$PWD

INCLUDEPATH += \
    $$PYROWAVE_ROOT/src \
    $$PYROWAVE_ROOT/src/pyrowave \
    $$PYROWAVE_ROOT/external

DEFINES += VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

# MSVC defaults to a non-conformant preprocessor that rejects __VA_OPT__ (used by
# the CHECK_VK macro in src/vk/check.h). Opt into the standards-conforming one.
# Also silence the CRT "unsafe" warning for std::getenv used in pyrowave_common.cpp.
win32-msvc {
    QMAKE_CFLAGS   += /Zc:preprocessor
    QMAKE_CXXFLAGS += /Zc:preprocessor
    DEFINES += _CRT_SECURE_NO_WARNINGS

    # Match the app's Control Flow Guard / EH-continuation hardening (set in
    # globaldefs.pri, which this lib does not include). Without /guard:ehcont the
    # codec objects lack EHCont metadata and the final Moonlight.exe link fails
    # with LNK2047/LNK1386.
    QMAKE_CFLAGS   += -guard:cf -guard:ehcont
    QMAKE_CXXFLAGS += -guard:cf -guard:ehcont
}

# VMA defaults to STATIC Vulkan functions, which reference vkAllocateMemory etc. as
# linked symbols and require vulkan-1.lib at link time. The Vulkan context instead
# feeds VMA dynamic function pointers (vkGetInstanceProcAddr/vkGetDeviceProcAddr), so
# load everything dynamically and avoid the import-library dependency. Mirrors the
# Android build (Android.mk uses the same two defines).
win32 {
    DEFINES += VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1
}

HEADERS += \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_common.h \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_encoder.h \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_decoder.h \
    $$PYROWAVE_ROOT/src/vk/allocation.h \
    $$PYROWAVE_ROOT/src/vk/vk_allocator.h \
    $$PYROWAVE_ROOT/src/vk/check.h \
    $$PYROWAVE_ROOT/src/utils/singleton.h

SOURCES += \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_common.cpp \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_encoder.cpp \
    $$PYROWAVE_ROOT/src/pyrowave/pyrowave_decoder.cpp \
    $$PYROWAVE_ROOT/src/vk/allocation.cpp \
    $$PYROWAVE_ROOT/src/vk/vk_allocator.cpp \
    $$PYROWAVE_ROOT/src/vk/error_category.cpp

# The VMA implementation translation unit. If the host application already
# links a VMA implementation (e.g. via libplacebo) and you hit duplicate-symbol
# errors at link time, define PYROWAVE_NO_VMA_IMPL to drop this TU.
!contains(DEFINES, PYROWAVE_NO_VMA_IMPL) {
    SOURCES += $$PYROWAVE_ROOT/src/vk/vk_mem_alloc.cpp
}

# --- Shader generation (GLSL -> SPIR-V -> C++ table) -------------------------
# Generated at QMAKE TIME (not as a parallel build step). The generator writes
# both pyrowave_shaders.cpp (the tracked output) and pyrowave_shaders.h. As a
# build-time extra-compiler the .h was an untracked side-product, so under jom's
# parallel jobs a cl process compiling a TU that #includes it could start before
# the generator ran -> "Cannot open include file: 'pyrowave_shaders.h'". Running
# at configure time guarantees both files exist on disk before any compile.
# (Re-run qmake after editing shaders/ to regenerate.)
PYROWAVE_GEN_DIR = $$OUT_PWD/pyrowave_generated
INCLUDEPATH += $$PYROWAVE_GEN_DIR

isEmpty(PYTHON): PYTHON = python3
isEmpty(PYROWAVE_GLSLANG): PYROWAVE_GLSLANG = glslangValidator

PYROWAVE_GEN_CMD = $$PYTHON $$PYROWAVE_ROOT/tools/generate_shaders.py \
    --shader-dir $$PYROWAVE_ROOT/shaders \
    --output-dir $$PYROWAVE_GEN_DIR \
    --glslang $$PYROWAVE_GLSLANG \
    --target-env vulkan1.1 \
    --namespace pyrowave
# Run it now (qmake time). $$system() discards the script's stderr progress lines.
PYROWAVE_GEN_LOG = $$system($$PYROWAVE_GEN_CMD)
!exists($$PYROWAVE_GEN_DIR/pyrowave_shaders.cpp) {
    error("PyroWave shader generation failed - ensure Python 3 and glslangValidator are on PATH. Tried: $$PYROWAVE_GEN_CMD")
}

SOURCES += $$PYROWAVE_GEN_DIR/pyrowave_shaders.cpp
