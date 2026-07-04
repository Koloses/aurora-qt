# Aurora

Aurora is a fork of [Moonlight PC](https://github.com/moonlight-stream/moonlight-qt) — the open
source PC client for [Sunshine](https://github.com/LizardByte/Sunshine) and NVIDIA GameStream —
extended with support for **PyroWave**, a GPU-only intra-frame wavelet codec decoded entirely in
Vulkan compute. Paired with a [Solarflare](https://github.com/Koloses/Solarflare) host, PyroWave
delivers very low, fixed-latency streaming without touching the hardware video decoder at all.

Everything Moonlight PC does still works: Aurora remains fully compatible with stock Sunshine and
GameStream hosts using H.264, HEVC, and AV1, on Windows, macOS, and Linux.

## What the fork adds

- **PyroWave decoder** (vendored from the [WiVRn](https://github.com/WiVRn/WiVRn) fork of
  [Hans-Kristian Arntzen's PyroWave](https://github.com/Themaister/pyrowave)): Vulkan compute
  decode with a pipelined presentation path, direct-present when the swapchain allows it, and
  HDR10 (ST 2084) plus 4:4:4 chroma support.
- **Loss resilience**: partial-frame salvage and per-packet parser resynchronization, so packet
  loss costs a few blocks for a moment instead of whole frames.
- **Self-initializing decode state**: the stream converges via the host's rolling refresh, with no
  dependency on catching the first frame.
- **Phase-locked pacing**: a present-wait feedback thread reports scanout timing to the host,
  which locks frame delivery to the display without adding latency.
- **New settings**: force the PyroWave codec, and opt into adaptive FEC and adaptive bitrate.
- **Rebranding**: Aurora name and artwork; upstream update checker disabled. Settings and pairings
  remain compatible with previous Moonlight installs.

PyroWave requires a Solarflare host; there is also an
[Aurora client for Android](https://github.com/Koloses/aurora-android).

## Building

Build like upstream Moonlight PC (Qt 6, `qmake`, platform toolchains). The PyroWave decoder
additionally needs a Vulkan SDK, with `python3` and `glslangValidator` available for shader
generation.

```bash
git clone --recursive https://github.com/Koloses/aurora-qt
cd aurora-qt
qmake6 moonlight-qt.pro
make release
```

On Windows, use `scripts\build-arch.bat release` from a Qt + VS 2022 environment.

## Upstream and credits

- [Moonlight PC](https://github.com/moonlight-stream/moonlight-qt) and the
  [Moonlight project](https://moonlight-stream.org) — the client this fork is based on.
- [PyroWave](https://github.com/Themaister/pyrowave) by Hans-Kristian Arntzen — the codec design
  and reference implementation.
- [WiVRn](https://github.com/WiVRn/WiVRn) — the PyroWave fork vendored here.
- [Sunshine](https://github.com/LizardByte/Sunshine) by LizardByte — the host Solarflare is
  forked from.

## License

GPL-3.0, same as upstream Moonlight. Original copyright and attribution notices are retained.
