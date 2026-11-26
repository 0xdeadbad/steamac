// SPDX-License-Identifier: Apache-2.0
// Vulkan eye textures to the host's VR path through linear Metal textures over wmtr arena memory (vulkan_bridge.cpp).
#pragma once
#include <stdint.h>

#include "openvr.h"
#include "steamac_vr_guest.h"

namespace svr {
typedef uint32_t (*wmtr_unix_call)(void *params);  // a winemetal unix call (the wmtr client's table entry)

// the wmtr client's winemetal call table; without it Vulkan submits are refused
void vulkan_bridge_init(wmtr_unix_call *table);
// copies the region `bounds` selects of the eye's Vulkan image into the top-left rw x rh of a Metal texture the host
// can show; nullptr on failure
const svr_metal_texture *vulkan_bridge_convert(int eye, const vr::Texture_t *tex, bool array_data,
                                               const vr::VRTextureBounds_t *bounds, uint32_t *rw, uint32_t *rh);
}  // namespace svr
