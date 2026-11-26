// SPDX-License-Identifier: Apache-2.0
// Vulkan eye textures (a native Linux game in the VM, e.g. Half-Life: Alyx's ARM64 build) to the host's VR path.
//
// The host composes DXMT textures: Metal textures that live on the Mac. A Vulkan game's images live in Venus
// instead, which the host's VR exporter cannot see. So each eye goes through a linear Metal texture this runtime
// creates itself over the wmtr client (the same winemetal calls DXMT makes): MTLDevice_newBuffer in shared storage
// is backed by wmtr arena memory that both the guest and the Mac map, and MTLBuffer_newTexture makes a texture
// over it. A submit copies the game's image into a host-visible Vulkan buffer on the game's queue, waits, copies
// the rows into the arena and submits that Metal texture the DXMT way.
//
// First version: a CPU copy per frame and a fence wait on the game's thread. Simple, not fast.
#define VK_NO_PROTOTYPES
#include "vulkan_bridge.h"

#if !__has_include(<vulkan/vulkan.h>)
// built without the Vulkan headers: Vulkan submits are refused, nothing else changes
#include "log.h"
namespace svr {
void vulkan_bridge_init(wmtr_unix_call *) {}
const svr_metal_texture *vulkan_bridge_convert(int, const vr::Texture_t *, bool, const vr::VRTextureBounds_t *,
                                               uint32_t *, uint32_t *) {
  SVR_ERR("vulkan bridge: this runtime was built without the Vulkan headers");
  return nullptr;
}
}  // namespace svr
#else

#include <dlfcn.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <mutex>

#include "log.h"

namespace svr {
namespace {
// winemetal unix calls (wmtr/src/wmtr_calls.inc) and their params (DXMT winemetal_thunks.h, pack(8))
enum : unsigned {
  kNSObjectRelease = 1,
  kNSArrayObject = 2,
  kCopyAllDevices = 4,
  kDeviceNewBuffer = 18,
  kBufferNewTexture = 22,
  kMinLinearAlignment = 24,
};
struct ObjRet { uint64_t ret; };
struct ObjU64ObjRet { uint64_t handle; uint64_t arg; uint64_t ret; };
struct ObjU64U64Ret { uint64_t handle; uint64_t arg; uint64_t ret; };
struct BufferInfo { uint64_t length; uint64_t options; void *memory; uint64_t gpu_address; };
struct NewBuffer { uint64_t device; BufferInfo *info; uint64_t ret; };
struct TextureInfo {
  uint32_t pixel_format, width, height, depth, array_length;
  uint32_t type : 8, mipmap_level_count : 8, sample_count : 8, usage : 8;
  uint64_t options;
  uint32_t reserved;
  uint32_t mach_port;
  uint64_t gpu_resource_id;
};
static_assert(sizeof(TextureInfo) == 48, "WMTTextureInfo");
struct BufferNewTexture { uint64_t buffer; TextureInfo *info; uint64_t offset, bytes_per_row; uint64_t ret; };

struct Format { uint32_t vk, mtl, dxgi, bpp; };
// VkFormat -> MTLPixelFormat, DXGI_FORMAT (what the host's VR exporter reads for sRGB handling), bytes per pixel
constexpr Format kFormats[] = {
    {37, 70, 28, 4},   // R8G8B8A8_UNORM -> RGBA8Unorm
    {43, 71, 29, 4},   // R8G8B8A8_SRGB -> RGBA8Unorm_sRGB
    {44, 80, 87, 4},   // B8G8R8A8_UNORM -> BGRA8Unorm
    {50, 81, 91, 4},   // B8G8R8A8_SRGB -> BGRA8Unorm_sRGB
    {64, 90, 24, 4},   // A2B10G10R10_UNORM_PACK32 -> RGB10A2Unorm
    {122, 92, 26, 4},  // B10G11R11_UFLOAT_PACK32 -> RG11B10Float
    {97, 115, 10, 8},  // R16G16B16A16_SFLOAT -> RGBA16Float
};
constexpr int kRing = 3;  // Metal images per eye: the host may still read one while the next is written

struct Vk {
  VkDevice dev = nullptr;
  PFN_vkGetDeviceProcAddr gdpa = nullptr;
  PFN_vkCreateCommandPool CreateCommandPool;
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
  PFN_vkBeginCommandBuffer BeginCommandBuffer;
  PFN_vkEndCommandBuffer EndCommandBuffer;
  PFN_vkResetCommandBuffer ResetCommandBuffer;
  PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
  PFN_vkQueueSubmit QueueSubmit;
  PFN_vkCreateFence CreateFence;
  PFN_vkWaitForFences WaitForFences;
  PFN_vkResetFences ResetFences;
  PFN_vkCreateBuffer CreateBuffer;
  PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
  PFN_vkAllocateMemory AllocateMemory;
  PFN_vkBindBufferMemory BindBufferMemory;
  PFN_vkMapMemory MapMemory;
  PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
  PFN_vkDestroyBuffer DestroyBuffer;
  PFN_vkFreeMemory FreeMemory;
  VkCommandPool pool = VK_NULL_HANDLE;
  uint32_t pool_family = ~0u;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
};

// Eye images are allocated once for the biggest region seen (Half-Life: Alyx's auto-fidelity changes the rendered
// size every few seconds, inside one big VkImage) and a frame uses its top-left rw x rh.
struct Eye {
  uint32_t w = 0, h = 0, vkfmt = 0;  // capacity
  const Format *fmt = nullptr;
  uint32_t pitch = 0;
  // Vulkan readback buffer
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *map = nullptr;
  bool coherent = false;
  // Metal images over wmtr arena memory
  uint64_t mtl_buf[kRing] = {}, mtl_tex[kRing] = {};
  void *arena[kRing] = {};
  int next = 0;
};

struct Bridge {
  std::mutex m;
  uint64_t device = 0;  // MTLDevice
  Vk vk;
  Eye eye[2];
  svr_metal_texture desc[2];
  bool failed = false;
};
Bridge g;

const Format *find_format(uint32_t vk) {
  for (const Format &f : kFormats)
    if (f.vk == vk) return &f;
  return nullptr;
}

bool load_vk(VkDevice dev) {
  if (g.vk.dev == dev) return true;
  void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW);
  auto gipa = lib ? (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
  auto gdpa = gipa ? (PFN_vkGetDeviceProcAddr)dlsym(lib, "vkGetDeviceProcAddr") : nullptr;
  if (!gdpa) return false;
  Vk v;
  v.dev = dev;
  v.gdpa = gdpa;
#define L(n)                                                \
  if (!(v.n = (PFN_vk##n)gdpa(dev, "vk" #n))) {             \
    SVR_ERR("vulkan bridge: no vk" #n);                     \
    return false;                                           \
  }
  L(CreateCommandPool) L(AllocateCommandBuffers) L(BeginCommandBuffer) L(EndCommandBuffer) L(ResetCommandBuffer)
  L(CmdCopyImageToBuffer) L(CmdPipelineBarrier) L(QueueSubmit) L(CreateFence) L(WaitForFences) L(ResetFences)
  L(CreateBuffer) L(GetBufferMemoryRequirements) L(AllocateMemory) L(BindBufferMemory) L(MapMemory)
  L(InvalidateMappedMemoryRanges) L(DestroyBuffer) L(FreeMemory)
#undef L
  g.vk = v;
  return true;
}

// one winemetal call through the wmtr client's table
template <typename P>
bool mcall(wmtr_unix_call *table, unsigned idx, P *p) {
  return table[idx]((void *)p) == 0;
}

wmtr_unix_call *g_table = nullptr;

bool ensure_device() {
  if (g.device) return true;
  ObjRet all{};
  if (!mcall(g_table, kCopyAllDevices, &all) || !all.ret) return false;
  ObjU64ObjRet first{all.ret, 0, 0};
  if (!mcall(g_table, kNSArrayObject, &first) || !first.ret) return false;
  g.device = first.ret;  // owned by the array, which is never released (one per process)
  return true;
}

bool ensure_command_buffer(uint32_t family) {
  Vk &v = g.vk;
  if (v.pool != VK_NULL_HANDLE && v.pool_family == family) return true;
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = family;
  if (v.CreateCommandPool(v.dev, &pci, nullptr, &v.pool) != VK_SUCCESS) return false;
  v.pool_family = family;
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = v.pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  if (v.AllocateCommandBuffers(v.dev, &ai, &v.cmd) != VK_SUCCESS) return false;
  // a dispatchable handle the loader did not make: give it the device's dispatch table
  *(void **)v.cmd = *(void **)v.dev;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  return v.CreateFence(v.dev, &fci, nullptr, &v.fence) == VK_SUCCESS;
}

void free_eye(Eye &e) {
  Vk &v = g.vk;
  if (e.buf) v.DestroyBuffer(v.dev, e.buf, nullptr);
  if (e.mem) v.FreeMemory(v.dev, e.mem, nullptr);  // unmaps it
  for (int i = 0; i < kRing; i++)
    for (uint64_t h : {e.mtl_tex[i], e.mtl_buf[i]})
      if (h) {
        uint64_t p = h;
        mcall(g_table, kNSObjectRelease, &p);
      }
  e = Eye{};
}

bool setup_eye(Eye &e, const vr::VRVulkanTextureData_t &t, uint32_t cw, uint32_t ch) {
  const Format *f = find_format(t.m_nFormat);
  if (!f) {
    SVR_ERR("vulkan bridge: VkFormat %u not supported", t.m_nFormat);
    return false;
  }
  free_eye(e);
  e.w = cw, e.h = ch, e.vkfmt = t.m_nFormat, e.fmt = f;
  ObjU64U64Ret align{g.device, f->mtl, 0};
  uint64_t a = mcall(g_table, kMinLinearAlignment, &align) && align.ret ? align.ret : 256;
  e.pitch = (uint32_t)((e.w * f->bpp + a - 1) / a * a);
  uint64_t size = (uint64_t)e.pitch * e.h;
  // Vulkan side: a host-visible buffer the image is copied into
  Vk &v = g.vk;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (v.CreateBuffer(v.dev, &bci, nullptr, &e.buf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  v.GetBufferMemoryRequirements(v.dev, e.buf, &req);
  auto gipa = (PFN_vkGetInstanceProcAddr)dlsym(dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD), "vkGetInstanceProcAddr");
  auto props_fn = (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(t.m_pInstance, "vkGetPhysicalDeviceMemoryProperties");
  VkPhysicalDeviceMemoryProperties mp;
  props_fn(t.m_pPhysicalDevice, &mp);
  uint32_t type = ~0u;
  for (int pass = 0; pass < 2 && type == ~0u; pass++)  // cached first (CPU reads it), then any host-visible
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | (pass ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
        type = i;
        break;
      }
    }
  if (type == ~0u) return false;
  e.coherent = mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  if (v.AllocateMemory(v.dev, &mai, nullptr, &e.mem) != VK_SUCCESS) return false;
  if (v.BindBufferMemory(v.dev, e.buf, e.mem, 0) != VK_SUCCESS) return false;
  if (v.MapMemory(v.dev, e.mem, 0, VK_WHOLE_SIZE, 0, &e.map) != VK_SUCCESS) return false;
  // Metal side: linear textures over shared buffers in wmtr arena memory
  for (int i = 0; i < kRing; i++) {
    BufferInfo bi{size, 0 /* shared storage, default cache */, nullptr, 0};
    NewBuffer nb{g.device, &bi, 0};
    if (!mcall(g_table, kDeviceNewBuffer, &nb) || !nb.ret || !bi.memory) return false;
    e.mtl_buf[i] = nb.ret, e.arena[i] = bi.memory;
    TextureInfo ti{};
    ti.pixel_format = f->mtl, ti.width = e.w, ti.height = e.h, ti.depth = 1, ti.array_length = 1;
    ti.type = 2 /* 2D */, ti.mipmap_level_count = 1, ti.sample_count = 1, ti.usage = 1 /* ShaderRead */;
    BufferNewTexture bt{nb.ret, &ti, 0, e.pitch, 0};
    if (!mcall(g_table, kBufferNewTexture, &bt) || !bt.ret) return false;
    e.mtl_tex[i] = bt.ret;
  }
  SVR_INFO("vulkan bridge: eye images %ux%u VkFormat %u -> MTLPixelFormat %u, pitch %u, %d buffers", e.w, e.h,
           e.vkfmt, f->mtl, e.pitch, kRing);
  return true;
}
}  // namespace

void vulkan_bridge_init(wmtr_unix_call *table) {
  std::lock_guard<std::mutex> l(g.m);
  g_table = table;
}

const svr_metal_texture *vulkan_bridge_convert(int eye, const vr::Texture_t *tex, bool array_data,
                                               const vr::VRTextureBounds_t *bounds, uint32_t *rw_out,
                                               uint32_t *rh_out) {
  std::lock_guard<std::mutex> l(g.m);
  if (!g_table || g.failed) return nullptr;
  const auto &t = *(const vr::VRVulkanTextureData_t *)tex->handle;
  uint32_t layer = array_data ? ((const vr::VRVulkanTextureArrayData_t *)tex->handle)->m_unArrayIndex : 0;
  if (t.m_nSampleCount > 1) {
    SVR_ERR("vulkan bridge: multisampled eye textures are not supported");
    return nullptr;
  }
  // the region the game drew: bounds are relative to m_nWidth x m_nHeight and may go past 1 (Alyx at low
  // fidelity: an 864x960 texture with bounds 1.222 x 1.2, a 1056x1152 render in a bigger image)
  float u0 = 0, v0 = 0, u1 = 1, v1 = 1;
  if (bounds) u0 = std::min(bounds->uMin, bounds->uMax), u1 = std::max(bounds->uMin, bounds->uMax),
              v0 = std::min(bounds->vMin, bounds->vMax), v1 = std::max(bounds->vMin, bounds->vMax);
  int32_t x0 = (int32_t)lroundf(u0 * t.m_nWidth), y0 = (int32_t)lroundf(v0 * t.m_nHeight);
  uint32_t rw = (uint32_t)lroundf((u1 - u0) * t.m_nWidth), rh = (uint32_t)lroundf((v1 - v0) * t.m_nHeight);
  if (x0 < 0 || y0 < 0 || !rw || !rh || rw > 8192 || rh > 8192) {
    SVR_ERR("vulkan bridge: bad bounds %.3f..%.3f x %.3f..%.3f", u0, u1, v0, v1);
    return nullptr;
  }
  Eye &e = g.eye[eye];
  bool grow = e.vkfmt != t.m_nFormat || rw > e.w || rh > e.h;
  if (!load_vk(t.m_pDevice) || !ensure_device() || !ensure_command_buffer(t.m_nQueueFamilyIndex) ||
      (grow && !setup_eye(e, t, std::max(rw, e.vkfmt == t.m_nFormat ? e.w : 0u), std::max(rh, e.vkfmt == t.m_nFormat ? e.h : 0u)))) {
    SVR_ERR("vulkan bridge: setup failed; Vulkan frames will not be shown");
    g.failed = true;
    return nullptr;
  }
  Vk &v = g.vk;
  // the image is in TRANSFER_SRC_OPTIMAL at Submit (OpenVR's Vulkan contract)
  v.ResetCommandBuffer(v.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  v.BeginCommandBuffer(v.cmd, &bi);
  VkBufferImageCopy r{};
  r.bufferRowLength = e.pitch / e.fmt->bpp;
  r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
  r.imageOffset = {x0, y0, 0};
  r.imageExtent = {rw, rh, 1};
  v.CmdCopyImageToBuffer(v.cmd, (VkImage)t.m_nImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, e.buf, 1, &r);
  VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = e.buf, b.size = VK_WHOLE_SIZE;
  v.CmdPipelineBarrier(v.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b, 0,
                       nullptr);
  v.EndCommandBuffer(v.cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &v.cmd;
  v.ResetFences(v.dev, 1, &v.fence);
  if (v.QueueSubmit(t.m_pQueue, 1, &si, v.fence) != VK_SUCCESS ||
      v.WaitForFences(v.dev, 1, &v.fence, VK_TRUE, 2000000000ull) != VK_SUCCESS) {
    SVR_ERR("vulkan bridge: copy of the eye image failed");
    return nullptr;
  }
  if (!e.coherent) {
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    mr.memory = e.mem, mr.size = VK_WHOLE_SIZE;
    v.InvalidateMappedMemoryRanges(v.dev, 1, &mr);
  }
  int i = e.next;
  e.next = (e.next + 1) % kRing;
  memcpy(e.arena[i], e.map, (size_t)e.pitch * rh);
  *rw_out = rw, *rh_out = rh;
  svr_metal_texture &d = g.desc[eye];
  memset(&d, 0, sizeof d);
  d.magic = SVR_METAL_TEXTURE_MAGIC, d.size = sizeof d;
  d.handle = e.mtl_tex[i];
  d.width = e.w, d.height = e.h, d.dxgi_format = e.fmt->dxgi;
  d.array_size = 1, d.sample_count = 1;
  return &d;
}
}  // namespace svr
#endif
