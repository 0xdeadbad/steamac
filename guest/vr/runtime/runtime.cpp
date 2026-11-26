// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "log.h"
#include "vulkan_bridge.h"
#include "vrmath.h"

namespace svr {
int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

// how often wait_frame asks again for a pacer frame that is due but not out yet (the host sends it at its wake-up;
// it reaches the exporter a few microseconds later)
static constexpr int64_t kPacerPollNs = 250000;
// the longest wait_frame waits for one pacer frame (a backstop: the exporter stops flagging PACER_LIVE first)
static constexpr int64_t kPacerWaitMaxNs = 200000000;

static void sleep_until(int64_t t) {
  timespec ts{(time_t)(t / 1000000000), (long)(t % 1000000000)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
}

Runtime &Runtime::get() {
  static Runtime *rt = new Runtime();  // never destroyed: the app may call into it from atexit handlers
  return *rt;
}

vr::EVRInitError Runtime::init(vr::EVRApplicationType type) {
  std::lock_guard<std::mutex> g(m_);
  if (transport_) return vr::VRInitError_None;
  transport_ = Transport::open_default();
  if (const char *e = getenv("STEAMAC_VR_RECENTER"); e && *e == '1') recenter_wanted_ = true;
  if (!transport_) {
    SVR_ERR("no host link (DXMT/wmtr not loaded in this process and STEAMAC_VR_SOCKET not set)");
    return vr::VRInitError_Init_HmdNotFound;
  }
  svr_hello_req req{};
  req.h = {SVR_HELLO, sizeof(req)};
  req.version = SVR_VERSION;
  req.pid = (uint32_t)getpid();
  strncpy(req.app_key, app_key.c_str(), sizeof(req.app_key) - 1);
  svr_hello_rep rep{};
  int r = transport_->call(&req.h, &rep.h, sizeof(rep));
  if (r || rep.h.op != SVR_HELLO || rep.version != SVR_VERSION) {
    SVR_ERR("host hello failed (%d, version %u)", r, rep.version);
    transport_.reset();
    return vr::VRInitError_Init_HmdNotFound;
  }
  hello_ = rep;
  // hosts since vr-polish send a view also without a headset (the last one's, or a Quest 3's): use it at once
  hello_view_ = rep.eye_width && rep.fov[0][1] - rep.fov[0][0] > 0;
  SVR_INFO("init app type %d over %s: %s %ux%u per eye %.0f Hz connected=%u", (int)type, transport_->name(),
           rep.name, rep.eye_width, rep.eye_height, rep.refresh_hz, rep.connected);
  if (!connected_[0]) {  // the HMD is present from VR_Init on (standby while no headset is connected)
    connected_[0] = true;
    vr::VREvent_t ev{};
    ev.eventType = vr::VREvent_TrackedDeviceActivated;
    ev.trackedDeviceIndex = 0;
    events_.push_back(ev);
  }
  return vr::VRInitError_None;
}

void Runtime::shutdown() {
  std::lock_guard<std::mutex> g(m_);
  if (pending_valid_) flush_submit_locked();
  transport_.reset();
  events_.clear();
  memset(connected_, 0, sizeof(connected_));
}

svr_hello_rep Runtime::hello() const {
  std::lock_guard<std::mutex> g(m_);
  return hello_;
}

float Runtime::refresh_hz() const {
  std::lock_guard<std::mutex> g(m_);
  return latest_.refresh_hz > 0 ? latest_.refresh_hz : hello_.refresh_hz > 0 ? hello_.refresh_hz : 90.0f;
}

// caller holds no lock
bool Runtime::fetch(svr_frame_rep *out) {
  svr_frame_req req{};
  req.h = {SVR_FRAME, sizeof(req)};
  bool need_view;
  {
    std::lock_guard<std::mutex> g(m_);
    if (!transport_) return false;
    req.last_frame_id = frame_.frame_id;
    need_view = !hello_view_;
  }
  svr_frame_rep rep{};
  int r = transport_->call(&req.h, &rep.h, sizeof(rep));
  int64_t now = now_ns();
  if (r || rep.h.op != SVR_FRAME || rep.h.size != sizeof(rep)) {
    SVR_ERR("frame request failed (%d)", r);
    return false;
  }
  if (recenter_wanted_) {
    std::lock_guard<std::mutex> g(m_);
    recenter_locked(rep);
  }
  if (rep.connected && need_view) {
    // VR_Init ran before a headset connected: its hello had no eye size, FOV or IPD (GetProjectionRaw would be all
    // zero). Ask again now that the host has the headset's view.
    svr_hello_req hq{};
    hq.h = {SVR_HELLO, sizeof(hq)};
    hq.version = SVR_VERSION;
    hq.pid = (uint32_t)getpid();
    strncpy(hq.app_key, app_key.c_str(), sizeof(hq.app_key) - 1);
    svr_hello_rep hr{};
    if (transport_->call(&hq.h, &hr.h, sizeof(hr)) == 0 && hr.h.op == SVR_HELLO && hr.connected && hr.eye_width) {
      std::lock_guard<std::mutex> g(m_);
      hello_ = hr;
      hello_view_ = true;
      SVR_INFO("headset view: %s %ux%u per eye %.0f Hz", hr.name, hr.eye_width, hr.eye_height, hr.refresh_hz);
    }
  }
  std::lock_guard<std::mutex> g(m_);
  clock_offset_ = now - rep.host_now;
  latest_ = rep;
  hello_.connected = rep.connected;
  // device connection changes become OpenVR events. The HMD stays connected while the runtime is up, as SteamVR
  // keeps it in standby when the headset leaves: its pose turns invalid, the game keeps running and rendering.
  bool conn[kDeviceCount] = {true,
                             rep.connected && (rep.pose[SVR_DEV_LEFT_GRIP].flags & SVR_POSE_VALID),
                             rep.connected && (rep.pose[SVR_DEV_RIGHT_GRIP].flags & SVR_POSE_VALID)};
  for (uint32_t i = 0; i < kDeviceCount; i++) {
    if (conn[i] != connected_[i]) {
      connected_[i] = conn[i];
      vr::VREvent_t ev{};
      ev.eventType = conn[i] ? vr::VREvent_TrackedDeviceActivated : vr::VREvent_TrackedDeviceDeactivated;
      ev.trackedDeviceIndex = i;
      events_.push_back(ev);
    }
  }
  if (out) *out = rep;
  return true;
}

bool Runtime::poll() { return fetch(nullptr); }

bool Runtime::wait_frame() {
  {
    std::lock_guard<std::mutex> g(m_);
    if (pending_valid_) flush_submit_locked();
  }
  svr_frame_rep rep;
  if (!fetch(&rep)) return false;
  int64_t wake = host_to_guest(rep.wake_up);
  if (wake > now_ns()) {
    sleep_until(wake);
    if (!fetch(&rep)) return false;  // fresher prediction for the same frame
  }
  // The host's pacer runs but its next frame is not out yet (spacewarp releases one every other display period; the
  // host holds one while the GPU is behind): wait for it instead of rendering a frame the headset never asked for,
  // which only queues GPU work ahead of the frames it does show. Bounded, so a pacer that stops (headset gone) leaves
  // the game on the local cadence; the bound is time, not display periods, since the host holds a timing for a whole
  // GPU frame of a GPU-bound game (exporter: PACER_LIVE ends WMTR_VR_PACER_LIVE_NS = 150 ms after the last timing).
  if ((rep.flags & SVR_FRAME_PACER_LIVE) && !(rep.flags & SVR_FRAME_FROM_PACER)) {
    int64_t period = (int64_t)(1e9f / (rep.refresh_hz > 0 ? rep.refresh_hz : 90.0f));
    int64_t give_up = now_ns() + std::max<int64_t>(4 * period, kPacerWaitMaxNs);
    while ((rep.flags & SVR_FRAME_PACER_LIVE) && !(rep.flags & SVR_FRAME_FROM_PACER) && now_ns() < give_up) {
      sleep_until(std::max(host_to_guest(rep.wake_up), now_ns() + kPacerPollNs));  // when it is due, then polls
      if (!fetch(&rep)) return false;
    }
  }
  std::lock_guard<std::mutex> g(m_);
  if (frame_.frame_id) last_vsync_ = host_to_guest(frame_.predicted_display);
  frame_ = rep;
  const svr_pose &head = rep.pose[SVR_DEV_HEAD];
  if (head.flags != logged_head_flags_) {
    // the render pose the game gets: invalid poses leave the scene head-locked in the headset
    logged_head_flags_ = head.flags;
    SVR_INFO("frame %llu head pose %s%s: orientation %.3f %.3f %.3f %.3f position %.3f %.3f %.3f",
             (unsigned long long)rep.frame_id, (head.flags & SVR_POSE_VALID) ? "valid" : "INVALID",
             (head.flags & SVR_POSE_TRACKED) ? ", tracked" : "", head.orientation[0], head.orientation[1],
             head.orientation[2], head.orientation[3], head.position[0], head.position[1], head.position[2]);
  }
  SVR_TRACE("frame %llu display in %.2f ms", (unsigned long long)rep.frame_id,
            (host_to_guest(rep.predicted_display) - now_ns()) / 1e6);
  return true;
}

void Runtime::note_render_head(const vr::TrackedDevicePose_t &p) {
  int64_t now = now_ns();
  std::lock_guard<std::mutex> g(m_);
  if (!p.bPoseIsValid) stats_head_invalid_++;
  else
    for (int i = 0; i < 3; i++) {
      // head yaw/pitch proxies (forward vector) and height: their range over the window shows whether it moves
      float v = i == 0 ? -p.mDeviceToAbsoluteTracking.m[0][2] : i == 1 ? -p.mDeviceToAbsoluteTracking.m[1][2]
                                                                         : p.mDeviceToAbsoluteTracking.m[1][3];
      if (v < stats_head_min_[i]) stats_head_min_[i] = v;
      if (v > stats_head_max_[i]) stats_head_max_[i] = v;
    }
  if (!stats_logged_) stats_logged_ = now;
  if (now - stats_logged_ < 5000000000LL) return;
  stats_logged_ = now;
  PoseStats &s = pose_stats_;
  SVR_INFO("poses 5s: WaitGetPoses %u (failed %u), GetLastPoses %u, GetLastPoseForTrackedDeviceIndex %u, "
           "GetDeviceToAbsoluteTrackingPose %u (seated %u), pose actions %u (inactive %u), IsTrackedDeviceConnected %u "
           "(HMD false %u); render head invalid %u, forward x %.2f..%.2f y %.2f..%.2f, height %.2f..%.2f m",
           s.wait.exchange(0), s.wait_fail.exchange(0), s.last_poses.exchange(0), s.last_pose_index.exchange(0),
           s.abs_pose.exchange(0), s.abs_seated.exchange(0), s.pose_action.exchange(0), s.pose_action_inactive.exchange(0),
           s.connected_query.exchange(0), s.connected_false_hmd.exchange(0), stats_head_invalid_, stats_head_min_[0],
           stats_head_max_[0], stats_head_min_[1], stats_head_max_[1], stats_head_min_[2], stats_head_max_[2]);
  stats_head_invalid_ = 0;
  for (int i = 0; i < 3; i++) stats_head_min_[i] = 1e9f, stats_head_max_[i] = -1e9f;
}

svr_pose Runtime::frame_pose(int dev) const {
  std::lock_guard<std::mutex> g(m_);
  return frame_.pose[dev];
}

int64_t Runtime::frame_display_guest() const {
  std::lock_guard<std::mutex> g(m_);
  return host_to_guest(frame_.predicted_display);
}

uint64_t Runtime::frame_id() const {
  std::lock_guard<std::mutex> g(m_);
  return frame_.frame_id;
}

svr_pose Runtime::pose_at(int dev, int64_t guest_t) const {
  std::lock_guard<std::mutex> g(m_);
  float dt = (float)(guest_t - host_to_guest(latest_.pose_time)) * 1e-9f;
  return extrapolate(latest_.pose[dev], dt);
}

static int device_to_svr(vr::TrackedDeviceIndex_t i) {
  return i == kHmd ? SVR_DEV_HEAD : i == kLeft ? SVR_DEV_LEFT_GRIP : i == kRight ? SVR_DEV_RIGHT_GRIP : -1;
}

bool Runtime::device_connected(vr::TrackedDeviceIndex_t i) const {
  std::lock_guard<std::mutex> g(m_);
  return i < kDeviceCount && connected_[i];
}

vr::TrackedDevicePose_t Runtime::device_pose(vr::TrackedDeviceIndex_t i, float seconds_from_now) const {
  int d = device_to_svr(i);
  if (d < 0) return to_openvr(svr_pose{}, false);
  return to_openvr(pose_at(d, now_ns() + (int64_t)(seconds_from_now * 1e9f)), device_connected(i));
}

vr::TrackedDevicePose_t Runtime::device_frame_pose(vr::TrackedDeviceIndex_t i) const {
  int d = device_to_svr(i);
  if (d < 0) return to_openvr(svr_pose{}, false);
  return to_openvr(frame_pose(d), device_connected(i));
}

int64_t Runtime::last_vsync_guest() const {
  std::lock_guard<std::mutex> g(m_);
  return last_vsync_;
}

float Runtime::input(int hand, int id) const {
  std::lock_guard<std::mutex> g(m_);
  return latest_.input[hand][id];
}

uint32_t Runtime::profile() const {
  std::lock_guard<std::mutex> g(m_);
  return latest_.profile ? latest_.profile : hello_.profile;
}

// 8-bit UNORM formats hold gamma-encoded data unless the app says the colour space is linear
static bool is_unorm8(uint32_t dxgi) {
  return dxgi == 28 /* R8G8B8A8_UNORM */ || dxgi == 87 /* B8G8R8A8_UNORM */ || dxgi == 88 /* B8G8R8X8_UNORM */;
}

vr::EVRCompositorError Runtime::submit(vr::EVREye eye, const vr::Texture_t *tex, const vr::VRTextureBounds_t *bounds,
                                       vr::EVRSubmitFlags flags, uint32_t array_index) {
  if (eye != vr::Eye_Left && eye != vr::Eye_Right) return vr::VRCompositorError_IndexOutOfRange;
  if (!tex || !tex->handle) return vr::VRCompositorError_InvalidTexture;
  const svr_metal_texture *mt;
  vr::VRTextureBounds_t vulkan_bounds;
  if (tex->eType == vr::TextureType_Vulkan) {
    // a native Vulkan game: copied into a Metal texture the host can show (vulkan_bridge.cpp)
    uint32_t rw = 0, rh = 0;
    mt = vulkan_bridge_convert(eye == vr::Eye_Left ? 0 : 1, tex, flags & vr::Submit_VulkanTextureWithArrayData, bounds,
                               &rw, &rh);
    if (!mt) return vr::VRCompositorError_InvalidTexture;
    // the bridge copied the bounds' region to the top-left of its texture: the game's bounds become that rect,
    // flipped the same way
    bool flipped = bounds && bounds->vMin > bounds->vMax;
    float fu = (float)rw / mt->width, fv = (float)rh / mt->height;
    vulkan_bounds = vr::VRTextureBounds_t{0, flipped ? fv : 0, fu, flipped ? 0 : fv};
    bounds = &vulkan_bounds;
  } else if (tex->eType == vr::TextureType_Metal) {
    mt = (const svr_metal_texture *)tex->handle;
  } else {
    SVR_ERR("unsupported texture type %d (DXMT textures via TextureType_Metal, or Vulkan)", (int)tex->eType);
    return vr::VRCompositorError_TextureUsesUnsupportedFormat;
  }
  if (mt->magic != SVR_METAL_TEXTURE_MAGIC || mt->size < sizeof(*mt) || !mt->handle || !mt->width || !mt->height) {
    SVR_ERR("TextureType_Metal handle is not a steamac DXMT texture description");
    return vr::VRCompositorError_InvalidTexture;
  }
  if (mt->sample_count > 1) return vr::VRCompositorError_TextureUsesUnsupportedFormat;  // must be resolved
  std::lock_guard<std::mutex> g(m_);
  if (!transport_) return vr::VRCompositorError_DoNotHaveFocus;
  if (pending_valid_ && (pending_.frame_id != frame_.frame_id || pending_eye_[eye])) flush_submit_locked();
  if (!pending_valid_) {
    memset(&pending_, 0, sizeof(pending_));
    pending_eye_[0] = pending_eye_[1] = false;
    pending_.frame_id = frame_.frame_id;
    pending_.display_time = (uint64_t)frame_.predicted_display;
    // per-eye render poses: the frame's head pose moved by half the IPD along the head's x axis. FOV and IPD are
    // the ones the game rendered with (GetProjectionRaw, GetEyeToHeadTransform: the hello's view), not the frame's:
    // a game that started on the standby view keeps that projection when the headset connects with its own, and
    // the headset reprojects the image with the FOV sent here (a mismatch stretches the picture).
    const svr_pose &head = frame_.pose[SVR_DEV_HEAD];
    float ipd = hello_.ipd_m > 0 ? hello_.ipd_m : 0.063f;
    Quat q{head.orientation[0], head.orientation[1], head.orientation[2], head.orientation[3]};
    for (int e = 0; e < 2; e++) {
      float off[3] = {(e ? 0.5f : -0.5f) * ipd, 0, 0}, w[3];
      qrot(q, off, w);
      memcpy(pending_.pose[e], head.orientation, 4 * sizeof(float));
      for (int i = 0; i < 3; i++) pending_.pose[e][4 + i] = head.position[i] + w[i];
      memcpy(pending_.fov[e], hello_.fov[e], sizeof(pending_.fov[e]));
      to_host_locked(pending_.pose[e]);
    }
    pending_valid_ = true;
  }
  svr_wmtvr_eye &e = pending_.eye[eye];
  float u0 = 0, v0 = 0, u1 = 1, v1 = 1;
  if (bounds) u0 = bounds->uMin, v0 = bounds->vMin, u1 = bounds->uMax, v1 = bounds->vMax;
  bool flip = v0 > v1;
  if (flip) std::swap(v0, v1);
  e.texture = mt->handle;
  e.x = (uint32_t)lroundf(u0 * mt->width), e.y = (uint32_t)lroundf(v0 * mt->height);
  e.w = (uint32_t)lroundf((u1 - u0) * mt->width), e.h = (uint32_t)lroundf((v1 - v0) * mt->height);
  if (e.x == 0 && e.y == 0 && e.w == mt->width && e.h == mt->height) e.w = e.h = 0;  // whole texture
  e.flags = (flip ? SVR_EYE_FLIP_Y : 0) |
            (is_unorm8(mt->dxgi_format) && tex->eColorSpace != vr::ColorSpace_Linear ? SVR_EYE_SRGB : 0);
  // Single-pass stereo uses an array texture with one eye per slice. Proton stores an explicit slice in the
  // description for SubmitWithArrayIndex; otherwise array textures default to one slice per eye.
  e.slice = array_index != ~0u ? array_index : mt->array_size > 1 ? mt->array_index : 0;
  SubmitGeometry geo{mt->width, mt->height, mt->dxgi_format, mt->array_size, e.slice, e.x, e.y, e.w, e.h, e.flags};
  if (memcmp(&geo, &logged_geometry_[eye], sizeof geo)) {
    // what the game submits, once per change: whole texture, a bounds rect (double-wide) or an array slice
    logged_geometry_[eye] = geo;
    SVR_INFO("submit %s eye: texture %ux%u dxgi %u array %u slice %u, bounds u %.3f..%.3f v %.3f..%.3f -> rect %u,%u %ux%u%s, "
             "flags %#x%s%s",
             eye == vr::Eye_Left ? "left" : "right", mt->width, mt->height, mt->dxgi_format, mt->array_size, e.slice,
             bounds ? bounds->uMin : 0.f, bounds ? bounds->uMax : 1.f, bounds ? bounds->vMin : 0.f,
             bounds ? bounds->vMax : 1.f, e.x, e.y, e.w ? e.w : mt->width, e.w ? e.h : mt->height,
             e.w ? "" : " (whole)", (unsigned)flags, flip ? ", bottom-up" : "", (e.flags & SVR_EYE_SRGB) ? ", sRGB" : "");
  }
  if (SVR_NPT_IS_HANDLE(mt->handle)) {
    svr_npt_eye &n = pending_npt_[eye];
    n.width = mt->width, n.height = mt->height, n.dxgi_format = mt->dxgi_format;
    n.slices = mt->array_size ? mt->array_size : 1;
    n.pitch = SVR_NPT_HANDLE_PITCH(mt->handle);
  }
  if (mt->event_handle && mt->event_value >= pending_.wait_value) {
    pending_.wait_event = mt->event_handle;
    pending_.wait_value = mt->event_value;
  }
  pending_eye_[eye] = true;
  if (pending_eye_[0] && pending_eye_[1]) flush_submit_locked();
  return vr::VRCompositorError_None;
}

void Runtime::recenter_locked(svr_frame_rep &rep) {
  const svr_pose &head = rep.pose[SVR_DEV_HEAD];
  if (!recentered_) {
    if (!rep.connected || (head.flags & (SVR_POSE_VALID | SVR_POSE_TRACKED)) != (SVR_POSE_VALID | SVR_POSE_TRACKED))
      return;
    Quat q{head.orientation[0], head.orientation[1], head.orientation[2], head.orientation[3]};
    const float fwd[3] = {0, 0, -1};
    float f[3];
    qrot(q, fwd, f);
    float yaw = std::atan2(-f[0], -f[2]);  // rotation about +y that takes -z to the head's horizontal forward
    recenter_q_ = Quat{0, std::sin(-yaw / 2), 0, std::cos(-yaw / 2)};
    recenter_c_[0] = head.position[0], recenter_c_[1] = 0, recenter_c_[2] = head.position[2];
    recentered_ = true;
    SVR_INFO("recentered: origin at host %.2f, %.2f m, yaw %.0f degrees", recenter_c_[0], recenter_c_[2],
             yaw * 57.29578f);
  }
  for (svr_pose &p : rep.pose) {
    float d[3] = {p.position[0] - recenter_c_[0], p.position[1], p.position[2] - recenter_c_[2]}, v[3];
    qrot(recenter_q_, d, p.position);
    Quat o = qmul(recenter_q_, Quat{p.orientation[0], p.orientation[1], p.orientation[2], p.orientation[3]});
    p.orientation[0] = o.x, p.orientation[1] = o.y, p.orientation[2] = o.z, p.orientation[3] = o.w;
    memcpy(v, p.lin_vel, sizeof v);
    qrot(recenter_q_, v, p.lin_vel);
    memcpy(v, p.ang_vel, sizeof v);
    qrot(recenter_q_, v, p.ang_vel);
  }
}

void Runtime::to_host_locked(float pose[7]) const {
  if (!recentered_) return;
  Quat inv{-recenter_q_.x, -recenter_q_.y, -recenter_q_.z, recenter_q_.w};
  Quat o = qmul(inv, Quat{pose[0], pose[1], pose[2], pose[3]});
  pose[0] = o.x, pose[1] = o.y, pose[2] = o.z, pose[3] = o.w;
  float p[3];
  qrot(inv, pose + 4, p);
  pose[4] = p[0] + recenter_c_[0], pose[5] = p[1] + recenter_c_[1], pose[6] = p[2] + recenter_c_[2];
}

void Runtime::flush_submit_locked() {
  pending_valid_ = false;
  if (!pending_eye_[0] || !pending_eye_[1]) {
    SVR_TRACE("frame %llu dropped: only one eye submitted", (unsigned long long)pending_.frame_id);
    return;
  }
  int r;
  if (!transport_) r = -ENOTCONN;
  else if (SVR_NPT_IS_HANDLE(pending_.eye[0].texture) && SVR_NPT_IS_HANDLE(pending_.eye[1].texture))
    r = transport_->submit_npt(pending_, pending_npt_);
  else r = transport_->submit(pending_);
  if (r) SVR_ERR("submit of frame %llu failed (%d)", (unsigned long long)pending_.frame_id, r);
  else frames_presented_++;
}

void Runtime::post_present_handoff() {
  std::lock_guard<std::mutex> g(m_);
  if (pending_valid_) flush_submit_locked();
}

void Runtime::clear_frame() {
  std::lock_guard<std::mutex> g(m_);
  pending_valid_ = false;
}

void Runtime::haptic(int hand, float duration_s, float frequency, float amplitude) {
  svr_haptic_req req{};
  req.h = {SVR_HAPTIC, sizeof(req)};
  req.hand = (uint32_t)hand;
  req.duration_s = duration_s, req.frequency = frequency, req.amplitude = amplitude;
  svr_ack ack{};
  std::lock_guard<std::mutex> g(m_);
  if (transport_) transport_->call(&req.h, &ack.h, sizeof(ack));
}

void Runtime::push_event(vr::EVREventType type, vr::TrackedDeviceIndex_t dev) {
  std::lock_guard<std::mutex> g(m_);
  vr::VREvent_t ev{};
  ev.eventType = type;
  ev.trackedDeviceIndex = dev;
  events_.push_back(ev);
}

bool Runtime::pop_event(vr::VREvent_t *ev, uint32_t size) {
  std::lock_guard<std::mutex> g(m_);
  if (events_.empty()) return false;
  vr::VREvent_t e = events_.front();
  events_.pop_front();
  if (ev) {
    e.eventAgeSeconds = 0;
    memcpy(ev, &e, size < sizeof(e) ? size : sizeof(e));
  }
  return true;
}
}  // namespace svr
