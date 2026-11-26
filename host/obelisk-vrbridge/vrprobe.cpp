// SPDX-License-Identifier: GPL-3.0-or-later
// vrprobe.exe: a minimal OpenVR app for Wine, to check a VR chain without a game. Does what Proton's steam.exe and a
// game do: vrclient_init_registry (Proton's vrclient_x64), then VR_Init through the given openvr_api.dll, then prints
// the render size and the headset pose for a few seconds.
//   vrprobe.exe <path to openvr_api.dll> [seconds]
// Build: zig c++ -target x86_64-windows-gnu -O2 -I<openvr headers> vrprobe.cpp -o vrprobe.exe
#include <windows.h>

#include <cstdio>
#include <cstdlib>

#include "openvr.h"

typedef uint32_t (*init_fn)(vr::EVRInitError *, vr::EVRApplicationType, const char *);
typedef void *(*get_fn)(const char *, vr::EVRInitError *);
typedef void (*shutdown_fn)();
typedef void (*registry_fn)();

static DWORD vr_state() {
  HKEY k;
  DWORD v = 0, size = sizeof v, type;
  if (RegOpenKeyExA(HKEY_CURRENT_USER, "Software\\Wine\\VR", 0, KEY_READ, &k)) return 0xffffffff;
  if (RegQueryValueExA(k, "state", nullptr, &type, (BYTE *)&v, &size)) v = 0xfffffffe;
  RegCloseKey(k);
  return v;
}

int wmain(int argc, wchar_t **argv) {
  if (argc < 2) {
    printf("usage: vrprobe.exe <openvr_api.dll> [seconds]\n");
    return 2;
  }
  int secs = argc > 2 ? _wtoi(argv[2]) : 10;
  setvbuf(stdout, nullptr, _IONBF, 0);

  // 1. what Proton's steam.exe does at start (steam_helper/steam.c setup_vr_registry)
  if (HMODULE vc = LoadLibraryW(L"vrclient_x64")) {
    if (auto reg = (registry_fn)GetProcAddress(vc, "vrclient_init_registry")) {
      reg();
      for (int i = 0; i < 100 && vr_state() != 1; i++) Sleep(100);
      printf("vrclient_init_registry: HKCU\\Software\\Wine\\VR state = %ld\n", (long)vr_state());
    } else printf("vrclient_x64 has no vrclient_init_registry\n");
  } else printf("LoadLibrary(vrclient_x64) failed: %lu\n", GetLastError());

  // 2. what a game does
  HMODULE api = LoadLibraryW(argv[1]);
  if (!api) return printf("LoadLibrary(openvr_api) failed: %lu\n", GetLastError()), 1;
  auto init = (init_fn)GetProcAddress(api, "VR_InitInternal2");
  auto get = (get_fn)GetProcAddress(api, "VR_GetGenericInterface");
  auto shut = (shutdown_fn)GetProcAddress(api, "VR_ShutdownInternal");
  if (!init || !get || !shut) return printf("openvr_api.dll lacks VR_InitInternal2/VR_GetGenericInterface\n"), 1;
  vr::EVRInitError err = vr::VRInitError_None;
  init(&err, vr::VRApplication_Scene, nullptr);
  printf("VR_Init: error %d\n", (int)err);
  if (err != vr::VRInitError_None) return 1;
  auto *sys = (vr::IVRSystem *)get(vr::IVRSystem_Version, &err);
  printf("%s: %p (error %d)\n", vr::IVRSystem_Version, (void *)sys, (int)err);
  if (!sys) return 1;
  uint32_t w = 0, h = 0;
  sys->GetRecommendedRenderTargetSize(&w, &h);
  printf("recommended render size %ux%u per eye\n", w, h);
  for (int i = 0; i < secs * 2; i++) {
    vr::TrackedDevicePose_t p[vr::k_unMaxTrackedDeviceCount] = {};
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, p, vr::k_unMaxTrackedDeviceCount);
    const vr::HmdMatrix34_t &m = p[0].mDeviceToAbsoluteTracking;
    int controllers = 0;
    for (uint32_t d = 1; d < vr::k_unMaxTrackedDeviceCount; d++) controllers += p[d].bPoseIsValid ? 1 : 0;
    printf("t=%4.1fs hmd valid %d  pos (%+.3f, %+.3f, %+.3f)  other valid devices %d\n", i * 0.5, (int)p[0].bPoseIsValid,
           m.m[0][3], m.m[1][3], m.m[2][3], controllers);
    Sleep(500);
  }
  shut();
  printf("done\n");
  return 0;
}
