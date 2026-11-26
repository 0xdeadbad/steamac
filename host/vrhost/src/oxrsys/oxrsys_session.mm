// SPDX-License-Identifier: GPL-3.0-or-later
// OxrsysSession: steamac-vrhost as an OpenXR application (Metal) on a host runtime such as OXRSys.
//
// Threads:
//  * the session thread owns every OpenXR frame call: events, xrWaitFrame/xrBeginFrame, xrSyncActions and
//    xrEndFrame, at the headset's rate. Each xrWaitFrame publishes a FrameTiming that next_frame_timing() hands out.
//  * submit() (producer thread) converts the NV12 eye images into intermediate sRGB textures and marks them latest.
//    The session thread copies the latest images into the swapchains and submits them with the pose they were
//    rendered with, so the runtime reprojects frames that arrive late or are repeated.
//  * views_at()/pose_at() locate spaces from any thread under a shared lock (session teardown takes it exclusively).
#include "oxrsys/oxrsys_session.h"

#include "session/clock.h"
#include "wivrn/input_map.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <time.h>

#define XR_USE_GRAPHICS_API_METAL
#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace vrhost
{
namespace
{
const char * kShader = R"(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
vertex VOut vs(uint vid [[vertex_id]])
{
	float2 p = float2((vid << 1) & 2, vid & 2);
	VOut o;
	o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
	o.uv = float2(p.x, 1.0 - p.y);
	return o;
}
// NV12, BT.709 full range (what the compositor and the test pattern write, gamma-encoded like sRGB) -> linear RGB;
// the sRGB render target encodes it again.
fragment float4 fs(VOut in [[stage_in]], texture2d<float> luma [[texture(0)]], texture2d<float> chroma [[texture(1)]])
{
	constexpr sampler s(filter::linear, address::clamp_to_edge);
	float y = luma.sample(s, in.uv).r;
	float2 c = chroma.sample(s, in.uv).rg - 0.5;
	float3 rgb = saturate(float3(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x));
	float3 lin = select(pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92, rgb <= 0.04045);
	return float4(lin, 1.0);
}
)";

constexpr int64_t kSwapchainFormat = 81; // MTLPixelFormatBGRA8Unorm_sRGB

Pose to_pose(const XrPosef & p)
{
	return {{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w}, {p.position.x, p.position.y, p.position.z}};
}

XrPosef to_xr(const Pose & p)
{
	return {{p.orientation[0], p.orientation[1], p.orientation[2], p.orientation[3]}, {p.position[0], p.position[1], p.position[2]}};
}

Fov to_fov(const XrFovf & f)
{
	return {f.angleLeft, f.angleRight, f.angleUp, f.angleDown};
}

XrFovf to_xr(const Fov & f)
{
	return {f.left, f.right, f.up, f.down};
}

const char * state_str(XrSessionState s)
{
	switch (s)
	{
		case XR_SESSION_STATE_IDLE: return "idle";
		case XR_SESSION_STATE_READY: return "ready";
		case XR_SESSION_STATE_SYNCHRONIZED: return "synchronized";
		case XR_SESSION_STATE_VISIBLE: return "visible";
		case XR_SESSION_STATE_FOCUSED: return "focused";
		case XR_SESSION_STATE_STOPPING: return "stopping";
		case XR_SESSION_STATE_LOSS_PENDING: return "loss pending";
		case XR_SESSION_STATE_EXITING: return "exiting";
		default: return "none";
	}
}

// One action per Touch input the guest table carries; svr index per hand (-1 = not on that hand).
struct InputAction
{
	const char * name;
	XrActionType type;
	const char * left_path;  // nullptr: none
	const char * right_path;
	int svr_index;           // for vector2: x, and y = svr_index + 1
};

const InputAction kInputs[] = {
	{"trigger", XR_ACTION_TYPE_FLOAT_INPUT, "input/trigger/value", "input/trigger/value", svr::trigger_value},
	{"trigger_touch", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/trigger/touch", "input/trigger/touch", svr::trigger_touch},
	{"squeeze", XR_ACTION_TYPE_FLOAT_INPUT, "input/squeeze/value", "input/squeeze/value", svr::squeeze_value},
	{"thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT, "input/thumbstick", "input/thumbstick", svr::thumbstick_x},
	{"thumbstick_click", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/thumbstick/click", "input/thumbstick/click", svr::thumbstick_click},
	{"thumbstick_touch", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/thumbstick/touch", "input/thumbstick/touch", svr::thumbstick_touch},
	{"a_click", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/x/click", "input/a/click", svr::a_click},
	{"a_touch", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/x/touch", "input/a/touch", svr::a_touch},
	{"b_click", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/y/click", "input/b/click", svr::b_click},
	{"b_touch", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/y/touch", "input/b/touch", svr::b_touch},
	{"menu_click", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/menu/click", nullptr, svr::menu_click},
	{"thumbrest_touch", XR_ACTION_TYPE_BOOLEAN_INPUT, "input/thumbrest/touch", "input/thumbrest/touch", svr::thumbrest_touch},
};
constexpr size_t kInputCount = std::size(kInputs);
} // namespace

struct OxrsysSession::Impl
{
	OxrsysConfig cfg;
	SessionListener * listener = nullptr;
	Stats stats;

	// runtime entry points
	void * lib = nullptr;
	PFN_xrGetInstanceProcAddr gipa = nullptr;
#define OXR_FN(name) PFN_##name name = nullptr;
#define OXR_FUNCS(X)                                                                                                       \
	X(xrDestroyInstance) X(xrGetInstanceProperties) X(xrGetSystem) X(xrGetSystemProperties)                              \
	X(xrEnumerateViewConfigurationViews) X(xrGetMetalGraphicsRequirementsKHR) X(xrCreateSession) X(xrDestroySession)     \
	X(xrPollEvent) X(xrBeginSession) X(xrEndSession) X(xrEnumerateReferenceSpaces) X(xrCreateReferenceSpace)               \
	X(xrDestroySpace) X(xrCreateSwapchain) X(xrDestroySwapchain) X(xrEnumerateSwapchainImages) X(xrAcquireSwapchainImage)  \
	X(xrWaitSwapchainImage) X(xrReleaseSwapchainImage) X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame) X(xrLocateViews)      \
	X(xrLocateSpace) X(xrConvertTimespecTimeToTimeKHR) X(xrConvertTimeToTimespecTimeKHR) X(xrStringToPath)                \
	X(xrCreateActionSet) X(xrDestroyActionSet) X(xrCreateAction) X(xrSuggestInteractionProfileBindings)                   \
	X(xrAttachSessionActionSets) X(xrSyncActions) X(xrGetActionStateFloat) X(xrGetActionStateBoolean)                     \
	X(xrGetActionStateVector2f) X(xrCreateActionSpace) X(xrResultToString)
	OXR_FUNCS(OXR_FN)

	XrInstance instance = XR_NULL_HANDLE;
	XrSystemId system = XR_NULL_SYSTEM_ID;
	std::string runtime;
	std::string system_name;

	// session (rebuilt after a loss); guarded by handles for views_at/pose_at
	mutable std::shared_mutex handles;
	XrSession session = XR_NULL_HANDLE;
	XrSpace stage = XR_NULL_HANDLE, view = XR_NULL_HANDLE;
	std::array<XrSpace, 4> hand_space{}; // left grip, left aim, right grip, right aim
	XrActionSet action_set = XR_NULL_HANDLE;
	std::array<XrAction, kInputCount> actions{};
	XrAction grip_action = XR_NULL_HANDLE, aim_action = XR_NULL_HANDLE;
	std::array<XrPath, 2> hand_path{};
	std::array<XrSwapchain, 2> swapchain{};
	std::array<std::vector<id<MTLTexture>>, 2> swapchain_images;
	std::atomic<XrSessionState> state{XR_SESSION_STATE_UNKNOWN};
	std::atomic<bool> running{false};

	// Metal
	id<MTLDevice> device = nil;
	id<MTLCommandQueue> queue = nil;
	id<MTLRenderPipelineState> nv12_pipeline = nil;

	// headset info, frame timing published by the session thread
	mutable std::mutex m;
	std::condition_variable cv;
	std::optional<HeadsetInfo> info;
	FrameTiming timing{};
	uint64_t frame_counter = 0, handed_out = 0;

	// submitted frames: 3 slots of intermediate textures, latest = the newest complete one
	struct Slot
	{
		std::array<id<MTLTexture>, 2> eye{};
		std::array<Pose, 2> pose;
		std::array<Fov, 2> fov;
	};
	std::array<Slot, 3> slots;
	int latest = -1, reading = -1;
	uint32_t slot_w = 0, slot_h = 0;

	std::array<float, 2 * svr::count> last_input{};

	std::thread thread;
	std::atomic<bool> quit{false};

	void log(const std::string & s) const
	{
		if (cfg.log)
			cfg.log("oxrsys: " + s);
	}
	std::string res(XrResult r) const
	{
		char buf[XR_MAX_RESULT_STRING_SIZE] = {};
		if (xrResultToString and instance)
			xrResultToString(instance, r, buf);
		return buf[0] ? buf : std::to_string(int(r));
	}
	bool check(XrResult r, const char * what) const
	{
		if (XR_SUCCEEDED(r))
			return true;
		log(std::string(what) + " failed: " + res(r));
		return false;
	}

	// --- clocks: host ns (CLOCK_UPTIME_RAW) <-> XrTime (the runtime's CLOCK_MONOTONIC base) ---
	static int64_t mono_minus_host()
	{
		return int64_t(clock_gettime_nsec_np(CLOCK_MONOTONIC)) - int64_t(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
	}
	XrTime to_xr_time(ns host) const
	{
		int64_t mono = host + mono_minus_host();
		timespec ts{time_t(mono / 1'000'000'000), long(mono % 1'000'000'000)};
		XrTime t = 0;
		xrConvertTimespecTimeToTimeKHR(instance, &ts, &t);
		return t;
	}
	ns to_host(XrTime t) const
	{
		timespec ts{};
		if (XR_FAILED(xrConvertTimeToTimespecTimeKHR(instance, t, &ts)))
			return now_ns();
		return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec - mono_minus_host();
	}

	// --- runtime loading (the loader's job, done directly) ---
	bool load_runtime()
	{
		std::string json = cfg.runtime_json;
		if (json.empty())
			if (const char * e = getenv("XR_RUNTIME_JSON"))
				json = e;
		if (json.empty())
			json = std::string(getenv("HOME") ?: "") + "/.config/openxr/1/active_runtime.json";
		std::error_code ec;
		auto real = std::filesystem::canonical(json, ec);
		if (ec)
		{
			log("no runtime manifest at " + json);
			return false;
		}
		NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:real.c_str()]];
		NSDictionary * root = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
		NSString * libpath = [root isKindOfClass:[NSDictionary class]] ? root[@"runtime"][@"library_path"] : nil;
		if (not [libpath isKindOfClass:[NSString class]])
		{
			log("manifest without runtime.library_path: " + real.string());
			return false;
		}
		std::filesystem::path lp = libpath.UTF8String;
		if (lp.is_relative())
			lp = real.parent_path() / lp;
		lib = dlopen(lp.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (not lib)
		{
			log(std::string("dlopen failed: ") + dlerror());
			return false;
		}
		auto negotiate = reinterpret_cast<PFN_xrNegotiateLoaderRuntimeInterface>(dlsym(lib, "xrNegotiateLoaderRuntimeInterface"));
		if (not negotiate)
		{
			log("runtime has no xrNegotiateLoaderRuntimeInterface");
			return false;
		}
		XrNegotiateLoaderInfo li{XR_LOADER_INTERFACE_STRUCT_LOADER_INFO, XR_LOADER_INFO_STRUCT_VERSION, sizeof li, 1,
		                         XR_CURRENT_LOADER_RUNTIME_VERSION, XR_MAKE_VERSION(1, 0, 0), XR_MAKE_VERSION(1, 0x3ff, 0xfff)};
		XrNegotiateRuntimeRequest rr{XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST, XR_RUNTIME_INFO_STRUCT_VERSION, sizeof rr};
		if (XR_FAILED(negotiate(&li, &rr)) or not rr.getInstanceProcAddr)
		{
			log("runtime negotiation failed");
			return false;
		}
		gipa = rr.getInstanceProcAddr;
		log("runtime " + lp.string());
		return true;
	}

	bool create_instance()
	{
		PFN_xrCreateInstance create = nullptr;
		gipa(XR_NULL_HANDLE, "xrCreateInstance", reinterpret_cast<PFN_xrVoidFunction *>(&create));
		if (not create)
			return false;
		const char * exts[] = {XR_KHR_METAL_ENABLE_EXTENSION_NAME, XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME};
		XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
		snprintf(ci.applicationInfo.applicationName, sizeof ci.applicationInfo.applicationName, "%s", cfg.app_name.c_str());
		ci.applicationInfo.applicationVersion = 1;
		snprintf(ci.applicationInfo.engineName, sizeof ci.applicationInfo.engineName, "steamac");
		ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
		ci.enabledExtensionCount = uint32_t(std::size(exts));
		ci.enabledExtensionNames = exts;
		if (not check(create(&ci, &instance), "xrCreateInstance"))
			return false;
#define OXR_LOAD(name) gipa(instance, #name, reinterpret_cast<PFN_xrVoidFunction *>(&name));
		OXR_FUNCS(OXR_LOAD)
		XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};
		if (xrGetInstanceProperties(instance, &ip) == XR_SUCCESS)
			runtime = ip.runtimeName;
		return true;
	}

	bool get_system()
	{
		XrSystemGetInfo gi{XR_TYPE_SYSTEM_GET_INFO};
		gi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
		XrResult r = xrGetSystem(instance, &gi, &system);
		if (r == XR_ERROR_FORM_FACTOR_UNAVAILABLE)
			return false; // no headset yet: retry later, quietly
		if (not check(r, "xrGetSystem"))
			return false;
		XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
		xrGetSystemProperties(instance, system, &sp);
		system_name = sp.systemName;

		XrGraphicsRequirementsMetalKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_METAL_KHR};
		if (not check(xrGetMetalGraphicsRequirementsKHR(instance, system, &req), "xrGetMetalGraphicsRequirementsKHR"))
			return false;
		device = req.metalDevice ? (__bridge id<MTLDevice>)req.metalDevice : MTLCreateSystemDefaultDevice();
		queue = [device newCommandQueue];
		NSError * err = nil;
		id<MTLLibrary> lib_ = [device newLibraryWithSource:[NSString stringWithUTF8String:kShader] options:nil error:&err];
		if (not lib_)
		{
			log(std::string("shader: ") + err.localizedDescription.UTF8String);
			return false;
		}
		MTLRenderPipelineDescriptor * pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = [lib_ newFunctionWithName:@"vs"];
		pd.fragmentFunction = [lib_ newFunctionWithName:@"fs"];
		pd.colorAttachments[0].pixelFormat = MTLPixelFormat(kSwapchainFormat);
		nv12_pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&err];
		if (not nv12_pipeline)
		{
			log(std::string("pipeline: ") + err.localizedDescription.UTF8String);
			return false;
		}
		return true;
	}

	XrPath path(const std::string & s)
	{
		XrPath p = XR_NULL_PATH;
		xrStringToPath(instance, s.c_str(), &p);
		return p;
	}

	bool create_actions()
	{
		XrActionSetCreateInfo ai{XR_TYPE_ACTION_SET_CREATE_INFO};
		strcpy(ai.actionSetName, "steamac");
		strcpy(ai.localizedActionSetName, "steamac");
		if (not check(xrCreateActionSet(instance, &ai, &action_set), "xrCreateActionSet"))
			return false;
		hand_path = {path("/user/hand/left"), path("/user/hand/right")};
		auto make = [&](const char * name, XrActionType type, XrAction & out) {
			XrActionCreateInfo ci{XR_TYPE_ACTION_CREATE_INFO};
			snprintf(ci.actionName, sizeof ci.actionName, "%s", name);
			snprintf(ci.localizedActionName, sizeof ci.localizedActionName, "%s", name);
			ci.actionType = type;
			ci.countSubactionPaths = 2;
			ci.subactionPaths = hand_path.data();
			return check(xrCreateAction(action_set, &ci, &out), "xrCreateAction");
		};
		for (size_t i = 0; i < kInputCount; i++)
			if (not make(kInputs[i].name, kInputs[i].type, actions[i]))
				return false;
		if (not make("grip", XR_ACTION_TYPE_POSE_INPUT, grip_action) or not make("aim", XR_ACTION_TYPE_POSE_INPUT, aim_action))
			return false;

		std::vector<XrActionSuggestedBinding> b;
		const char * hands[] = {"/user/hand/left/", "/user/hand/right/"};
		for (int h = 0; h < 2; h++)
		{
			b.push_back({grip_action, path(std::string(hands[h]) + "input/grip/pose")});
			b.push_back({aim_action, path(std::string(hands[h]) + "input/aim/pose")});
			for (size_t i = 0; i < kInputCount; i++)
				if (const char * p = h ? kInputs[i].right_path : kInputs[i].left_path)
					b.push_back({actions[i], path(std::string(hands[h]) + p)});
		}
		XrInteractionProfileSuggestedBinding sb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
		sb.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
		sb.countSuggestedBindings = uint32_t(b.size());
		sb.suggestedBindings = b.data();
		return check(xrSuggestInteractionProfileBindings(instance, &sb), "xrSuggestInteractionProfileBindings");
	}

	bool create_session()
	{
		XrGraphicsBindingMetalKHR gb{XR_TYPE_GRAPHICS_BINDING_METAL_KHR};
		gb.commandQueue = (__bridge void *)queue;
		XrSessionCreateInfo ci{XR_TYPE_SESSION_CREATE_INFO, &gb};
		ci.systemId = system;
		std::unique_lock l(handles);
		if (not check(xrCreateSession(instance, &ci, &session), "xrCreateSession"))
			return false;

		uint32_t n = 0;
		xrEnumerateReferenceSpaces(session, 0, &n, nullptr);
		std::vector<XrReferenceSpaceType> types(n);
		xrEnumerateReferenceSpaces(session, n, &n, types.data());
		bool has_stage = std::find(types.begin(), types.end(), XR_REFERENCE_SPACE_TYPE_STAGE) != types.end();
		XrReferenceSpaceCreateInfo rs{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
		rs.poseInReferenceSpace.orientation.w = 1;
		rs.referenceSpaceType = has_stage ? XR_REFERENCE_SPACE_TYPE_STAGE : XR_REFERENCE_SPACE_TYPE_LOCAL;
		check(xrCreateReferenceSpace(session, &rs, &stage), "xrCreateReferenceSpace(stage)");
		rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
		check(xrCreateReferenceSpace(session, &rs, &view), "xrCreateReferenceSpace(view)");
		log(std::string("tracking space: ") + (has_stage ? "stage" : "local"));

		XrSessionActionSetsAttachInfo at{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
		at.countActionSets = 1;
		at.actionSets = &action_set;
		check(xrAttachSessionActionSets(session, &at), "xrAttachSessionActionSets");
		for (int h = 0; h < 2; h++)
		{
			XrActionSpaceCreateInfo as{XR_TYPE_ACTION_SPACE_CREATE_INFO};
			as.poseInActionSpace.orientation.w = 1;
			as.subactionPath = hand_path[h];
			as.action = grip_action;
			xrCreateActionSpace(session, &as, &hand_space[h * 2]);
			as.action = aim_action;
			xrCreateActionSpace(session, &as, &hand_space[h * 2 + 1]);
		}

		std::array<XrViewConfigurationView, 2> vv{{{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}}};
		uint32_t count = 0;
		if (not check(xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &count,
		                                                vv.data()),
		              "xrEnumerateViewConfigurationViews"))
			return false;
		uint32_t w = vv[0].recommendedImageRectWidth, h = vv[0].recommendedImageRectHeight;
		for (int e = 0; e < 2; e++)
		{
			XrSwapchainCreateInfo sc{XR_TYPE_SWAPCHAIN_CREATE_INFO};
			sc.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
			sc.format = kSwapchainFormat;
			sc.sampleCount = 1;
			sc.width = w;
			sc.height = h;
			sc.faceCount = 1;
			sc.arraySize = 1;
			sc.mipCount = 1;
			if (not check(xrCreateSwapchain(session, &sc, &swapchain[e]), "xrCreateSwapchain"))
				return false;
			uint32_t ni = 0;
			xrEnumerateSwapchainImages(swapchain[e], 0, &ni, nullptr);
			std::vector<XrSwapchainImageMetalKHR> imgs(ni, {XR_TYPE_SWAPCHAIN_IMAGE_METAL_KHR});
			xrEnumerateSwapchainImages(swapchain[e], ni, &ni, reinterpret_cast<XrSwapchainImageBaseHeader *>(imgs.data()));
			swapchain_images[e].clear();
			for (auto & i: imgs)
				swapchain_images[e].push_back((__bridge id<MTLTexture>)i.texture);
		}

		HeadsetInfo hi;
		hi.eye_width = hi.render_eye_width = hi.nominal_render_width = w;
		hi.eye_height = hi.render_eye_height = hi.nominal_render_height = h;
		hi.codecs = {Codec::raw};
		hi.name = system_name.empty() ? runtime : system_name;
		hi.foveation = FoveationModel::none; // the compositor only converts; the runtime streams at its own size
		hi.preferred_refresh_rate = 72;
		hi.refresh_rates = {72};
		for (auto & f: hi.fov)
			f = Fov{-0.75f, 0.75f, 0.8f, -0.8f};
		std::lock_guard lm(m);
		info = hi;
		log("session created, " + std::to_string(w) + "x" + std::to_string(h) + " per eye");
		return true;
	}

	void destroy_session()
	{
		running = false;
		std::unique_lock l(handles);
		for (auto & s: swapchain)
			if (s)
				xrDestroySwapchain(s), s = XR_NULL_HANDLE;
		for (auto & s: hand_space)
			if (s)
				xrDestroySpace(s), s = XR_NULL_HANDLE;
		if (stage)
			xrDestroySpace(stage), stage = XR_NULL_HANDLE;
		if (view)
			xrDestroySpace(view), view = XR_NULL_HANDLE;
		if (session)
			xrDestroySession(session), session = XR_NULL_HANDLE;
		swapchain_images = {};
		state = XR_SESSION_STATE_UNKNOWN;
	}

	void set_connected(bool on)
	{
		if (running.exchange(on) == on)
			return;
		if (on)
		{
			std::optional<HeadsetInfo> hi;
			{
				std::lock_guard l(m);
				hi = info;
			}
			log("running (" + std::string(state_str(state)) + ")");
			if (listener and hi)
				listener->on_connected(*hi);
		}
		else
		{
			log("stopped");
			if (listener)
				listener->on_disconnected();
			cv.notify_all();
		}
	}

	// Returns false when the session is gone (lost/exiting) and must be rebuilt.
	bool poll_events()
	{
		XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
		while (xrPollEvent(instance, &ev) == XR_SUCCESS)
		{
			if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
			{
				auto s = reinterpret_cast<XrEventDataSessionStateChanged &>(ev).state;
				state = s;
				log(std::string("state ") + state_str(s));
				if (s == XR_SESSION_STATE_READY)
				{
					XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
					bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
					if (check(xrBeginSession(session, &bi), "xrBeginSession"))
						set_connected(true);
				}
				else if (s == XR_SESSION_STATE_STOPPING)
				{
					set_connected(false);
					xrEndSession(session);
				}
				else if (s == XR_SESSION_STATE_EXITING or s == XR_SESSION_STATE_LOSS_PENDING)
				{
					set_connected(false);
					return false;
				}
			}
			else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING)
			{
				set_connected(false);
				return false;
			}
			ev = {XR_TYPE_EVENT_DATA_BUFFER};
		}
		return true;
	}

	void sync_inputs()
	{
		XrActiveActionSet aas{action_set, XR_NULL_PATH};
		XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
		si.countActiveActionSets = 1;
		si.activeActionSets = &aas;
		if (xrSyncActions(session, &si) != XR_SUCCESS)
			return; // XR_SESSION_NOT_FOCUSED: nothing to read
		std::vector<InputValue> out;
		ns t = now_ns();
		auto put = [&](int hand, int index, float v) {
			float & last = last_input[hand * svr::count + index];
			if (last == v)
				return;
			last = v;
			uint32_t id;
			if (vrl_input_to_wivrn(hand, index, id))
				out.push_back({id, v, t});
		};
		for (int h = 0; h < 2; h++)
		{
			for (size_t i = 0; i < kInputCount; i++)
			{
				if (not(h ? kInputs[i].right_path : kInputs[i].left_path))
					continue;
				XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
				gi.action = actions[i];
				gi.subactionPath = hand_path[h];
				int idx = kInputs[i].svr_index;
				switch (kInputs[i].type)
				{
					case XR_ACTION_TYPE_FLOAT_INPUT: {
						XrActionStateFloat s{XR_TYPE_ACTION_STATE_FLOAT};
						xrGetActionStateFloat(session, &gi, &s);
						put(h, idx, s.currentState);
						// Touch has no trigger/grip click: derive it as the WiVRn clients' runtimes do.
						if (idx == svr::trigger_value)
							put(h, svr::trigger_click, s.currentState > 0.9f);
						if (idx == svr::squeeze_value)
							put(h, svr::squeeze_click, s.currentState > 0.9f);
						break;
					}
					case XR_ACTION_TYPE_BOOLEAN_INPUT: {
						XrActionStateBoolean s{XR_TYPE_ACTION_STATE_BOOLEAN};
						xrGetActionStateBoolean(session, &gi, &s);
						put(h, idx, s.currentState ? 1.f : 0.f);
						break;
					}
					case XR_ACTION_TYPE_VECTOR2F_INPUT: {
						XrActionStateVector2f s{XR_TYPE_ACTION_STATE_VECTOR2F};
						xrGetActionStateVector2f(session, &gi, &s);
						put(h, idx, s.currentState.x);
						put(h, idx + 1, s.currentState.y);
						break;
					}
					default:
						break;
				}
			}
		}
		if (not out.empty())
		{
			stats.inputs += out.size();
			if (listener)
				listener->on_input(out);
		}
	}

	// Copies the latest submitted frame into the swapchains; false when nothing was submitted yet.
	bool compose(std::array<XrCompositionLayerProjectionView, 2> & pv)
	{
		int r;
		{
			std::lock_guard l(m);
			if (latest < 0)
				return false;
			r = reading = latest;
		}
		auto & slot = slots[r];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		std::array<uint32_t, 2> index{};
		for (int e = 0; e < 2; e++)
		{
			XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
			if (not check(xrAcquireSwapchainImage(swapchain[e], &ai, &index[e]), "xrAcquireSwapchainImage"))
			{
				std::lock_guard l(m);
				reading = -1;
				return false;
			}
			XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
			wi.timeout = 100'000'000;
			check(xrWaitSwapchainImage(swapchain[e], &wi), "xrWaitSwapchainImage");
			id<MTLTexture> dst = swapchain_images[e][index[e]];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromTexture:slot.eye[e] toTexture:dst];
			[blit endEncoding];
		}
		[cb commit];
		[cb waitUntilCompleted]; // the runtime may read the image on its own queue
		for (int e = 0; e < 2; e++)
		{
			XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
			xrReleaseSwapchainImage(swapchain[e], &ri);
			pv[e] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
			pv[e].pose = to_xr(slot.pose[e]);
			pv[e].fov = to_xr(slot.fov[e]);
			pv[e].subImage.swapchain = swapchain[e];
			pv[e].subImage.imageRect = {{0, 0}, {int32_t(slot.eye[e].width), int32_t(slot.eye[e].height)}};
		}
		std::lock_guard l(m);
		reading = -1;
		return true;
	}

	void frame()
	{
		XrFrameState fs{XR_TYPE_FRAME_STATE};
		XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
		if (not check(xrWaitFrame(session, &wi, &fs), "xrWaitFrame"))
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			return;
		}
		XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
		xrBeginFrame(session, &bi);
		float new_hz = 0;
		{
			std::lock_guard l(m);
			timing.wake_up = now_ns();
			timing.predicted_display = to_host(fs.predictedDisplayTime);
			timing.display_period = fs.predictedDisplayPeriod;
			timing.frame_id = ++frame_counter;
			if (info and fs.predictedDisplayPeriod > 0)
			{
				float hz = std::round(1e9f / float(fs.predictedDisplayPeriod));
				if (std::abs(hz - info->preferred_refresh_rate) > 0.5f)
				{
					info->preferred_refresh_rate = hz;
					info->refresh_rates = {hz};
					new_hz = hz;
				}
			}
		}
		cv.notify_all();
		// Outside m: listeners call back into headset()/views_at().
		if (new_hz > 0 and listener)
			listener->on_refresh_rate(new_hz);
		sync_inputs();

		std::array<XrCompositionLayerProjectionView, 2> pv{};
		XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
		const XrCompositionLayerBaseHeader * layers[] = {reinterpret_cast<XrCompositionLayerBaseHeader *>(&layer)};
		XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
		ei.displayTime = fs.predictedDisplayTime;
		ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
		if (fs.shouldRender and compose(pv))
		{
			layer.space = stage;
			layer.viewCount = 2;
			layer.views = pv.data();
			ei.layerCount = 1;
			ei.layers = layers;
		}
		else
			stats.frames_skipped++;
		check(xrEndFrame(session, &ei), "xrEndFrame");
	}

	void run()
	{
		while (not quit)
		{
			if (not instance and not(create_instance() and create_actions()))
			{
				std::this_thread::sleep_for(std::chrono::seconds(2));
				continue;
			}
			if (system == XR_NULL_SYSTEM_ID and not get_system())
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}
			if (not session and not create_session())
			{
				destroy_session();
				std::this_thread::sleep_for(std::chrono::seconds(2));
				continue;
			}
			if (not poll_events())
			{
				destroy_session();
				std::this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}
			if (running)
				frame();
			else
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		if (session and running)
			xrEndSession(session);
		destroy_session();
		cv.notify_all();
	}

	// NV12 IOSurface -> intermediate sRGB texture (render pass; the swapchain images are render targets only).
	void convert(IOSurfaceRef surface, id<MTLTexture> dst, id<MTLCommandBuffer> cb)
	{
		auto plane = [&](size_t i, MTLPixelFormat f) {
			MTLTextureDescriptor * d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
			                                                                              width:IOSurfaceGetWidthOfPlane(surface, i)
			                                                                             height:IOSurfaceGetHeightOfPlane(surface, i)
			                                                                          mipmapped:NO];
			d.usage = MTLTextureUsageShaderRead;
			return [device newTextureWithDescriptor:d iosurface:surface plane:i];
		};
		id<MTLTexture> y = plane(0, MTLPixelFormatR8Unorm), c = plane(1, MTLPixelFormatRG8Unorm);
		MTLRenderPassDescriptor * rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = dst;
		rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
		[enc setRenderPipelineState:nv12_pipeline];
		[enc setFragmentTexture:y atIndex:0];
		[enc setFragmentTexture:c atIndex:1];
		[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[enc endEncoding];
	}

	void submit(const Frame & f)
	{
		if (not running or not f.eye[0].surface or not f.eye[1].surface)
			return;
		ns t0 = now_ns();
		uint32_t w = uint32_t(IOSurfaceGetWidth(f.eye[0].surface)), h = uint32_t(IOSurfaceGetHeight(f.eye[0].surface));
		int s;
		{
			std::lock_guard l(m);
			if (w != slot_w or h != slot_h)
			{
				MTLTextureDescriptor * d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormat(kSwapchainFormat)
				                                                                              width:w
				                                                                             height:h
				                                                                          mipmapped:NO];
				d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
				d.storageMode = MTLStorageModePrivate;
				for (auto & sl: slots)
					for (auto & t: sl.eye)
						t = [device newTextureWithDescriptor:d];
				slot_w = w;
				slot_h = h;
				latest = -1;
			}
			s = 0;
			while (s == latest or s == reading)
				s++;
		}
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		for (int e = 0; e < 2; e++)
			convert(f.eye[e].surface, slots[s].eye[e], cb);
		[cb commit];
		[cb waitUntilCompleted]; // the caller reuses the IOSurfaces when submit returns
		{
			std::lock_guard l(m);
			slots[s].pose = f.pose;
			slots[s].fov = f.fov;
			latest = s;
		}
		stats.frames_submitted++;
		stats.last_submit_ns = now_ns() - t0;
	}
};

OxrsysSession::OxrsysSession(OxrsysConfig cfg) : p(std::make_unique<Impl>())
{
	p->cfg = std::move(cfg);
}

OxrsysSession::~OxrsysSession()
{
	stop();
	if (p->instance)
	{
		if (p->action_set)
			p->xrDestroyActionSet(p->action_set);
		p->xrDestroyInstance(p->instance);
	}
}

void OxrsysSession::start(SessionListener * l)
{
	p->listener = l;
	if (not p->load_runtime())
		return;
	p->quit = false;
	p->thread = std::thread([this] { p->run(); });
}

void OxrsysSession::stop()
{
	p->quit = true;
	p->cv.notify_all();
	if (p->thread.joinable())
		p->thread.join();
}

bool OxrsysSession::connected() const
{
	return p->running;
}

std::optional<HeadsetInfo> OxrsysSession::headset() const
{
	std::lock_guard l(p->m);
	return p->info;
}

FrameTiming OxrsysSession::next_frame_timing()
{
	std::unique_lock l(p->m);
	p->cv.wait_for(l, std::chrono::milliseconds(100), [&] { return p->frame_counter > p->handed_out or not p->running or p->quit; });
	if (p->frame_counter == p->handed_out)
	{
		// No new headset frame (not running, or the runtime stalled): keep the producer paced at ~72 Hz.
		FrameTiming t = p->timing;
		t.wake_up = now_ns();
		t.predicted_display = t.wake_up + 14'000'000;
		return t;
	}
	p->handed_out = p->frame_counter;
	return p->timing;
}

bool OxrsysSession::views_at(ns t, std::array<Pose, 2> & pose, std::array<Fov, 2> & fov)
{
	std::shared_lock l(p->handles);
	if (not p->session or not p->stage)
		return false;
	XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
	li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
	li.displayTime = p->to_xr_time(t);
	li.space = p->stage;
	XrViewState vs{XR_TYPE_VIEW_STATE};
	std::array<XrView, 2> v{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
	uint32_t n = 0;
	if (XR_FAILED(p->xrLocateViews(p->session, &li, &vs, 2, &n, v.data())) or n != 2)
		return false;
	for (int e = 0; e < 2; e++)
	{
		pose[e] = to_pose(v[e].pose);
		fov[e] = to_fov(v[e].fov);
	}
	if (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)
	{
		std::lock_guard lm(p->m);
		if (p->info)
			p->info->fov = fov;
	}
	return true;
}

std::optional<TrackedPose> OxrsysSession::pose_at(Device d, ns t)
{
	std::shared_lock l(p->handles);
	XrSpace space = XR_NULL_HANDLE;
	switch (d)
	{
		case Device::head: space = p->view; break;
		case Device::left_grip: space = p->hand_space[0]; break;
		case Device::left_aim: space = p->hand_space[1]; break;
		case Device::right_grip: space = p->hand_space[2]; break;
		case Device::right_aim: space = p->hand_space[3]; break;
		default: break;
	}
	if (not p->session or not space or not p->stage)
		return std::nullopt;
	XrSpaceVelocity vel{XR_TYPE_SPACE_VELOCITY};
	XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION, &vel};
	if (XR_FAILED(p->xrLocateSpace(space, p->stage, p->to_xr_time(t), &loc)))
		return std::nullopt;
	TrackedPose tp{.device = d, .time = t, .pose = to_pose(loc.pose)};
	tp.orientation_valid = loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
	tp.position_valid = loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT;
	if (not tp.orientation_valid and not tp.position_valid)
		return std::nullopt;
	if (vel.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT)
		tp.linear_velocity = {vel.linearVelocity.x, vel.linearVelocity.y, vel.linearVelocity.z};
	if (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT)
		tp.angular_velocity = {vel.angularVelocity.x, vel.angularVelocity.y, vel.angularVelocity.z};
	return tp;
}

void OxrsysSession::submit(const Frame & f)
{
	p->submit(f);
}

std::string OxrsysSession::runtime_name() const
{
	return p->runtime;
}

std::string OxrsysSession::state_name() const
{
	return state_str(p->state);
}

const OxrsysSession::Stats & OxrsysSession::stats() const
{
	return p->stats;
}

} // namespace vrhost
