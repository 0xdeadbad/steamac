// SPDX-License-Identifier: GPL-3.0-or-later
// VrSession on top of a host OpenXR runtime (OXRSys): steamac-vrhost becomes an OpenXR application with a Metal
// graphics binding, and the runtime does the encoding and streaming to the headset.
//
// The runtime is loaded directly (active_runtime.json, or XR_RUNTIME_JSON) through the loader negotiation
// interface, so no OpenXR loader is linked.
//
// Prototype scope: head and eye poses, frame submission, controller poses and Touch inputs. No audio (OXRSys does
// not stream headset audio yet), no haptics.
#pragma once

#include "session/session_control.h"
#include "session/vr_session.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace vrhost
{
struct OxrsysConfig
{
	std::string runtime_json;  // empty: $XR_RUNTIME_JSON, then ~/.config/openxr/1/active_runtime.json
	std::string app_name = "steamac-vrhost";
	std::function<void(const std::string &)> log;
};

class OxrsysSession final : public VrSession
{
public:
	explicit OxrsysSession(OxrsysConfig);
	~OxrsysSession() override;

	void start(SessionListener *) override;
	void stop() override;
	bool connected() const override;
	std::optional<HeadsetInfo> headset() const override;
	FrameTiming next_frame_timing() override;
	bool views_at(ns t, std::array<Pose, 2> & pose, std::array<Fov, 2> & fov) override;
	std::optional<TrackedPose> pose_at(Device, ns t) override;
	void submit(const Frame &) override;
	void haptics(Device, float duration_s, float frequency, float amplitude) override {}
	void request_idr() override {}
	void send_audio(std::span<const int16_t>) override {}
	void set_microphone(bool) override {}

	// Status for OxrsysControl and the report line.
	std::string runtime_name() const;
	std::string state_name() const;
	struct Stats
	{
		std::atomic<uint64_t> frames_submitted{0}, frames_skipped{0}, inputs{0};
		std::atomic<int64_t> last_submit_ns{0}; // CPU time of the last submit (NV12 -> swapchain + xrEndFrame)
	};
	const Stats & stats() const;

	struct Impl;

private:
	std::unique_ptr<Impl> p;
};

class OxrsysControl final : public SessionControl
{
public:
	explicit OxrsysControl(OxrsysSession & s) : s(s) {}
	VrSession & session() override { return s; }
	std::string protocol() const override { return "oxrsys"; }
	PairingState pairing_state() const override { return PairingState::waiting; }
	std::vector<Param> params() const override { return {{"runtime", s.runtime_name()}, {"state", s.state_name()}}; }
	bool set_pairing(bool) override { return false; }
	std::vector<Peer> paired() const override { return {}; }
	int unpair(const std::string &) override { return 0; }
	uint32_t bitrate_bps() const override { return 0; }
	Counters counters() const override
	{
		Counters c;
		c.frames_sent = s.stats().frames_submitted.load();
		c.last_encode_ns = s.stats().last_submit_ns.load();
		return c;
	}

private:
	OxrsysSession & s;
};

} // namespace vrhost
