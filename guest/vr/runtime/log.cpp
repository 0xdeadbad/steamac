// SPDX-License-Identifier: Apache-2.0
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>

namespace svr {
static int log_level() {
  static int level = [] {
    const char *s = getenv("STEAMAC_VR_LOG");
    if (s && *s) return atoi(s);
    // under Obelisk (see transport.cpp) no variable reaches the game: log the interface requests by default there
    return access("/run/obelisk/vr.sock", F_OK) == 0 ? 2 : 0;
  }();
  return level;
}

void logf(int level, const char *fmt, ...) {
  if (level > log_level()) return;
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fprintf(stderr, "steamac-vr: %s\n", buf);
}

void stub_once(const char *name) {
  static std::mutex m;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> g(m);
  if (seen.insert(name).second) logf(1, "unimplemented %s", name);
}
}  // namespace svr
