// SPDX-License-Identifier: GPL-3.0-or-later
// obelisk-vrbridge: lets a game outside the steamac VM (Obelisk's Linux engine, voile-lkd) use steamac-vrhost.
//
//   game (Wine) -> Proton's vrclient_x64 -> steamac guest runtime (vrclient.so, STEAMAC_VR_SOCKET=stream:<path>)
//     -> this bridge (guest VR protocol, steamac_vr_guest.h) -> wmtr_vr (the VM-side half of the VR link, reused
//     as is) -> steamac-vrhost --link-socket <path> --mach-name <name> -> headset
//
// Stage 1 (tracking only): HELLO/FRAME/HAPTIC are answered by wmtr_vr_query from the host's newest state, timing and
// poses. SVR_SUBMIT carries DXMT texture handles of the steamac VM, which mean nothing here: submits are acknowledged
// and dropped (counted), so the headset shows no game image yet.
//
// usage: obelisk-vrbridge --listen PATH --vrhost PATH [--mach-name NAME]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "steamac_vr_guest.h"
#include "winemetal_thunks.h"
#include "wmtr_vr.h"

#define MSG_MAX 4096

static struct wmtr_vr *g_vr;
static pthread_mutex_t g_vr_lock = PTHREAD_MUTEX_INITIALIZER;
static const char *g_vrhost, *g_mach;
static atomic_ulong g_clients, g_queries, g_submits, g_errors;

static void logf_(const char *fmt, ...) {
  char t[32];
  time_t now = time(NULL);
  strftime(t, sizeof t, "%H:%M:%S", localtime(&now));
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[%s obelisk-vrbridge] ", t);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

// the link to steamac-vrhost, made on first use and kept (wmtr_vr reads the host's packets on its own thread)
static struct wmtr_vr *vr_link(void) {
  pthread_mutex_lock(&g_vr_lock);
  if (!g_vr) {
    g_vr = wmtr_vr_connect(g_vrhost, g_mach);
    logf_(g_vr ? "linked to steamac-vrhost at %s" : "cannot link to steamac-vrhost at %s (is it running with --link-socket?)",
          g_vrhost);
  }
  struct wmtr_vr *vr = g_vr;
  pthread_mutex_unlock(&g_vr_lock);
  return vr;
}

static int send_all(int fd, const void *p, size_t len) {
  for (size_t done = 0; done < len;) {
    ssize_t n = send(fd, (const char *)p + done, len - done, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    done += (size_t)n;
  }
  return 0;
}
// A connection: its socket and the fds that arrived with its bytes (SCM_RIGHTS), oldest first.
#define FDQ_MAX 16
struct conn {
  int fd;
  unsigned long id;
  int fdq[FDQ_MAX];
  int nfdq;
};

static int recv_all(struct conn *c, void *p, size_t len) {
  for (size_t done = 0; done < len;) {
    _Alignas(struct cmsghdr) char ctl[CMSG_SPACE(sizeof(int) * 4)];
    struct iovec iov = {(char *)p + done, len - done};
    struct msghdr m = {.msg_iov = &iov, .msg_iovlen = 1, .msg_control = ctl, .msg_controllen = sizeof ctl};
    ssize_t n = recvmsg(c->fd, &m, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    for (struct cmsghdr *ch = CMSG_FIRSTHDR(&m); ch; ch = CMSG_NXTHDR(&m, ch)) {
      if (ch->cmsg_level != SOL_SOCKET || ch->cmsg_type != SCM_RIGHTS) continue;
      int k = (int)((ch->cmsg_len - CMSG_LEN(0)) / sizeof(int));
      for (int i = 0; i < k; i++) {
        int f;
        memcpy(&f, CMSG_DATA(ch) + i * sizeof(int), sizeof f);
        if (c->nfdq < FDQ_MAX) c->fdq[c->nfdq++] = f;
        else close(f);
      }
    }
    done += (size_t)n;
  }
  return 0;
}

static void fdq_drop(struct conn *c) {
  for (int i = 0; i < c->nfdq; i++) close(c->fdq[i]);
  c->nfdq = 0;
}

// --- Neptune eye images (SVR_NPT_SUBMIT) ---------------------------------------------------------------------------
// Each is a linear image in memory the render server shares by fd: mapped once, wrapped as an MTLBuffer and a linear
// MTLTexture on it, kept by its fd number in the game process (stable while its pool lives).
static atomic_ulong g_npt_frames, g_npt_errors;
static NSMutableDictionary<NSString *, id<MTLTexture>> *g_npt_images; // "<connection>:<fd in the game>"
static id<MTLDevice> g_device;
static pthread_mutex_t g_npt_lock = PTHREAD_MUTEX_INITIALIZER;

static MTLPixelFormat npt_pixel_format(uint32_t dxgi) {
  switch (dxgi) {
  case 27: case 28: case 29: case 30: return MTLPixelFormatRGBA8Unorm;  // R8G8B8A8 TYPELESS/UNORM/UNORM_SRGB/UINT
  case 87: case 90: case 91: return MTLPixelFormatBGRA8Unorm;           // B8G8R8A8 UNORM/TYPELESS/UNORM_SRGB
  case 88: case 92: case 93: return MTLPixelFormatBGRA8Unorm;           // B8G8R8X8 (alpha ignored by the headset)
  case 24: return MTLPixelFormatRGB10A2Unorm;                            // R10G10B10A2_UNORM
  case 10: return MTLPixelFormatRGBA16Float;                             // R16G16B16A16_FLOAT
  default: return MTLPixelFormatInvalid;
  }
}

// the texture for this eye image; takes the fd (closes it when the image is known already). nil on failure.
static id<MTLTexture> npt_image(unsigned long conn_id, const struct svr_npt_eye *n, int fd) {
  pthread_mutex_lock(&g_npt_lock);
  if (!g_npt_images) g_npt_images = [NSMutableDictionary dictionary];
  if (!g_device) g_device = MTLCreateSystemDefaultDevice();
  NSString *key = [NSString stringWithFormat:@"%lu:%llu", conn_id, (unsigned long long)n->game_fd];
  id<MTLTexture> t = g_npt_images[key];
  uint32_t h_total = n->height * (n->slices ? n->slices : 1);
  if (t && t.width == n->width && t.height == h_total && t.bufferBytesPerRow == n->pitch) {
    pthread_mutex_unlock(&g_npt_lock);
    if (fd >= 0) close(fd);
    return t;
  }
  t = nil;
  MTLPixelFormat pf = npt_pixel_format(n->dxgi_format);
  struct stat st;
  if (fd < 0 || pf == MTLPixelFormatInvalid || fstat(fd, &st)) {
    logf_("neptune image fd %llu: %s", (unsigned long long)n->game_fd, fd < 0 ? "no fd" : pf == MTLPixelFormatInvalid ? "unsupported format" : "fstat failed");
    goto out;
  }
  {
    size_t need = (size_t)n->pitch * h_total;
    size_t page = (size_t)getpagesize();
    size_t len = ((st.st_size > 0 ? (size_t)st.st_size : need) + page - 1) & ~(page - 1);
    if (len < need) {
      logf_("neptune image fd %llu: %zu bytes, %ux%u pitch %u needs %zu", (unsigned long long)n->game_fd, len, n->width, h_total, n->pitch, need);
      goto out;
    }
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      logf_("neptune image fd %llu: mmap %zu: %s", (unsigned long long)n->game_fd, len, strerror(errno));
      goto out;
    }
    id<MTLBuffer> b = [g_device newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared
                                            deallocator:^(void *ptr, NSUInteger l) { munmap(ptr, l); }];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:n->width height:h_total mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    t = b ? [b newTextureWithDescriptor:d offset:0 bytesPerRow:n->pitch] : nil;
    if (!t) {
      logf_("neptune image fd %llu: no linear texture (%ux%u pitch %u, %zu bytes, buffer %s)", (unsigned long long)n->game_fd, n->width, h_total, n->pitch, len, b ? "ok" : "failed");
      munmap(p, len);
      goto out;
    }
    g_npt_images[key] = t;  // until its connection ends (npt_forget): the buffer's deallocator unmaps it
    logf_("neptune image fd %llu: %ux%u (%u slices) dxgi %u pitch %u mapped %zu bytes", (unsigned long long)n->game_fd, n->width, h_total, n->slices, n->dxgi_format, n->pitch, len);
  }
out:
  pthread_mutex_unlock(&g_npt_lock);
  if (fd >= 0) close(fd);
  return t;
}

// the images of a connection that ended: their game process is gone, and its fd numbers may name new images
static void npt_forget(unsigned long conn_id) {
  NSString *prefix = [NSString stringWithFormat:@"%lu:", conn_id];
  pthread_mutex_lock(&g_npt_lock);
  NSArray<NSString *> *keys = [g_npt_images.allKeys copy];
  unsigned n = 0;
  for (NSString *k in keys)
    if ([k hasPrefix:prefix]) [g_npt_images removeObjectForKey:k], n++;
  pthread_mutex_unlock(&g_npt_lock);
  if (n) logf_("runtime %lu: released %u Neptune images", conn_id, n);
}

// fence waits: a serial queue, so frames reach the host in order and the game's queries never wait for its GPU
static dispatch_queue_t g_fence_queue;
static atomic_ulong g_fence_waits, g_fence_timeouts;
static _Atomic double g_fence_wait_ms;

static int npt_submit(struct conn *c, const struct svr_npt_submit_req *q) {
  int fds[2] = {-1, -1}, fence = -1;
  for (int e = 0; e < 2; e++) {
    uint32_t s = q->npt[e].fd_slot;
    if ((int)s < c->nfdq) fds[e] = c->fdq[s];
  }
  if (q->fence_slot != SVR_NPT_NO_SLOT && (int)q->fence_slot < c->nfdq) fence = c->fdq[q->fence_slot];
  if (fds[0] == fds[1]) fds[1] = fds[1] >= 0 ? dup(fds[1]) : -1;  // one fd for both eyes: each lookup takes one
  c->nfdq = 0;  // ownership moved: images to npt_image, the fence to the wait below
  struct svr_wmtvr_submit info = q->info;
  for (int e = 0; e < 2; e++) {
    const struct svr_npt_eye *n = &q->npt[e];
    id<MTLTexture> t = npt_image(c->id, n, fds[e]);
    if (!t) {
      if (e == 0 && fds[1] >= 0) close(fds[1]);
      if (fence >= 0) close(fence);
      return -EINVAL;
    }
    struct svr_wmtvr_eye *ey = &info.eye[e];
    if (!ey->w) ey->x = 0, ey->y = 0, ey->w = n->width, ey->h = n->height;
    ey->y += ey->slice * n->height;  // the slices are stacked in the image
    ey->slice = 0;
    ey->texture = (uint64_t)(uintptr_t)(__bridge void *)t;
  }
  info.wait_event = 0, info.wait_value = 0;
  _Static_assert(sizeof(struct WMTVRSubmitInfo) == sizeof(struct svr_wmtvr_submit), "WMTVRSubmitInfo layout");
  if (fence < 0) {  // Neptune waited for the copy itself
    int st = wmtr_vr_submit(vr_link(), (const struct WMTVRSubmitInfo *)&info);
    return st ? -EIO : 0;
  }
  dispatch_async(g_fence_queue, ^{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    struct pollfd pf = {fence, POLLIN, 0};
    int r;
    do r = poll(&pf, 1, 100); while (r < 0 && errno == EINTR);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    close(fence);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    atomic_fetch_add(&g_fence_waits, 1);
    double cur = atomic_load(&g_fence_wait_ms);
    while (!atomic_compare_exchange_weak(&g_fence_wait_ms, &cur, cur + ms)) {}
    if (r <= 0) {
      // not signalled in 100 ms: show it anyway rather than freeze the headset on the last frame
      if (atomic_fetch_add(&g_fence_timeouts, 1) < 5) logf_("fence %d not signalled after %.0f ms (poll %d)", fence, ms, r);
    }
    wmtr_vr_submit(vr_link(), (const struct WMTVRSubmitInfo *)&info);
  });
  return 0;
}

static void *client_main(void *arg) {
  struct conn cn = {.fd = (int)(intptr_t)arg}, *c = &cn;
  int fd = c->fd;
  int on = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
  unsigned long id = atomic_fetch_add(&g_clients, 1) + 1;
  c->id = id;
  logf_("runtime %lu connected", id);
  _Alignas(8) uint8_t buf[MSG_MAX];
  struct svr_hdr *h = (struct svr_hdr *)buf;
  for (;;) {
    if (recv_all(c, h, sizeof *h)) break;
    if (h->size < sizeof *h || h->size > MSG_MAX) {
      logf_("runtime %lu: bad message size %u (op %u)", id, h->size, h->op);
      break;
    }
    if (recv_all(c, buf + sizeof *h, h->size - sizeof *h)) break;
    if (h->op == SVR_HELLO) {
      const struct svr_hello_req *q = (const void *)buf;
      if (h->size >= sizeof *q) logf_("runtime %lu: hello from pid %u, app '%.64s'", id, q->pid, q->app_key);
    }
    if (h->op == SVR_NPT_SUBMIT) {
      int st = h->size == sizeof(struct svr_npt_submit_req) ? npt_submit(c, (const void *)buf) : -EINVAL;
      fdq_drop(c);
      atomic_fetch_add(st ? &g_npt_errors : &g_npt_frames, 1);
      struct svr_ack a = {{SVR_NPT_SUBMIT, sizeof a}, st, 0};
      if (send_all(fd, &a, sizeof a)) break;
      continue;
    }
    fdq_drop(c);  // nothing else carries fds
    if (h->op == SVR_SUBMIT) {
      atomic_fetch_add(&g_submits, 1);
      struct svr_ack a = {{SVR_SUBMIT, sizeof a}, 0, 0};
      if (send_all(fd, &a, sizeof a)) break;
      continue;
    }
    atomic_fetch_add(&g_queries, 1);
    uint32_t op = h->op;
    int st = wmtr_vr_query(vr_link(), buf, sizeof buf);
    if (st != 0) {
      atomic_fetch_add(&g_errors, 1);
      struct svr_ack a = {{op, sizeof a}, -EINVAL, 0};
      if (send_all(fd, &a, sizeof a)) break;
      continue;
    }
    if (send_all(fd, buf, h->size)) break;
  }
  logf_("runtime %lu disconnected", id);
  fdq_drop(c);
  npt_forget(id);
  close(fd);
  return NULL;
}

static void *stats_main(void *arg) {
  (void)arg;
  unsigned long q0 = 0, s0 = 0, f0 = 0, e0 = 0, fw0 = 0;
  for (;;) {
    sleep(5);
    unsigned long q = atomic_load(&g_queries), s = atomic_load(&g_submits);
    unsigned long f = atomic_load(&g_npt_frames), e = atomic_load(&g_npt_errors);
    unsigned long fw = atomic_load(&g_fence_waits);
    double fms = atomic_exchange(&g_fence_wait_ms, 0.0);
    if (q != q0 || s != s0 || f != f0 || e != e0)
      logf_("5 s: %lu queries, %lu Neptune frames (%.1f fps), %lu Neptune errors, fence wait avg %.2f ms (%lu timeouts), %lu other submits dropped, %lu errors",
            q - q0, f - f0, (f - f0) / 5.0, e - e0, fw - fw0 ? fms / (fw - fw0) : 0.0, atomic_load(&g_fence_timeouts), s - s0, atomic_load(&g_errors));
    q0 = q, s0 = s, f0 = f, e0 = e, fw0 = fw;
  }
  return NULL;
}

// a listening socket at `path`, or -1
static int listen_at(const char *path) {
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  if (strlen(path) >= sizeof a.sun_path) {
    logf_("socket path too long (max %zu): %s", sizeof a.sun_path - 1, path);
    return -1;
  }
  snprintf(a.sun_path, sizeof a.sun_path, "%s", path);
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  unlink(path);
  if (s < 0 || bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 8)) {
    logf_("cannot listen at %s: %s", path, strerror(errno));
    if (s >= 0) close(s);
    return -1;
  }
  chmod(path, 0600);
  logf_("listening at %s", path);
  return s;
}

static void *accept_main(void *arg) {
  int s = (int)(intptr_t)arg;
  for (;;) {
    int c = accept(s, NULL, NULL);
    if (c < 0) {
      if (errno == EINTR) continue;
      break;  // the folder went away with its game session
    }
    pthread_t t;
    pthread_create(&t, NULL, client_main, (void *)(intptr_t)c);
    pthread_detach(t);
  }
  close(s);
  return NULL;
}

// --obelisk: Obelisk's Linux engine makes a socket folder per game launch, $TMPDIR/obelisk-engine-<pid>, which the
// game sees as /run/obelisk (FEXDARWIN_BIND). A vr.sock is put in each new one: STEAMAC_VR_SOCKET=stream:/run/obelisk/vr.sock
// in the game's launch settings. Short and the same on every launch (macOS caps socket paths at 104 bytes; the
// prefix folder is longer than that).
static void watch_obelisk(void) {
  const char *tmp = getenv("TMPDIR");
  NSString *root = [NSString stringWithUTF8String:tmp && *tmp ? tmp : "/tmp"];
  NSMutableSet<NSString *> *seen = [NSMutableSet set];
  logf_("watching %s for obelisk-engine-* session folders", root.UTF8String);
  for (;;) {
    @autoreleasepool {
      NSArray<NSString *> *names = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil];
      for (NSString *n in names) {
        if (![n hasPrefix:@"obelisk-engine-"] || [seen containsObject:n]) continue;
        [seen addObject:n];
        NSString *sock = [[root stringByAppendingPathComponent:n] stringByAppendingPathComponent:@"vr.sock"];
        int s = listen_at(sock.fileSystemRepresentation);
        if (s < 0) continue;
        pthread_t t;
        pthread_create(&t, NULL, accept_main, (void *)(intptr_t)s);
        pthread_detach(t);
      }
    }
    usleep(200 * 1000);
  }
}

static int usage(const char *argv0) {
  fprintf(stderr, "usage: %s (--listen PATH | --obelisk) --vrhost PATH [--mach-name NAME]\n", argv0);
  return 2;
}

int main(int argc, char **argv) {
  const char *listen_path = NULL;
  int obelisk = 0;
  g_mach = "obelisk.vrbridge";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--listen") && i + 1 < argc) listen_path = argv[++i];
    else if (!strcmp(argv[i], "--obelisk")) obelisk = 1;
    else if (!strcmp(argv[i], "--vrhost") && i + 1 < argc) g_vrhost = argv[++i];
    else if (!strcmp(argv[i], "--mach-name") && i + 1 < argc) g_mach = argv[++i];
    else return usage(argv[0]);
  }
  if ((!listen_path && !obelisk) || !g_vrhost) return usage(argv[0]);
  signal(SIGPIPE, SIG_IGN);
  g_fence_queue = dispatch_queue_create("obelisk-vrbridge.fence", DISPATCH_QUEUE_SERIAL);
  logf_("vrhost link %s, mach name %s", g_vrhost, g_mach);
  vr_link();
  pthread_t t;
  pthread_create(&t, NULL, stats_main, NULL);
  if (obelisk) watch_obelisk();
  int s = listen_at(listen_path);
  if (s < 0) return 1;
  accept_main((void *)(intptr_t)s);
  return 1;
}
