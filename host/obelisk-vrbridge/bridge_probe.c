// SPDX-License-Identifier: GPL-3.0-or-later
// bridge_probe: acts as the guest runtime against obelisk-vrbridge (stream socket): one SVR_HELLO, then SVR_FRAME
// for a few seconds, printing what the host reports. usage: bridge_probe SOCKET [SECONDS]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "steamac_vr_guest.h"

static int xfer(int fd, const struct svr_hdr *q, void *rep, size_t cap) {
  if (send(fd, q, q->size, 0) != (ssize_t)q->size) return -1;
  struct svr_hdr *h = rep;
  size_t got = 0;
  while (got < sizeof *h) {
    ssize_t n = recv(fd, (char *)rep + got, sizeof *h - got, 0);
    if (n <= 0) return -1;
    got += n;
  }
  if (h->size > cap) return -1;
  while (got < h->size) {
    ssize_t n = recv(fd, (char *)rep + got, h->size - got, 0);
    if (n <= 0) return -1;
    got += n;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) return fprintf(stderr, "usage: %s SOCKET [SECONDS]\n", argv[0]), 2;
  int secs = argc > 2 ? atoi(argv[2]) : 3;
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  snprintf(a.sun_path, sizeof a.sun_path, "%s", argv[1]);
  if (connect(fd, (struct sockaddr *)&a, sizeof a)) return perror("connect"), 1;
  struct svr_hello_req hq = {{SVR_HELLO, sizeof hq}, SVR_VERSION, (uint32_t)getpid(), "bridge_probe"};
  struct svr_hello_rep hr;
  if (xfer(fd, &hq.h, &hr, sizeof hr)) return fprintf(stderr, "hello failed\n"), 1;
  printf("hello: version %u connected %u eye %ux%u refresh %.1f Hz\n", hr.version, hr.connected, hr.eye_width,
         hr.eye_height, hr.refresh_hz);
  uint64_t last = 0;
  int frames = 0;
  time_t end = time(NULL) + secs;
  while (time(NULL) < end) {
    struct svr_frame_req fq = {{SVR_FRAME, sizeof fq}, last};
    struct svr_frame_rep fr;
    if (xfer(fd, &fq.h, &fr, sizeof fr)) return fprintf(stderr, "frame failed\n"), 1;
    if (fr.frame_id != last) frames++;
    last = fr.frame_id;
    usleep(5000);
  }
  printf("frames: %d distinct frame ids in %d s (last %llu)\n", frames, secs, (unsigned long long)last);
  return 0;
}
