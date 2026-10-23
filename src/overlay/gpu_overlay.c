#define _GNU_SOURCE
#define VK_USE_PLATFORM_XLIB_KHR 1

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <vulkan/vulkan.h>

#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <getopt.h>
#include <glob.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "overlay_frag.h"
#include "overlay_vert.h"

#define MAX_CARDS 8
#define HISTORY 3600
#define TOP_METRICS 4
#define RANGE_COUNT 3
#define MAX_SWAP_IMAGES 8
#define MAX_VERTICES 65536
#define ENGINE_COUNT 5
#define MAX_DRM_CLIENTS 256
#define CLIENT_SCAN_INTERVAL 2.0

static const char *ENGINE_KEYS[ENGINE_COUNT] = {
    "drm-engine-render:",
    "drm-engine-copy:",
    "drm-engine-video:",
    "drm-engine-video-enhance:",
    "drm-engine-compute:"
};

static const char *ENGINE_NAMES[ENGINE_COUNT] = {
    "render", "copy", "video", "video_enhance", "compute"
};

static const char *RANGE_LABELS[RANGE_COUNT] = {"1m", "15min", "1h"};
static const int RANGE_SAMPLES[RANGE_COUNT] = {60, 900, 3600};


#ifndef ShapeInput
#define ShapeInput 2
#endif
#ifndef ShapeSet
#define ShapeSet 0
#endif
#ifndef Unsorted
#define Unsorted 0
#endif

typedef struct {
    float r, g, b, a;
} Color;

typedef struct {
    float x, y;
    float u, v;
    float r, g, b, a;
} Vertex;

static const Color C_BG = {0.035f, 0.045f, 0.065f, 0.62f};
static const Color C_BORDER = {0.18f, 0.23f, 0.31f, 0.85f};
static const Color C_TEXT = {0.90f, 0.93f, 0.97f, 1.0f};
static const Color C_DIM = {0.55f, 0.61f, 0.70f, 1.0f};
static const Color C_UNIT = {0.30f, 0.75f, 1.0f, 1.0f};
static const Color C_LINE = {0.22f, 0.28f, 0.37f, 0.7f};
static const Color C_ACCENT = {0.30f, 0.75f, 1.0f, 1.0f};

static const Color C_GREEN = {0.25f, 0.85f, 0.45f, 1.0f};
static const Color C_YELLOW = {0.95f, 0.90f, 0.25f, 1.0f};
static const Color C_ORANGE = {1.0f, 0.55f, 0.18f, 1.0f};
static const Color C_RED = {1.0f, 0.28f, 0.25f, 1.0f};
static const Color C_BAR_BG = {1.0f, 1.0f, 1.0f, 0.10f};

static volatile sig_atomic_t g_running = 1;
static volatile sig_atomic_t g_toggle = 0;

typedef struct {
    uint64_t id;
    uint64_t engine_ns[ENGINE_COUNT];
    uint64_t resident_local;
    uint64_t total_local;
} DrmClientSnapshot;

typedef struct {
    int index;
    int present;
    char name[16];
    char model[64];
    char drv[32];
    char dir[512];
    char hwmon[512];
    char gt0[512];
    char bdf[32];
    char pcie_dir[PATH_MAX];
    int has_energy;
    uint64_t energy;
    int has_rc6;
    uint64_t rc6_ms;
    double last_time;
    double power_w;
    double power_limit_w;
    double rc6_pct;
    int temp_c;
    int fan_rpm;
    int voltage_mv;
    int act_mhz;
    int max_mhz;
    int rp0_mhz;
    int req_mhz;
    int pcie_gen;
    int pcie_width;
    int throttled;
    char throttle[96];
    double hist[TOP_METRICS][HISTORY];
    double last_hist_time;
    double last_client_scan;
    DrmClientSnapshot clients[MAX_DRM_CLIENTS];
    int nclients;
    int clients_scanned;
    uint64_t mem_resident_bytes;
    uint64_t mem_total_bytes;
    double engine_pct[ENGINE_COUNT];
    double engine_busy_pct;
} Gpu;

typedef struct {
    unsigned char *pixels;
    int w, h;
    int cell_w, cell_h;
    int glyph_w, glyph_h;
    int advance;
} FontAtlas;

typedef struct {
    Vertex *verts;
    int count;
    int cap;
    FontAtlas *font;
} Canvas;

typedef struct {
    double interval;
    int device;
    const char *output;
    int x, y, w, h;
    int no_panel;
    int click_through;
    int managed;
    int draggable;
    int probe;
    int json;
    int selftest;
} Options;

typedef struct {
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice phys;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkSwapchainKHR swapchain;
    VkFormat format;
    VkColorSpaceKHR colorspace;
    VkExtent2D extent;
    VkCompositeAlphaFlagBitsKHR composite;
    int premult;
    uint32_t image_count;
    VkImage images[MAX_SWAP_IMAGES];
    VkImageView views[MAX_SWAP_IMAGES];
    VkFramebuffer framebuffers[MAX_SWAP_IMAGES];
    VkExtent2D ss_extent;
    VkImage ss_image;
    VkDeviceMemory ss_mem;
    VkImageView ss_view;
    VkFramebuffer ss_framebuffer;
    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkSemaphore sem_acquire;
    VkSemaphore sem_render;
    VkFence fence;
    VkBuffer vbo;
    VkDeviceMemory vbo_mem;
    Vertex *vbo_ptr;
    VkImage atlas_image;
    VkDeviceMemory atlas_mem;
    VkImageView atlas_view;
    VkSampler sampler;
    int ready;
} VkCtx;

typedef struct {
    int x0, y0, x1, y1;
} HitRect;

typedef struct {
    Gpu gpus[MAX_CARDS];
    int ngpus;
    Options opt;
    FontAtlas font;
    Canvas canvas;
    Display *dpy;
    int screen;
    Window root;
    Window win;
    Colormap cmap;
    int depth;
    Visual *visual;
    int x, y, w, h;
    int hidden;
    int dragging;
    int tab;
    int graph_metric;
    int graph_range;
    int range_open;
    int range_owner;
    HitRect tab_hits[MAX_CARDS * 2];
    int tab_hit_tabs[MAX_CARDS * 2];
    int tab_hit_count;
    HitRect metric_hits[MAX_CARDS * TOP_METRICS];
    int metric_hit_metrics[MAX_CARDS * TOP_METRICS];
    int metric_hit_count;
    HitRect range_toggle_hits[MAX_CARDS];
    int range_toggle_gpus[MAX_CARDS];
    int range_toggle_count;
    HitRect range_hits[RANGE_COUNT];
    int range_hit_ranges[RANGE_COUNT];
    int range_hit_count;
    int drag_root_x, drag_root_y;
    int drag_win_x, drag_win_y;
    VkCtx vk;
} App;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "gpu_overlay: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void vk_die(VkResult r, const char *what)
{
    die("%s failed (VkResult %d)", what, (int)r);
}

#define VKOK(x, what) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) vk_die(r_, what); } while (0)

static int read_text(const char *path, char *buf, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    if (!fgets(buf, (int)n, f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    size_t len = strlen(buf);
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' '))
        buf[--len] = '\0';
    return 1;
}

static int read_u64(const char *path, uint64_t *out)
{
    char buf[128];
    if (!read_text(path, buf, sizeof(buf)))
        return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(buf, &end, 10);
    if (errno || end == buf)
        return 0;
    *out = (uint64_t)v;
    return 1;
}

static int read_scaled(const char *path, double scale, double *out)
{
    char buf[128];
    if (!read_text(path, buf, sizeof(buf)))
        return 0;
    char *end = NULL;
    errno = 0;
    double v = strtod(buf, &end);
    if (errno || end == buf)
        return 0;
    *out = v * scale;
    return 1;
}

static int read_int_scaled(const char *path, double scale, int *out)
{
    double v;
    if (!read_scaled(path, scale, &v))
        return 0;
    *out = (int)lround(v);
    return 1;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s)
        return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_size_bytes(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s)
        return 0;
    while (*end == ' ' || *end == '\t')
        end++;
    uint64_t mult = 1;
    if (strncmp(end, "KiB", 3) == 0)
        mult = 1024ULL;
    else if (strncmp(end, "MiB", 3) == 0)
        mult = 1024ULL * 1024ULL;
    else if (strncmp(end, "GiB", 3) == 0)
        mult = 1024ULL * 1024ULL * 1024ULL;
    else if (*end && *end != 'B')
        return 0;
    if (v > UINT64_MAX / mult)
        return 0;
    *out = (uint64_t)v * mult;
    return 1;
}

static int pid_name(const char *s)
{
    if (!*s)
        return 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (!isdigit(*p))
            return 0;
    }
    return 1;
}

static const char *proc_root(void)
{
    const char *root = getenv("GPU_OVERLAY_PROC");
    return (root && *root) ? root : "/proc";
}

static const char *engine_key(const char *line)
{
    for (int i = 0; i < ENGINE_COUNT; i++) {
        size_t n = strlen(ENGINE_KEYS[i]);
        if (strncmp(line, ENGINE_KEYS[i], n) == 0)
            return ENGINE_KEYS[i];
    }
    return NULL;
}

static int engine_from_key(const char *key)
{
    for (int i = 0; i < ENGINE_COUNT; i++) {
        if (strcmp(key, ENGINE_KEYS[i]) == 0)
            return i;
    }
    return -1;
}

static int read_fdinfo_client(const char *path, const char *pdev, DrmClientSnapshot *out)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    char line_pdev[64] = {0};
    int has_pdev = 0;
    int has_id = 0;
    memset(out, 0, sizeof(*out));

    while ((n = getline(&line, &cap, f)) >= 0) {
        if (strncmp(line, "drm-pdev:", 9) == 0) {
            char *v = line + 9;
            while (*v == ' ' || *v == '\t')
                v++;
            size_t len = strlen(v);
            while (len && (v[len - 1] == '\n' || v[len - 1] == '\r' ||
                           v[len - 1] == ' ' || v[len - 1] == '\t'))
                v[--len] = '\0';
            snprintf(line_pdev, sizeof(line_pdev), "%s", v);
            has_pdev = 1;
        } else if (strncmp(line, "drm-client-id:", 14) == 0) {
            has_id = parse_u64(line + 14, &out->id);
        } else if (strncmp(line, "drm-resident-local0:", 20) == 0) {
            parse_size_bytes(line + 20, &out->resident_local);
        } else if (strncmp(line, "drm-total-local0:", 17) == 0) {
            parse_size_bytes(line + 17, &out->total_local);
        } else {
            const char *key = engine_key(line);
            if (key) {
                int e = engine_from_key(key);
                if (e >= 0)
                    parse_u64(line + strlen(key), &out->engine_ns[e]);
            }
        }
    }
    free(line);
    fclose(f);

    if (!has_pdev || !has_id || strcmp(line_pdev, pdev) != 0)
        return 0;
    return 1;
}

static void gpu_scan_clients(Gpu *g, double now)
{
    if (!g->present || !g->bdf[0])
        return;
    if (g->last_client_scan > 0.0 && now - g->last_client_scan < CLIENT_SCAN_INTERVAL)
        return;

    DrmClientSnapshot next[MAX_DRM_CLIENTS];
    int nnext = 0;
    uint64_t engine_delta[ENGINE_COUNT] = {0};
    uint64_t mem_resident = 0;
    int have_prev = g->last_client_scan > 0.0;
    double dt = have_prev ? now - g->last_client_scan : 0.0;

    const char *root = proc_root();
    DIR *proc = opendir(root);
    if (!proc) {
        g->clients_scanned = 0;
        return;
    }

    struct dirent *de;
    while ((de = readdir(proc)) != NULL) {
        if (!pid_name(de->d_name))
            continue;
        char fddir[1200];
        snprintf(fddir, sizeof(fddir), "%s/%s/fdinfo", root, de->d_name);
        DIR *fd = opendir(fddir);
        if (!fd)
            continue;

        struct dirent *fe;
        while ((fe = readdir(fd)) != NULL) {
            if (!pid_name(fe->d_name))
                continue;
            char path[1600];
            snprintf(path, sizeof(path), "%s/%s", fddir, fe->d_name);
            DrmClientSnapshot tc;
            if (!read_fdinfo_client(path, g->bdf, &tc))
                continue;

            int duplicate = 0;
            for (int i = 0; i < nnext; i++) {
                if (next[i].id == tc.id) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate)
                continue;

            int old = -1;
            for (int i = 0; i < g->nclients; i++) {
                if (g->clients[i].id == tc.id) {
                    old = i;
                    break;
                }
            }
            if (old >= 0) {
                for (int e = 0; e < ENGINE_COUNT; e++) {
                    if (tc.engine_ns[e] >= g->clients[old].engine_ns[e])
                        engine_delta[e] += tc.engine_ns[e] - g->clients[old].engine_ns[e];
                }
            }

            mem_resident += tc.resident_local;
            if (nnext < MAX_DRM_CLIENTS)
                next[nnext++] = tc;
        }
        closedir(fd);
    }
    closedir(proc);

    memcpy(g->clients, next, sizeof(next[0]) * (size_t)nnext);
    g->nclients = nnext;
    g->mem_resident_bytes = mem_resident;
    g->clients_scanned = 1;

    if (have_prev && dt > 0.0) {
        double max_pct = 0.0;
        for (int e = 0; e < ENGINE_COUNT; e++) {
            double pct = (double)engine_delta[e] / dt / 1e9 * 100.0;
            if (pct < 0.0)
                pct = 0.0;
            if (pct > 100.0)
                pct = 100.0;
            g->engine_pct[e] = pct;
            if (pct > max_pct)
                max_pct = pct;
        }
        g->engine_busy_pct = max_pct;
    } else {
        for (int e = 0; e < ENGINE_COUNT; e++)
            g->engine_pct[e] = NAN;
        g->engine_busy_pct = NAN;
    }
    g->last_client_scan = now;
}

static void gpu_query_total_memory(Gpu *g)
{
    if (!g->bdf[0])
        return;
    const char *bin = getenv("GPU_OVERLAY_XPU_SMI");
    if (!bin || !*bin)
        bin = "xpu-smi";
    char cmd[900];
    snprintf(cmd, sizeof(cmd),
             "timeout 5 %s --query-gpu=memory.total,memory.used,name "
             "--format=csv,noheader,nounits --id=%s",
             bin, g->bdf);
    FILE *p = popen(cmd, "r");
    if (!p) {
        fprintf(stderr, "gpu_overlay: xpu-smi unavailable for %s\n", g->bdf);
        return;
    }
    char line[256];
    int got = 0;
    if (fgets(line, sizeof(line), p)) {
        unsigned long long total_mib = 0;
        unsigned long long used_mib = 0;
        char model[128] = "";
        int fields = sscanf(line, "%llu, %llu, %127[^\n]", &total_mib, &used_mib, model);
        if (fields >= 1 && total_mib > 0) {
            g->mem_total_bytes = (uint64_t)total_mib * 1024ULL * 1024ULL;
            got = 1;
        }
        if (fields >= 3) {
            char *mp = model;
            while (*mp == ' ' || *mp == '\t')
                mp++;
            snprintf(g->model, sizeof(g->model), "%s", mp);
        }
    }
    if (pclose(p) != 0 && !got)
        fprintf(stderr, "gpu_overlay: xpu-smi total memory failed for %s\n", g->bdf);
    else if (!got)
        fprintf(stderr, "gpu_overlay: xpu-smi total memory missing for %s\n", g->bdf);
}

static double calc_power(uint64_t before, uint64_t after, double dt)
{
    if (dt <= 0.0 || after < before)
        return NAN;
    return (double)(after - before) / 1e6 / dt;
}

static double calc_rc6(uint64_t before, uint64_t after, double dt)
{
    if (dt <= 0.0 || after < before)
        return NAN;
    double pct = 100.0 * (double)(after - before) / 1000.0 / dt;
    if (pct < 0.0)
        pct = 0.0;
    if (pct > 100.0)
        pct = 100.0;
    return pct;
}

static const char *short_bdf(const char *bdf)
{
    size_t len = strlen(bdf);
    if (len >= 7)
        return bdf + len - 7;
    return bdf;
}

static int pcie_gen_from_speed(const char *speed)
{
    char *end = NULL;
    double gt = strtod(speed, &end);
    if (end == speed || gt <= 0.0)
        return -1;
    if (gt >= 64.0)
        return 6;
    if (gt >= 32.0)
        return 5;
    if (gt >= 16.0)
        return 4;
    if (gt >= 8.0)
        return 3;
    if (gt >= 5.0)
        return 2;
    return 1;
}

static int pci_domain_dir(const char *path)
{
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    return strncmp(base, "pci", 3) == 0;
}

static void gpu_resolve_pcie_link(Gpu *g, const char *dev)
{
    char cur[PATH_MAX];
    if (!realpath(dev, cur)) {
        snprintf(g->pcie_dir, sizeof(g->pcie_dir), "%s", g->dir);
        return;
    }

    char parent[PATH_MAX];
    char grand[PATH_MAX];
    for (;;) {
        snprintf(parent, sizeof(parent), "%s", cur);
        char *slash = strrchr(parent, '/');
        if (!slash || slash == parent)
            break;
        *slash = '\0';
        if (pci_domain_dir(parent))
            break;

        snprintf(grand, sizeof(grand), "%s", parent);
        char *slash2 = strrchr(grand, '/');
        if (!slash2 || slash2 == grand)
            break;
        *slash2 = '\0';
        if (pci_domain_dir(grand)) {
            snprintf(g->pcie_dir, sizeof(g->pcie_dir), "%s", cur);
            return;
        }
        snprintf(cur, sizeof(cur), "%s", parent);
    }
    snprintf(g->pcie_dir, sizeof(g->pcie_dir), "%s", cur);
}

static void gpu_read_throttle(Gpu *g)
{
    static const char *reasons[] = {
        "pl1", "pl2", "pl4", "prochot", "ratl", "status", "thermal", "vr_tdc", "vr_thermalert"
    };
    g->throttled = 0;
    g->throttle[0] = '\0';
    if (!g->gt0[0])
        return;
    for (size_t i = 0; i < sizeof(reasons) / sizeof(reasons[0]); i++) {
        char path[PATH_MAX + 64];
        char buf[64];
        snprintf(path, sizeof(path), "%s/throttle_reason_%s", g->gt0, reasons[i]);
        if (!read_text(path, buf, sizeof(buf)))
            continue;
        if (buf[0] == '0' || buf[0] == '\0')
            continue;
        size_t used = strlen(g->throttle);
        size_t need = strlen(reasons[i]) + (used ? 1 : 0);
        if (used + need >= sizeof(g->throttle))
            continue;
        if (used)
            strcat(g->throttle, ",");
        strcat(g->throttle, reasons[i]);
        g->throttled = 1;
    }
}

static int gpu_discover(Gpu *gpus, int max)
{
    const char *root = getenv("GPU_OVERLAY_SYSFS");
    if (!root || !*root)
        root = "/sys/class/drm";

    char pattern[600];
    snprintf(pattern, sizeof(pattern), "%s/card[0-9]*", root);
    glob_t gl;
    if (glob(pattern, 0, NULL, &gl) != 0)
        return 0;

    int n = 0;
    for (size_t i = 0; i < gl.gl_pathc && n < max; i++) {
        const char *dir = gl.gl_pathv[i];
        const char *base = strrchr(dir, '/');
        base = base ? base + 1 : dir;
        if (strncmp(base, "card", 4) != 0)
            continue;
        char *end = NULL;
        long idx = strtol(base + 4, &end, 10);
        if (end == base + 4 || *end)
            continue;

        char dev[600];
        snprintf(dev, sizeof(dev), "%s/device", dir);
        struct stat st;
        if (stat(dev, &st) != 0)
            continue;

        Gpu *g = &gpus[n];
        memset(g, 0, sizeof(*g));
        g->index = (int)idx;
        g->present = 1;
        g->temp_c = -1;
        g->fan_rpm = -1;
        g->voltage_mv = -1;
        g->act_mhz = -1;
        g->max_mhz = -1;
        g->rp0_mhz = -1;
        g->req_mhz = -1;
        g->pcie_gen = -1;
        g->pcie_width = -1;
        g->power_w = NAN;
        g->rc6_pct = NAN;
        g->power_limit_w = 0.0;
        for (int m = 0; m < TOP_METRICS; m++)
            for (int h = 0; h < HISTORY; h++)
                g->hist[m][h] = NAN;
        for (int e = 0; e < ENGINE_COUNT; e++)
            g->engine_pct[e] = NAN;
        g->engine_busy_pct = NAN;

        snprintf(g->name, sizeof(g->name), "card%ld", idx);
        snprintf(g->dir, sizeof(g->dir), "%s", dir);
        snprintf(g->gt0, sizeof(g->gt0), "%s/gt/gt0", dir);

        char hwpat[600];
        snprintf(hwpat, sizeof(hwpat), "%s/device/hwmon/hwmon*", dir);
        glob_t hg;
        if (glob(hwpat, 0, NULL, &hg) == 0) {
            if (hg.gl_pathc > 0)
                snprintf(g->hwmon, sizeof(g->hwmon), "%s", hg.gl_pathv[0]);
            globfree(&hg);
        }

        char link[600];
        ssize_t lr = readlink(dev, link, sizeof(link) - 1);
        if (lr > 0) {
            link[lr] = '\0';
            const char *slash = strrchr(link, '/');
            const char *base = slash ? slash + 1 : link;
            size_t n = strlen(base);
            if (n >= sizeof(g->bdf))
                n = sizeof(g->bdf) - 1;
            memcpy(g->bdf, base, n);
            g->bdf[n] = '\0';
        }
        gpu_resolve_pcie_link(g, dev);
        gpu_query_total_memory(g);
        if (!read_text("/proc/sys/kernel/osrelease", g->drv, sizeof(g->drv)))
            snprintf(g->drv, sizeof(g->drv), "%s", "unknown");
        n++;
    }
    globfree(&gl);
    return n;
}

static double gpu_mem_pct(const Gpu *g);

static void gpu_sample(Gpu *g, double now)
{
    if (!g->present)
        return;

    char path[PATH_MAX + 64];
    uint64_t energy = 0;
    uint64_t rc6 = 0;
    int has_energy = 0;
    int has_rc6 = 0;

    if (g->hwmon[0]) {
        snprintf(path, sizeof(path), "%s/energy1_input", g->hwmon);
        has_energy = read_u64(path, &energy);
        snprintf(path, sizeof(path), "%s/temp1_input", g->hwmon);
        if (!read_int_scaled(path, 0.001, &g->temp_c))
            g->temp_c = -1;
        snprintf(path, sizeof(path), "%s/fan1_input", g->hwmon);
        if (!read_int_scaled(path, 1.0, &g->fan_rpm))
            g->fan_rpm = -1;
        snprintf(path, sizeof(path), "%s/in0_input", g->hwmon);
        if (!read_int_scaled(path, 1.0, &g->voltage_mv))
            g->voltage_mv = -1;
        snprintf(path, sizeof(path), "%s/power1_max", g->hwmon);
        double limit;
        if (read_scaled(path, 1e-6, &limit))
            g->power_limit_w = limit;
        else
            g->power_limit_w = 0.0;
    }

    snprintf(path, sizeof(path), "%s/gt_act_freq_mhz", g->dir);
    if (!read_int_scaled(path, 1.0, &g->act_mhz)) {
        snprintf(path, sizeof(path), "%s/rps_act_freq_mhz", g->gt0);
        if (!read_int_scaled(path, 1.0, &g->act_mhz))
            g->act_mhz = -1;
    }
    snprintf(path, sizeof(path), "%s/gt_max_freq_mhz", g->dir);
    if (!read_int_scaled(path, 1.0, &g->max_mhz)) {
        snprintf(path, sizeof(path), "%s/rps_max_freq_mhz", g->gt0);
        if (!read_int_scaled(path, 1.0, &g->max_mhz))
            g->max_mhz = -1;
    }
    snprintf(path, sizeof(path), "%s/rps_RP0_freq_mhz", g->gt0);
    if (!read_int_scaled(path, 1.0, &g->rp0_mhz))
        g->rp0_mhz = -1;
    snprintf(path, sizeof(path), "%s/punit_req_freq_mhz", g->gt0);
    if (!read_int_scaled(path, 1.0, &g->req_mhz))
        g->req_mhz = -1;

    char speed[64];
    snprintf(path, sizeof(path), "%s/current_link_speed", g->pcie_dir);
    if (read_text(path, speed, sizeof(speed)))
        g->pcie_gen = pcie_gen_from_speed(speed);
    else
        g->pcie_gen = -1;
    snprintf(path, sizeof(path), "%s/current_link_width", g->pcie_dir);
    if (!read_int_scaled(path, 1.0, &g->pcie_width))
        g->pcie_width = -1;

    if (g->gt0[0]) {
        snprintf(path, sizeof(path), "%s/rc6_residency_ms", g->gt0);
        has_rc6 = read_u64(path, &rc6);
    }

    double dt = now - g->last_time;
    if (g->last_time > 0.0 && dt > 0.05) {
        g->power_w = (has_energy && g->has_energy) ? calc_power(g->energy, energy, dt) : NAN;
        g->rc6_pct = (has_rc6 && g->has_rc6) ? calc_rc6(g->rc6_ms, rc6, dt) : NAN;
    } else {
        g->power_w = NAN;
        g->rc6_pct = NAN;
    }

    g->has_energy = has_energy;
    if (has_energy)
        g->energy = energy;
    g->has_rc6 = has_rc6;
    if (has_rc6)
        g->rc6_ms = rc6;
    g->last_time = now;

    gpu_read_throttle(g);
    gpu_scan_clients(g, now);

    if (g->last_hist_time <= 0.0 || now - g->last_hist_time >= 1.0) {
        double values[TOP_METRICS];
        values[0] = g->power_w;
        values[1] = g->temp_c >= 0 ? (double)g->temp_c : NAN;
        values[2] = g->engine_busy_pct;
        values[3] = gpu_mem_pct(g);
        for (int m = 0; m < TOP_METRICS; m++) {
            memmove(&g->hist[m][0], &g->hist[m][1],
                    sizeof(g->hist[m][0]) * (HISTORY - 1));
            g->hist[m][HISTORY - 1] = values[m];
        }
        g->last_hist_time = now;
    }
}

static void gpu_update_all(App *app)
{
    double now = now_sec();
    for (int i = 0; i < app->ngpus; i++)
        gpu_sample(&app->gpus[i], now);
}

static void json_string(const char *s)
{
    if (!s || !*s) {
        fputs("null", stdout);
        return;
    }
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\')
            printf("\\%c", *p);
        else if (*p < 0x20)
            printf("\\u%04x", *p);
        else
            putchar(*p);
    }
    putchar('"');
}

static void json_double(double v)
{
    if (isnan(v))
        fputs("null", stdout);
    else
        printf("%.3f", v);
}

static void json_int(int v)
{
    if (v < 0)
        fputs("null", stdout);
    else
        printf("%d", v);
}

static void probe(App *app, int json)
{
    gpu_update_all(app);
    usleep(2100000);
    gpu_update_all(app);

    if (json) {
        putchar('[');
        for (int i = 0; i < app->ngpus; i++) {
            Gpu *g = &app->gpus[i];
            if (i)
                putchar(',');
            printf("{\"card\":");
            json_string(g->name);
            printf(",\"pci\":");
            json_string(g->bdf);
            printf(",\"power_w\":");
            json_double(g->power_w);
            printf(",\"power_limit_w\":");
            if (g->power_limit_w > 0.0)
                printf("%.3f", g->power_limit_w);
            else
                fputs("null", stdout);
            printf(",\"temp_c\":");
            json_int(g->temp_c);
            printf(",\"fan_rpm\":");
            json_int(g->fan_rpm);
            printf(",\"voltage_mv\":");
            json_int(g->voltage_mv);
            printf(",\"act_mhz\":");
            json_int(g->act_mhz);
            printf(",\"max_mhz\":");
            json_int(g->max_mhz);
            printf(",\"req_mhz\":");
            json_int(g->req_mhz);
            printf(",\"pcie_gen\":");
            json_int(g->pcie_gen);
            printf(",\"pcie_width\":");
            json_int(g->pcie_width);
            printf(",\"rc6_pct\":");
            json_double(g->rc6_pct);
            printf(",\"engine_busy_pct\":");
            json_double(g->engine_busy_pct);
            for (int e = 0; e < ENGINE_COUNT; e++) {
                printf(",\"engine_%s_pct\":", ENGINE_NAMES[e]);
                json_double(g->engine_pct[e]);
            }
            printf(",\"mem_resident_mb\":");
            printf("%.1f", (double)g->mem_resident_bytes / (1024.0 * 1024.0));
            printf(",\"mem_total_mb\":");
            printf("%.1f", (double)g->mem_total_bytes / (1024.0 * 1024.0));
            printf(",\"throttle\":");
            json_string(g->throttle);
            printf(",\"energy_uj\":");
            if (g->has_energy)
                printf("%llu", (unsigned long long)g->energy);
            else
                fputs("null", stdout);
            putchar('}');
        }
        puts("]");
        return;
    }

    for (int i = 0; i < app->ngpus; i++) {
        Gpu *g = &app->gpus[i];
        printf("%-6s %-8s ", g->name, g->bdf[0] ? short_bdf(g->bdf) : "-");
        if (isnan(g->power_w))
            printf("  --W");
        else
            printf("%5.1fW", g->power_w);
        if (g->power_limit_w > 0.0)
            printf("/%3.0fW", g->power_limit_w);
        if (g->temp_c >= 0)
            printf("  %3d C", g->temp_c);
        if (g->fan_rpm >= 0)
            printf("  %4d RPM", g->fan_rpm);
        if (g->act_mhz >= 0)
            printf("  %4d/%4d MHz", g->act_mhz, g->max_mhz);
        if (g->pcie_gen > 0 && g->pcie_width > 0)
            printf("  Gen%d x%d", g->pcie_gen, g->pcie_width);
        if (!isnan(g->rc6_pct))
            printf("  RC6 %3.0f%%", g->rc6_pct);
        if (!isnan(g->engine_busy_pct))
            printf("  ENG %3.0f%%", g->engine_busy_pct);
        printf("  MEM %.0f MB", (double)g->mem_resident_bytes / (1024.0 * 1024.0));
        if (g->throttled)
            printf("  THR %s", g->throttle);
        putchar('\n');
    }
}

static int selftest(void)
{
    if (!(fabs(calc_power(1000000, 3000000, 2.0) - 1.0) < 1e-9)) {
        fprintf(stderr, "selftest: power\n");
        return 1;
    }
    if (!(fabs(calc_rc6(1000, 2000, 2.0) - 50.0) < 1e-9)) {
        fprintf(stderr, "selftest: rc6\n");
        return 1;
    }
    if (!isnan(calc_power(3000000, 1000000, 2.0))) {
        fprintf(stderr, "selftest: counter reset\n");
        return 1;
    }
    if (!isnan(calc_power(1, 2, 0.0))) {
        fprintf(stderr, "selftest: zero dt\n");
        return 1;
    }
    if (fabs(calc_power(5, 5, 1.0)) > 1e-9) {
        fprintf(stderr, "selftest: unchanged counter\n");
        return 1;
    }
    if (!isnan(calc_rc6(2, 1, 1.0))) {
        fprintf(stderr, "selftest: rc6 reset\n");
        return 1;
    }
    if (!isnan(calc_rc6(1, 2, 0.0))) {
        fprintf(stderr, "selftest: rc6 zero dt\n");
        return 1;
    }
    uint64_t sz = 0;
    if (!parse_size_bytes("4 MiB", &sz) || sz != 4ULL * 1024ULL * 1024ULL) {
        fprintf(stderr, "selftest: size MiB\n");
        return 1;
    }
    if (!parse_size_bytes("30180 KiB", &sz) || sz != 30180ULL * 1024ULL) {
        fprintf(stderr, "selftest: size KiB\n");
        return 1;
    }
    if (!parse_size_bytes("0", &sz) || sz != 0) {
        fprintf(stderr, "selftest: size zero\n");
        return 1;
    }
    puts("selftest ok");
    return 0;
}

static int font_atlas_create(FontAtlas *fa, Display *dpy, int screen)
{
    static const char *names[] = {
        "-misc-fixed-medium-r-normal--20-200-75-75-c-100-iso10646-1",
        "-misc-fixed-medium-r-normal--18-120-100-100-c-90-iso10646-1",
        "10x20",
        "9x15",
        "fixed",
        NULL
    };

    XFontStruct *font = NULL;
    const char *picked = NULL;
    for (int i = 0; names[i]; i++) {
        font = XLoadQueryFont(dpy, names[i]);
        if (font) {
            picked = names[i];
            break;
        }
    }
    if (!font)
        return 0;

    int gw = font->max_bounds.width;
    if (gw <= 0)
        gw = font->min_bounds.width;
    if (gw <= 0)
        gw = 8;
    int gh = font->ascent + font->descent;
    if (gh <= 0)
        gh = 16;

    int cell_w = gw + 2;
    int cell_h = gh + 2;
    int cols = 16;
    int rows = 8;
    int aw = cols * cell_w;
    int ah = rows * cell_h;

    Pixmap pix = XCreatePixmap(dpy, RootWindow(dpy, screen), (unsigned)aw, (unsigned)ah, 1);
    if (!pix) {
        XFreeFont(dpy, font);
        return 0;
    }
    GC gc = XCreateGC(dpy, pix, 0, NULL);
    XSetFont(dpy, gc, font->fid);
    XSetForeground(dpy, gc, 0);
    XFillRectangle(dpy, pix, gc, 0, 0, (unsigned)aw, (unsigned)ah);
    XSetForeground(dpy, gc, 1);

    for (int c = 32; c < 127; c++) {
        int idx = c - 32;
        int col = idx % cols;
        int row = idx / cols;
        char ch = (char)c;
        int x = col * cell_w + 1;
        int y = row * cell_h + 1 + font->ascent;
        XDrawString(dpy, pix, gc, x, y, &ch, 1);
    }
    XFlush(dpy);

    XImage *img = XGetImage(dpy, pix, 0, 0, (unsigned)aw, (unsigned)ah, AllPlanes, ZPixmap);
    if (!img) {
        XFreeGC(dpy, gc);
        XFreePixmap(dpy, pix);
        XFreeFont(dpy, font);
        return 0;
    }

    unsigned char *pixels = calloc((size_t)aw * (size_t)ah, 1);
    if (!pixels) {
        XDestroyImage(img);
        XFreeGC(dpy, gc);
        XFreePixmap(dpy, pix);
        XFreeFont(dpy, font);
        return 0;
    }
    for (int y = 0; y < ah; y++) {
        for (int x = 0; x < aw; x++) {
            if (XGetPixel(img, x, y))
                pixels[(size_t)y * (size_t)aw + (size_t)x] = 255;
        }
    }

    XDestroyImage(img);
    XFreeGC(dpy, gc);
    XFreePixmap(dpy, pix);
    XFreeFont(dpy, font);

    fa->pixels = pixels;
    fa->w = aw;
    fa->h = ah;
    fa->cell_w = cell_w;
    fa->cell_h = cell_h;
    fa->glyph_w = gw;
    fa->glyph_h = gh;
    fa->advance = gw;
    fprintf(stderr, "gpu_overlay: font %s (%dx%d)\n", picked, gw, gh);
    return 1;
}

static void font_atlas_destroy(FontAtlas *fa)
{
    free(fa->pixels);
    memset(fa, 0, sizeof(*fa));
}

static int canvas_init(Canvas *c, FontAtlas *font)
{
    c->verts = calloc(MAX_VERTICES, sizeof(Vertex));
    if (!c->verts)
        return 0;
    c->cap = MAX_VERTICES;
    c->count = 0;
    c->font = font;
    return 1;
}

static void canvas_destroy(Canvas *c)
{
    free(c->verts);
    memset(c, 0, sizeof(*c));
}

static void canvas_reset(Canvas *c)
{
    c->count = 0;
}

static void canvas_vertex(Canvas *c, float x, float y, float u, float v, Color col)
{
    if (c->count >= c->cap)
        return;
    Vertex *vt = &c->verts[c->count++];
    vt->x = x;
    vt->y = y;
    vt->u = u;
    vt->v = v;
    vt->r = col.r;
    vt->g = col.g;
    vt->b = col.b;
    vt->a = col.a;
}

static void canvas_quad(Canvas *c, float x0, float y0, float x1, float y1,
                        float u0, float v0, float u1, float v1, Color col)
{
    canvas_vertex(c, x0, y0, u0, v0, col);
    canvas_vertex(c, x1, y0, u1, v0, col);
    canvas_vertex(c, x1, y1, u1, v1, col);
    canvas_vertex(c, x0, y0, u0, v0, col);
    canvas_vertex(c, x1, y1, u1, v1, col);
    canvas_vertex(c, x0, y1, u0, v1, col);
}

static void canvas_rect(Canvas *c, float x, float y, float w, float h, Color col)
{
    canvas_quad(c, x, y, x + w, y + h, -1.0f, -1.0f, -1.0f, -1.0f, col);
}

static int canvas_text(Canvas *c, int x, int y, int scale, Color col, const char *text)
{
    FontAtlas *f = c->font;
    if (!f || !f->pixels)
        return 0;
    int start = x;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        int code = *p;
        if (code < 32 || code >= 127)
            code = '?';
        int idx = code - 32;
        int col_i = idx % 16;
        int row_i = idx / 16;
        float u0 = (float)(col_i * f->cell_w + 1) / (float)f->w;
        float v0 = (float)(row_i * f->cell_h + 1) / (float)f->h;
        float u1 = (float)(col_i * f->cell_w + 1 + f->glyph_w) / (float)f->w;
        float v1 = (float)(row_i * f->cell_h + 1 + f->glyph_h) / (float)f->h;
        float x0 = (float)x;
        float y0 = (float)y;
        float x1 = x0 + (float)(f->glyph_w * scale);
        float y1 = y0 + (float)(f->glyph_h * scale);
        canvas_quad(c, x0, y0, x1, y1, u0, v0, u1, v1, col);
        x += f->advance * scale;
    }
    return x - start;
}

static int canvas_text_scaled(Canvas *c, int x, int y, float scale, Color col, const char *text)
{
    FontAtlas *f = c->font;
    if (!f || !f->pixels)
        return 0;
    float cursor = (float)x;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        int code = *p;
        if (code < 32 || code >= 127)
            code = '?';
        int idx = code - 32;
        int col_i = idx % 16;
        int row_i = idx / 16;
        float u0 = (float)(col_i * f->cell_w + 1) / (float)f->w;
        float v0 = (float)(row_i * f->cell_h + 1) / (float)f->h;
        float u1 = (float)(col_i * f->cell_w + 1 + f->glyph_w) / (float)f->w;
        float v1 = (float)(row_i * f->cell_h + 1 + f->glyph_h) / (float)f->h;
        float x0 = cursor;
        float y0 = (float)y;
        float x1 = x0 + (float)f->glyph_w * scale;
        float y1 = y0 + (float)f->glyph_h * scale;
        canvas_quad(c, x0, y0, x1, y1, u0, v0, u1, v1, col);
        cursor += (float)f->advance * scale;
    }
    return (int)(cursor - (float)x);
}

static int canvas_textf(Canvas *c, int x, int y, int scale, Color col, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return canvas_text(c, x, y, scale, col, buf);
}

static void canvas_value_unit_colored(Canvas *c, int x, int y, const char *value,
                                      const char *unit, Color value_col, Color unit_col)
{
    int width = canvas_text(c, x, y, 1, value_col, value);
    int unit_h = (int)((float)c->font->glyph_h * 0.75f + 0.5f);
    int unit_y = y + c->font->glyph_h - unit_h;
    canvas_text_scaled(c, x + width, unit_y, 0.75f, unit_col, unit);
}

static void canvas_value_unit(Canvas *c, int x, int y, const char *value, const char *unit)
{
    canvas_value_unit_colored(c, x, y, value, unit, C_TEXT, C_UNIT);
}

static void canvas_graph(Canvas *c, int x, int y, int w, int h,
                         const double *hist, int n, double maximum, Color col)
{
    if (w < 4 || h < 4)
        return;
    canvas_rect(c, (float)x, (float)y, (float)w, (float)h, C_BAR_BG);
    for (int i = 1; i < 4; i++) {
        float gy = (float)y + (float)h * (float)i / 4.0f;
        canvas_rect(c, (float)x, gy, (float)w, 1.0f, C_LINE);
    }
    if (!(maximum > 0.0))
        maximum = 200.0;

    if (n > w) {
        for (int px = 0; px < w; px++) {
            int start = px * n / w;
            int end = (px + 1) * n / w;
            double sum = 0.0;
            int used = 0;
            for (int i = start; i < end; i++) {
                double v = hist[i];
                if (isnan(v) || v < 0.0)
                    continue;
                sum += v;
                used++;
            }
            if (used == 0)
                continue;
            double v = sum / used;
            double f = v / maximum;
            if (f > 1.0)
                f = 1.0;
            float bh = (float)h * (float)f;
            if (bh < 1.0f && v > 0.0)
                bh = 1.0f;
            float by = (float)y + (float)h - bh;
            canvas_rect(c, (float)(x + px), by, 1.0f, bh, col);
        }
        return;
    }

    float bar_w = (float)w / (float)n;
    float draw_w = bar_w > 2.0f ? bar_w - 1.0f : bar_w;
    for (int i = 0; i < n; i++) {
        double v = hist[i];
        if (isnan(v) || v < 0.0)
            continue;
        double f = v / maximum;
        if (f > 1.0)
            f = 1.0;
        float bh = (float)h * (float)f;
        if (bh < 1.0f && v > 0.0)
            bh = 1.0f;
        float bx = (float)x + (float)i * bar_w;
        float by = (float)y + (float)h - bh;
        canvas_rect(c, bx, by, draw_w, bh, col);
    }
}

static Color ramp_color(double value, double limit)
{
    static const double stops[5] = {0.0, 0.35, 0.60, 0.85, 1.0};
    static const Color colors[5] = {
        C_TEXT, C_GREEN, C_YELLOW, C_ORANGE, C_RED
    };
    if (isnan(value) || !(limit > 0.0))
        return C_DIM;
    double f = value / limit;
    if (f < 0.0)
        f = 0.0;
    if (f > 1.0)
        f = 1.0;
    int i = 0;
    while (i < 4 && f > stops[i + 1])
        i++;
    double span = stops[i + 1] - stops[i];
    float t = span > 0.0 ? (float)((f - stops[i]) / span) : 0.0f;
    Color c;
    c.r = colors[i].r + (colors[i + 1].r - colors[i].r) * t;
    c.g = colors[i].g + (colors[i + 1].g - colors[i].g) * t;
    c.b = colors[i].b + (colors[i + 1].b - colors[i].b) * t;
    c.a = 1.0f;
    return c;
}

static double gpu_mem_pct(const Gpu *g)
{
    if (!g->clients_scanned || g->mem_total_bytes == 0)
        return NAN;
    return 100.0 * (double)g->mem_resident_bytes / (double)g->mem_total_bytes;
}

static const char *model_short(const char *model, char *buf, size_t n)
{
    if (!model || !*model) {
        snprintf(buf, n, "Arc A770");
        return buf;
    }
    const char *p = strstr(model, "Arc");
    if (!p)
        p = model;
    size_t o = 0;
    while (*p && o + 1 < n) {
        if (strncmp(p, "(TM)", 4) == 0) {
            p += 4;
            continue;
        }
        if (strncmp(p, " Graphics", 9) == 0)
            break;
        buf[o++] = *p++;
    }
    while (o > 0 && buf[o - 1] == ' ')
        o--;
    buf[o] = '\0';
    return buf;
}

static void draw_range_dropdown(App *app, Gpu *g, int x, int y)
{
    Canvas *c = &app->canvas;
    FontAtlas *f = &app->font;
    const char *label = RANGE_LABELS[app->graph_range];
    int box_w = 5 * f->advance + 26;
    int box_h = f->glyph_h + 6;
    int menu_y = y + box_h + 2;
    int item_h = f->glyph_h + 4;

    canvas_rect(c, (float)x, (float)y, (float)box_w, (float)box_h, C_BAR_BG);
    canvas_rect(c, (float)x, (float)y, (float)box_w, 1.0f, C_LINE);
    canvas_rect(c, (float)x, (float)(y + box_h - 1), (float)box_w, 1.0f, C_LINE);
    canvas_rect(c, (float)x, (float)y, 1.0f, (float)box_h, C_LINE);
    canvas_rect(c, (float)(x + box_w - 1), (float)y, 1.0f, (float)box_h, C_LINE);
    canvas_text(c, x + 6, y + 3, 1, C_TEXT, label);
    canvas_text(c, x + box_w - 14, y + 3, 1, C_DIM, "v");

    if (app->range_toggle_count < MAX_CARDS) {
        HitRect *hit = &app->range_toggle_hits[app->range_toggle_count];
        hit->x0 = x;
        hit->y0 = y;
        hit->x1 = x + box_w;
        hit->y1 = y + box_h;
        app->range_toggle_gpus[app->range_toggle_count] = g->index;
        app->range_toggle_count++;
    }

    if (!app->range_open || app->range_owner != g->index)
        return;

    int menu_h = item_h * RANGE_COUNT;
    canvas_rect(c, (float)x, (float)menu_y, (float)box_w, (float)menu_h, C_BG);
    canvas_rect(c, (float)x, (float)menu_y, (float)box_w, 1.0f, C_LINE);
    canvas_rect(c, (float)x, (float)(menu_y + menu_h - 1), (float)box_w, 1.0f, C_LINE);
    canvas_rect(c, (float)x, (float)menu_y, 1.0f, (float)menu_h, C_LINE);
    canvas_rect(c, (float)(x + box_w - 1), (float)menu_y, 1.0f, (float)menu_h, C_LINE);

    for (int i = 0; i < RANGE_COUNT; i++) {
        int iy = menu_y + i * item_h;
        Color col = app->graph_range == i ? C_ACCENT : C_TEXT;
        canvas_text(c, x + 6, iy + 2, 1, col, RANGE_LABELS[i]);
        if (app->range_hit_count < RANGE_COUNT) {
            HitRect *hit = &app->range_hits[app->range_hit_count];
            hit->x0 = x;
            hit->y0 = iy;
            hit->x1 = x + box_w;
            hit->y1 = iy + item_h;
            app->range_hit_ranges[app->range_hit_count] = i;
            app->range_hit_count++;
        }
    }
}

static void draw_gpu_block(App *app, Gpu *g, int x, int y, int w, int h)
{
    Canvas *c = &app->canvas;
    FontAtlas *f = &app->font;
    int tab = app->tab;
    int line = f->glyph_h + 8;
    int gy = y;
    double power_max = g->power_limit_w > 0.0 ? g->power_limit_w : 200.0;
    int colw = w / 4;

    canvas_textf(c, x, gy, 3, C_ACCENT, "GPU%d", g->index);
    char model[64];
    canvas_textf(c, x + 136, gy + 6, 1, C_ACCENT, "%s",
                 model_short(g->model, model, sizeof(model)));
    if (g->pcie_gen > 0 && g->pcie_width > 0)
        canvas_textf(c, x + 136, gy + 32, 1, C_DIM, "Gen%d x%d",
                     g->pcie_gen, g->pcie_width);
    gy += 66;

    static const char *top_labels[TOP_METRICS] = {"PWR", "TMP", "ENG", "MEM"};
    static const char *top_units[TOP_METRICS] = {"W", "C", "%", "%"};
    int top_y = gy;
    for (int i = 0; i < TOP_METRICS; i++) {
        Color label_col = app->graph_metric == i ? C_ACCENT : C_DIM;
        canvas_text(c, x + i * colw, top_y, 1, label_col, top_labels[i]);
    }
    gy += line;

    double values[TOP_METRICS];
    double limits[TOP_METRICS];
    values[0] = g->power_w;
    values[1] = g->temp_c >= 0 ? (double)g->temp_c : NAN;
    values[2] = g->engine_busy_pct;
    values[3] = gpu_mem_pct(g);
    limits[0] = power_max;
    limits[1] = 100.0;
    limits[2] = 100.0;
    limits[3] = 100.0;

    for (int i = 0; i < TOP_METRICS; i++) {
        int vx = x + i * colw;
        if (isnan(values[i])) {
            canvas_text(c, vx, gy, 1, C_DIM, "--");
        } else {
            char value[32];
            snprintf(value, sizeof(value), "%.0f", values[i]);
            Color col = ramp_color(values[i], limits[i]);
            canvas_value_unit_colored(c, vx, gy, value, top_units[i], col, col);
        }
        if (app->metric_hit_count < MAX_CARDS * TOP_METRICS) {
            HitRect *hit = &app->metric_hits[app->metric_hit_count];
            hit->x0 = vx - 2;
            hit->y0 = top_y - 2;
            hit->x1 = vx + colw - 6;
            hit->y1 = gy + f->glyph_h + 2;
            app->metric_hit_metrics[app->metric_hit_count] = i;
            app->metric_hit_count++;
        }
    }
    gy += line + 4;

    canvas_rect(c, x, (float)gy, (float)w, 1.0f, C_LINE);
    gy += 5;

    int tab_x[2] = {x, x + 105};
    static const char *tab_names[2] = {"GRAPH", "DETAILS"};
    for (int i = 0; i < 2; i++) {
        canvas_text(c, tab_x[i], gy, 1, tab == i ? C_ACCENT : C_DIM, tab_names[i]);
        if (tab == i)
            canvas_rect(c, (float)tab_x[i], (float)(gy + f->glyph_h + 1),
                        (float)(strlen(tab_names[i]) * f->advance), 1.0f, C_ACCENT);
        if (app->tab_hit_count < MAX_CARDS * 2) {
            HitRect *hit = &app->tab_hits[app->tab_hit_count];
            hit->x0 = tab_x[i] - 2;
            hit->y0 = gy - 2;
            hit->x1 = tab_x[i] + (int)(strlen(tab_names[i]) * f->advance) + 2;
            hit->y1 = gy + f->glyph_h + 4;
            app->tab_hit_tabs[app->tab_hit_count] = i;
            app->tab_hit_count++;
        }
    }
    gy += line + 14;

    if (tab == 0) {
        double limit = app->graph_metric == 0 ? power_max : 100.0;
        char max_label[32];
        if (app->graph_metric == 0)
            snprintf(max_label, sizeof(max_label), "%.0fW", limit);
        else if (app->graph_metric == 1)
            snprintf(max_label, sizeof(max_label), "%.0fC", limit);
        else
            snprintf(max_label, sizeof(max_label), "%.0f%%", limit);

        int max_x = x + w - (int)strlen(max_label) * f->advance;
        canvas_text(c, max_x, gy, 1, C_DIM, max_label);

        int row_y = gy;
        gy += line;

        int count = RANGE_SAMPLES[app->graph_range];
        if (count > HISTORY)
            count = HISTORY;
        const double *hist = &g->hist[app->graph_metric][HISTORY - count];

        int graph_h = y + h - gy - 4;
        if (graph_h > 135)
            graph_h = 135;
        if (graph_h < 30)
            graph_h = 30;
        canvas_graph(c, x, gy, w, graph_h, hist, count, limit, C_ACCENT);
        draw_range_dropdown(app, g, x, row_y);
        return;
    }

    char value[32];
    canvas_text(c, x, gy, 1, C_DIM, "LIM");
    if (g->power_limit_w > 0.0) {
        snprintf(value, sizeof(value), "%.0f", g->power_limit_w);
        canvas_value_unit(c, x + 42, gy, value, "W");
    } else {
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    }
    if (g->voltage_mv < 0) {
        canvas_text(c, x + 120, gy, 1, C_DIM, "--");
    } else {
        snprintf(value, sizeof(value), "%.3f", g->voltage_mv / 1000.0);
        canvas_value_unit(c, x + 120, gy, value, "V");
    }
    gy += line;

    canvas_text(c, x, gy, 1, C_DIM, "CLK");
    if (g->act_mhz < 0) {
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    } else {
        int vmax = g->rp0_mhz >= 0 ? g->rp0_mhz : g->max_mhz;
        snprintf(value, sizeof(value), "%.2f", g->act_mhz / 1000.0);
        canvas_value_unit(c, x + 42, gy, value, "GHz");
        canvas_text(c, x + 120, gy, 1, C_DIM, "MAX");
        if (vmax >= 0) {
            snprintf(value, sizeof(value), "%.2f", vmax / 1000.0);
            canvas_value_unit(c, x + 156, gy, value, "GHz");
        } else {
            canvas_text(c, x + 156, gy, 1, C_DIM, "--");
        }
    }
    gy += line;

    canvas_text(c, x, gy, 1, C_DIM, "REQ");
    if (g->req_mhz < 0) {
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    } else {
        snprintf(value, sizeof(value), "%.2f", g->req_mhz / 1000.0);
        canvas_value_unit(c, x + 42, gy, value, "GHz");
    }
    canvas_text(c, x + 120, gy, 1, C_DIM, "RC6");
    if (isnan(g->rc6_pct)) {
        canvas_text(c, x + 156, gy, 1, C_DIM, "--");
    } else {
        snprintf(value, sizeof(value), "%.0f", g->rc6_pct);
        canvas_value_unit(c, x + 156, gy, value, "%");
    }
    gy += line;

    canvas_text(c, x, gy, 1, C_DIM, "MEM");
    if (g->clients_scanned) {
        double used_gib = (double)g->mem_resident_bytes / (1024.0 * 1024.0 * 1024.0);
        snprintf(value, sizeof(value), "%.2f", used_gib);
        canvas_value_unit(c, x + 42, gy, value, "G");
        if (g->mem_total_bytes > 0) {
            double total_gib = (double)g->mem_total_bytes / (1024.0 * 1024.0 * 1024.0);
            canvas_text(c, x + 120, gy, 1, C_DIM, "TOT");
            snprintf(value, sizeof(value), "%.1f", total_gib);
            canvas_value_unit(c, x + 156, gy, value, "G");
        }
    } else {
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    }
    gy += line;

    canvas_text(c, x, gy, 1, C_DIM, "FAN");
    if (g->fan_rpm < 0) {
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    } else {
        snprintf(value, sizeof(value), "%d", g->fan_rpm);
        canvas_value_unit(c, x + 42, gy, value, "RPM");
    }
    gy += line;

    canvas_text(c, x, gy, 1, C_DIM, "DRV");
    if (!g->drv[0])
        canvas_text(c, x + 42, gy, 1, C_DIM, "--");
    else
        canvas_textf(c, x + 42, gy, 1, C_TEXT, "%.12s", g->drv);
    gy += line;

    if (g->throttled)
        canvas_textf(c, x, gy, 1, C_RED, "THR %.22s", g->throttle);
}

static void draw_scene(App *app)
{
    Canvas *c = &app->canvas;
    FontAtlas *f = &app->font;
    canvas_reset(c);
    app->tab_hit_count = 0;
    app->metric_hit_count = 0;
    app->range_toggle_count = 0;
    app->range_hit_count = 0;

    int W = app->w;
    int H = app->h;
    int pad = 12;

    if (!app->opt.no_panel) {
        canvas_rect(c, 0, 0, (float)W, (float)H, C_BORDER);
        canvas_rect(c, 1, 1, (float)(W - 2), (float)(H - 2), C_BG);
    }

    int x = pad;
    int y = pad;

    int footer_h = 22;
    int avail = H - y - pad - footer_h;
    int block_h = app->ngpus > 0 ? avail / app->ngpus : avail;
    if (block_h < 160)
        block_h = 160;

    for (int i = 0; i < app->ngpus; i++) {
        draw_gpu_block(app, &app->gpus[i], x, y, W - 2 * pad, block_h);
        y += block_h;
    }
    if (app->ngpus == 0)
        canvas_text(c, x, y, 1, C_RED, "NO GPU TELEMETRY");

    double total = 0.0;
    int have_total = 0;
    uint64_t mem_used = 0;
    uint64_t mem_capacity = 0;
    int have_mem = 0;
    for (int i = 0; i < app->ngpus; i++) {
        if (!isnan(app->gpus[i].power_w)) {
            total += app->gpus[i].power_w;
            have_total = 1;
        }
        if (app->gpus[i].clients_scanned && app->gpus[i].mem_total_bytes > 0) {
            mem_used += app->gpus[i].mem_resident_bytes;
            mem_capacity += app->gpus[i].mem_total_bytes;
            have_mem = 1;
        }
    }
    int fy = H - pad - footer_h + 2;
    canvas_rect(c, x, (float)fy, (float)(W - 2 * pad), 1.0f, C_LINE);
    char footer[96];
    Color footer_col = C_ACCENT;
    if (have_total && have_mem) {
        double used_gib = (double)mem_used / (1024.0 * 1024.0 * 1024.0);
        double cap_gib = (double)mem_capacity / (1024.0 * 1024.0 * 1024.0);
        snprintf(footer, sizeof(footer), "TOTAL %.0fW | %.0fG/%.0fG",
                 total, used_gib, cap_gib);
    } else if (have_total) {
        snprintf(footer, sizeof(footer), "TOTAL %.0fW", total);
    } else {
        snprintf(footer, sizeof(footer), "TOTAL --");
        footer_col = C_DIM;
    }
    int footer_x = (W - (int)strlen(footer) * f->advance) / 2;
    if (footer_x < 0)
        footer_x = 0;
    canvas_text(c, footer_x, fy + 4, 1, footer_col, footer);
}

static int parse_primary_output(const char *want, int *ox, int *oy, int *ow, int *oh)
{
    FILE *p = popen("xrandr --current 2>/dev/null", "r");
    if (!p)
        return 0;
    char line[512];
    int found = 0;
    int fx = 0, fy = 0, fw = 0, fh = 0;
    int have_first = 0;
    int first_x = 0, first_y = 0, first_w = 0, first_h = 0;
    while (fgets(line, sizeof(line), p)) {
        if (want) {
            size_t n = strlen(want);
            if (strncmp(line, want, n) != 0 || (line[n] != ' ' && line[n] != '\0'))
                continue;
        }
        char *prim = want ? NULL : strstr(line, " primary ");
        char *geom = NULL;
        if (prim) {
            geom = prim + 9;
        } else if (strstr(line, " connected ")) {
            geom = strstr(line, " connected ") + 11;
            if (strncmp(geom, "primary ", 8) == 0)
                geom += 8;
        }
        if (!geom)
            continue;
        int x, y, w, h;
        if (sscanf(geom, "%dx%d+%d+%d", &w, &h, &x, &y) != 4)
            continue;
        if (!have_first) {
            first_x = x;
            first_y = y;
            first_w = w;
            first_h = h;
            have_first = 1;
        }
        if (prim || want) {
            fx = x;
            fy = y;
            fw = w;
            fh = h;
            found = 1;
            break;
        }
    }
    pclose(p);
    if (!found && have_first && !want) {
        fx = first_x;
        fy = first_y;
        fw = first_w;
        fh = first_h;
        found = 1;
    }
    if (!found)
        return 0;
    *ox = fx;
    *oy = fy;
    *ow = fw;
    *oh = fh;
    return 1;
}

static void set_atom(Display *dpy, Window win, const char *prop, const char *type, const char *value)
{
    Atom p = XInternAtom(dpy, prop, True);
    Atom t = XInternAtom(dpy, type, True);
    XChangeProperty(dpy, win, p, t, 8, PropModeReplace,
                    (const unsigned char *)value, (int)strlen(value));
}

static void add_wm_state(Display *dpy, Window win, const char *state)
{
    Atom wm_state = XInternAtom(dpy, "_NET_WM_STATE", True);
    Atom s = XInternAtom(dpy, state, True);
    XChangeProperty(dpy, win, wm_state, XA_ATOM, 32, PropModeAppend,
                    (const unsigned char *)&s, 1);
}

static void set_window_type(Display *dpy, Window win, const char *type)
{
    Atom prop = XInternAtom(dpy, "_NET_WM_WINDOW_TYPE", True);
    Atom value = XInternAtom(dpy, type, True);
    XChangeProperty(dpy, win, prop, XA_ATOM, 32, PropModeReplace,
                    (const unsigned char *)&value, 1);
}

static int set_click_through(Display *dpy, Window win)
{
    static void *lib;
    static int tried;
    if (!tried) {
        tried = 1;
        lib = dlopen("libXext.so.6", RTLD_LAZY);
    }
    if (!lib)
        return 0;
    int (*query)(Display *, int *, int *) = NULL;
    void (*combine)(Display *, Window, int, int, int, XRectangle *, int, int, int) = NULL;
    *(void **)(&query) = dlsym(lib, "XShapeQueryExtension");
    *(void **)(&combine) = dlsym(lib, "XShapeCombineRectangles");
    if (!query || !combine)
        return 0;
    int ev = 0, er = 0;
    if (!query(dpy, &ev, &er))
        return 0;
    combine(dpy, win, ShapeInput, 0, 0, NULL, 0, ShapeSet, Unsorted);
    XFlush(dpy);
    return 1;
}

static int x_error(Display *dpy, XErrorEvent *e)
{
    char text[192];
    XGetErrorText(dpy, e->error_code, text, sizeof(text));
    fprintf(stderr, "gpu_overlay: X error: %s (request %d)\n", text, e->request_code);
    return 0;
}

static void x11_open(App *app)
{
    app->dpy = XOpenDisplay(NULL);
    if (!app->dpy)
        die("cannot open X display (is DISPLAY set?)");
    XSetErrorHandler(x_error);

    app->screen = DefaultScreen(app->dpy);
    app->root = RootWindow(app->dpy, app->screen);

    int ox = 0, oy = 0, ow = DisplayWidth(app->dpy, app->screen), oh = DisplayHeight(app->dpy, app->screen);
    if (parse_primary_output(app->opt.output, &ox, &oy, &ow, &oh))
        fprintf(stderr, "gpu_overlay: output %dx%d+%d+%d\n", ow, oh, ox, oy);
    else if (app->opt.output)
        die("output %s is not connected", app->opt.output);

    int x = ox + ow * 3 / 4;
    int y = oy + oh / 2;
    int w = ox + ow - x - 16;
    int h = oy + oh - y - 162;

    if (app->opt.x != INT_MIN)
        x = app->opt.x;
    if (app->opt.y != INT_MIN)
        y = app->opt.y;
    if (app->opt.w != INT_MIN)
        w = app->opt.w;
    if (app->opt.h != INT_MIN)
        h = app->opt.h;
    if (w < 120)
        w = 120;
    if (h < 80)
        h = 80;

    app->x = x;
    app->y = y;
    app->w = w;
    app->h = h;

    XVisualInfo vinfo;
    if (XMatchVisualInfo(app->dpy, app->screen, 32, TrueColor, &vinfo)) {
        app->visual = vinfo.visual;
        app->depth = 32;
    } else {
        app->visual = DefaultVisual(app->dpy, app->screen);
        app->depth = DefaultDepth(app->dpy, app->screen);
        fprintf(stderr, "gpu_overlay: no 32-bit visual, transparency unavailable\n");
    }
    app->cmap = XCreateColormap(app->dpy, app->root, app->visual, AllocNone);

    XSetWindowAttributes attrs;
    memset(&attrs, 0, sizeof(attrs));
    attrs.background_pixel = 0;
    attrs.border_pixel = 0;
    attrs.colormap = app->cmap;
    attrs.override_redirect = app->opt.managed ? False : True;
    attrs.event_mask = ExposureMask | StructureNotifyMask | KeyPressMask;
    if (!app->opt.click_through || app->opt.draggable)
        attrs.event_mask |= ButtonPressMask;
    if (app->opt.draggable)
        attrs.event_mask |= ButtonReleaseMask | PointerMotionMask;

    app->win = XCreateWindow(app->dpy, app->root, x, y, (unsigned)w, (unsigned)h, 0,
                             app->depth, InputOutput, app->visual,
                             CWBackPixel | CWBorderPixel | CWColormap |
                             CWOverrideRedirect | CWEventMask, &attrs);
    if (!app->win)
        die("cannot create X window");

    XStoreName(app->dpy, app->win, "gpu_overlay");
    XClassHint hint = {"gpu_overlay", "GpuOverlay"};
    XSetClassHint(app->dpy, app->win, &hint);
    set_atom(app->dpy, app->win, "_NET_WM_NAME", "UTF8_STRING", "gpu_overlay");
    set_window_type(app->dpy, app->win, "_NET_WM_WINDOW_TYPE_UTILITY");
    add_wm_state(app->dpy, app->win, "_NET_WM_STATE_ABOVE");
    add_wm_state(app->dpy, app->win, "_NET_WM_STATE_SKIP_TASKBAR");
    add_wm_state(app->dpy, app->win, "_NET_WM_STATE_SKIP_PAGER");

    struct {
        unsigned long flags;
        unsigned long functions;
        unsigned long decorations;
        long input_mode;
        unsigned long status;
    } motif = {2, 0, 0, 0, 0};
    Atom motif_atom = XInternAtom(app->dpy, "_MOTIF_WM_HINTS", True);
    XChangeProperty(app->dpy, app->win, motif_atom, motif_atom, 32, PropModeReplace,
                    (unsigned char *)&motif, 5);

    if (app->opt.managed) {
        Atom wm_delete = XInternAtom(app->dpy, "WM_DELETE_WINDOW", True);
        XSetWMProtocols(app->dpy, app->win, &wm_delete, 1);
    }

    XMapRaised(app->dpy, app->win);
    XFlush(app->dpy);

    if (app->opt.click_through && !set_click_through(app->dpy, app->win))
        fprintf(stderr, "gpu_overlay: warning: click-through unavailable (XShape)\n");

    app->hidden = 0;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    die("no suitable Vulkan memory type");
    return 0;
}

static void create_buffer(App *app, VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags props, VkBuffer *buf, VkDeviceMemory *mem)
{
    VkBufferCreateInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKOK(vkCreateBuffer(app->vk.device, &bi, NULL, buf), "vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(app->vk.device, *buf, &req);
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(app->vk.phys, req.memoryTypeBits, props);
    VKOK(vkAllocateMemory(app->vk.device, &ai, NULL, mem), "vkAllocateMemory");
    VKOK(vkBindBufferMemory(app->vk.device, *buf, *mem, 0), "vkBindBufferMemory");
}

static void create_image(App *app, VkFormat format, VkExtent2D extent, VkImageUsageFlags usage,
                         VkImage *image, VkDeviceMemory *mem)
{
    VkImageCreateInfo ii = {0};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent.width = extent.width;
    ii.extent.height = extent.height;
    ii.extent.depth = 1;
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKOK(vkCreateImage(app->vk.device, &ii, NULL, image), "vkCreateImage");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(app->vk.device, *image, &req);
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(app->vk.phys, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKOK(vkAllocateMemory(app->vk.device, &ai, NULL, mem), "vkAllocateMemory");
    VKOK(vkBindImageMemory(app->vk.device, *image, *mem, 0), "vkBindImageMemory");
}

static VkCommandBuffer cmd_begin(App *app)
{
    VkCommandBufferAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = app->vk.cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKOK(vkAllocateCommandBuffers(app->vk.device, &ai, &cb), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKOK(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    return cb;
}

static void cmd_end_submit(App *app, VkCommandBuffer cb)
{
    VKOK(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VKOK(vkQueueSubmit(app->vk.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    VKOK(vkQueueWaitIdle(app->vk.queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(app->vk.device, app->vk.cmd_pool, 1, &cb);
}

static void transition_image(VkCommandBuffer cb, VkImage image,
                             VkImageLayout from, VkImageLayout to,
                             VkAccessFlags src_access, VkAccessFlags dst_access,
                             VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier b = {0};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

static void create_atlas_resources(App *app)
{
    FontAtlas *f = &app->font;
    VkCtx *v = &app->vk;

    VkExtent2D extent = {(uint32_t)f->w, (uint32_t)f->h};
    create_image(app, VK_FORMAT_R8_UNORM, extent,
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 &v->atlas_image, &v->atlas_mem);

    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkDeviceSize size = (VkDeviceSize)f->w * (VkDeviceSize)f->h;
    create_buffer(app, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &staging, &staging_mem);
    void *mapped = NULL;
    VKOK(vkMapMemory(v->device, staging_mem, 0, size, 0, &mapped), "vkMapMemory");
    memcpy(mapped, f->pixels, (size_t)size);
    vkUnmapMemory(v->device, staging_mem);

    VkCommandBuffer cb = cmd_begin(app);
    transition_image(cb, v->atlas_image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy = {0};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent.width = (uint32_t)f->w;
    copy.imageExtent.height = (uint32_t)f->h;
    copy.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(cb, staging, v->atlas_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    transition_image(cb, v->atlas_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    cmd_end_submit(app, cb);

    vkDestroyBuffer(v->device, staging, NULL);
    vkFreeMemory(v->device, staging_mem, NULL);

    VkImageViewCreateInfo vi = {0};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = v->atlas_image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VKOK(vkCreateImageView(v->device, &vi, NULL, &v->atlas_view), "vkCreateImageView");

    VkSamplerCreateInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VKOK(vkCreateSampler(v->device, &si, NULL, &v->sampler), "vkCreateSampler");
}

static void create_descriptor_layout(App *app)
{
    VkCtx *v = &app->vk;
    VkDescriptorSetLayoutBinding b = {0};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings = &b;
    VKOK(vkCreateDescriptorSetLayout(v->device, &ci, NULL, &v->dsl),
         "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VKOK(vkCreateDescriptorPool(v->device, &pci, NULL, &v->dpool),
         "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = v->dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &v->dsl;
    VKOK(vkAllocateDescriptorSets(v->device, &ai, &v->dset),
         "vkAllocateDescriptorSets");
}

static void update_descriptor(App *app)
{
    VkCtx *v = &app->vk;
    VkDescriptorImageInfo ii = {0};
    ii.sampler = v->sampler;
    ii.imageView = v->atlas_view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w = {0};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = v->dset;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(v->device, 1, &w, 0, NULL);
}

static void create_pipeline_layout(App *app)
{
    VkCtx *v = &app->vk;
    VkPushConstantRange range = {0};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = sizeof(float) * 3;

    VkPipelineLayoutCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &v->dsl;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &range;
    VKOK(vkCreatePipelineLayout(v->device, &ci, NULL, &v->pipeline_layout),
         "vkCreatePipelineLayout");
}

static void create_render_pass(App *app)
{
    VkCtx *v = &app->vk;
    VkAttachmentDescription color = {0};
    color.format = v->format;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    VkSubpassDependency dep = {0};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    VKOK(vkCreateRenderPass(v->device, &ci, NULL, &v->render_pass),
         "vkCreateRenderPass");
}

static VkSurfaceFormatKHR pick_surface_format(uint32_t count, const VkSurfaceFormatKHR *formats)
{
    for (uint32_t i = 0; i < count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return formats[i];
    }
    for (uint32_t i = 0; i < count; i++) {
        if (formats[i].format == VK_FORMAT_R8G8B8A8_UNORM &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return formats[i];
    }
    for (uint32_t i = 0; i < count; i++) {
        if (formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return formats[i];
    }
    return formats[0];
}

static void create_swapchain(App *app)
{
    VkCtx *v = &app->vk;
    VkSurfaceCapabilitiesKHR caps;
    VKOK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(v->phys, v->surface, &caps),
         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    uint32_t format_count = 0;
    VKOK(vkGetPhysicalDeviceSurfaceFormatsKHR(v->phys, v->surface, &format_count, NULL),
         "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (format_count == 0)
        die("surface reports no formats");
    VkSurfaceFormatKHR *formats = calloc(format_count, sizeof(*formats));
    if (!formats)
        die("out of memory");
    VKOK(vkGetPhysicalDeviceSurfaceFormatsKHR(v->phys, v->surface, &format_count, formats),
         "vkGetPhysicalDeviceSurfaceFormatsKHR");
    VkSurfaceFormatKHR chosen = pick_surface_format(format_count, formats);
    free(formats);

    v->format = chosen.format;
    v->colorspace = chosen.colorSpace;

    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
        v->composite = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        v->premult = 1;
    } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
        v->composite = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        v->premult = 0;
    } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
        v->composite = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        v->premult = 0;
    } else {
        v->composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        v->premult = 0;
        fprintf(stderr, "gpu_overlay: surface supports opaque alpha only\n");
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = (uint32_t)app->w;
        extent.height = (uint32_t)app->h;
        if (extent.width < caps.minImageExtent.width)
            extent.width = caps.minImageExtent.width;
        if (extent.width > caps.maxImageExtent.width)
            extent.width = caps.maxImageExtent.width;
        if (extent.height < caps.minImageExtent.height)
            extent.height = caps.minImageExtent.height;
        if (extent.height > caps.maxImageExtent.height)
            extent.height = caps.maxImageExtent.height;
    }
    if (extent.width == 0 || extent.height == 0)
        die("zero-sized surface");
    v->extent = extent;

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount && image_count > caps.maxImageCount)
        image_count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = v->surface;
    ci.minImageCount = image_count;
    ci.imageFormat = v->format;
    ci.imageColorSpace = v->colorspace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = v->composite;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VKOK(vkCreateSwapchainKHR(v->device, &ci, NULL, &v->swapchain),
         "vkCreateSwapchainKHR");

    VKOK(vkGetSwapchainImagesKHR(v->device, v->swapchain, &image_count, NULL),
         "vkGetSwapchainImagesKHR");
    if (image_count > MAX_SWAP_IMAGES)
        die("swapchain has %u images (max %d)", image_count, MAX_SWAP_IMAGES);
    v->image_count = image_count;
    VKOK(vkGetSwapchainImagesKHR(v->device, v->swapchain, &image_count, v->images),
         "vkGetSwapchainImagesKHR");

    for (uint32_t i = 0; i < v->image_count; i++) {
        VkImageViewCreateInfo vi = {0};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = v->images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = v->format;
        vi.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VKOK(vkCreateImageView(v->device, &vi, NULL, &v->views[i]), "vkCreateImageView");
    }

    create_render_pass(app);

    v->ss_extent.width = extent.width * 2;
    v->ss_extent.height = extent.height * 2;
    create_image(app, v->format, v->ss_extent,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 &v->ss_image, &v->ss_mem);

    VkImageViewCreateInfo sv = {0};
    sv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    sv.image = v->ss_image;
    sv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    sv.format = v->format;
    sv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sv.subresourceRange.levelCount = 1;
    sv.subresourceRange.layerCount = 1;
    VKOK(vkCreateImageView(v->device, &sv, NULL, &v->ss_view), "vkCreateImageView");

    VkFramebufferCreateInfo fi = {0};
    fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fi.renderPass = v->render_pass;
    fi.attachmentCount = 1;
    fi.pAttachments = &v->ss_view;
    fi.width = v->ss_extent.width;
    fi.height = v->ss_extent.height;
    fi.layers = 1;
    VKOK(vkCreateFramebuffer(v->device, &fi, NULL, &v->ss_framebuffer),
         "vkCreateFramebuffer");
}

static void create_pipeline(App *app)
{
    VkCtx *v = &app->vk;
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = sizeof(overlay_vert_spv);
    si.pCode = overlay_vert_spv;
    VKOK(vkCreateShaderModule(v->device, &si, NULL, &vert), "vkCreateShaderModule");
    si.codeSize = sizeof(overlay_frag_spv);
    si.pCode = overlay_frag_spv;
    VKOK(vkCreateShaderModule(v->device, &si, NULL, &frag), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stages[2] = {{0}};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding = {0};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3] = {{0}};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[0].offset = offsetof(Vertex, x);
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[1].offset = offsetof(Vertex, u);
    attrs[2].location = 2;
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[2].offset = offsetof(Vertex, r);

    VkPipelineVertexInputStateCreateInfo vi = {0};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia = {0};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = {0};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {0};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {0};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState ba = {0};
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = v->premult ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.alphaBlendOp = VK_BLEND_OP_ADD;
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb = {0};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {0};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo gp = {0};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = v->pipeline_layout;
    gp.renderPass = v->render_pass;
    gp.subpass = 0;
    VKOK(vkCreateGraphicsPipelines(v->device, VK_NULL_HANDLE, 1, &gp, NULL, &v->pipeline),
         "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(v->device, vert, NULL);
    vkDestroyShaderModule(v->device, frag, NULL);
}

static void destroy_swapchain_objects(App *app)
{
    VkCtx *v = &app->vk;
    if (!v->device)
        return;
    if (v->pipeline) {
        vkDestroyPipeline(v->device, v->pipeline, NULL);
        v->pipeline = VK_NULL_HANDLE;
    }
    if (v->ss_framebuffer) {
        vkDestroyFramebuffer(v->device, v->ss_framebuffer, NULL);
        v->ss_framebuffer = VK_NULL_HANDLE;
    }
    if (v->ss_view) {
        vkDestroyImageView(v->device, v->ss_view, NULL);
        v->ss_view = VK_NULL_HANDLE;
    }
    if (v->ss_image) {
        vkDestroyImage(v->device, v->ss_image, NULL);
        v->ss_image = VK_NULL_HANDLE;
    }
    if (v->ss_mem) {
        vkFreeMemory(v->device, v->ss_mem, NULL);
        v->ss_mem = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < v->image_count; i++) {
        if (v->framebuffers[i])
            vkDestroyFramebuffer(v->device, v->framebuffers[i], NULL);
        if (v->views[i])
            vkDestroyImageView(v->device, v->views[i], NULL);
        v->framebuffers[i] = VK_NULL_HANDLE;
        v->views[i] = VK_NULL_HANDLE;
    }
    if (v->render_pass) {
        vkDestroyRenderPass(v->device, v->render_pass, NULL);
        v->render_pass = VK_NULL_HANDLE;
    }
    if (v->swapchain) {
        vkDestroySwapchainKHR(v->device, v->swapchain, NULL);
        v->swapchain = VK_NULL_HANDLE;
    }
    v->image_count = 0;
}

static void recreate_swapchain(App *app)
{
    VkCtx *v = &app->vk;
    if (!v->device || !v->swapchain)
        return;
    VKOK(vkDeviceWaitIdle(v->device), "vkDeviceWaitIdle");
    destroy_swapchain_objects(app);
    create_swapchain(app);
    create_pipeline(app);
    v->ready = 1;
}

static int has_device_extension(VkPhysicalDevice phys, const char *name)
{
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &count, NULL) != VK_SUCCESS)
        return 0;
    VkExtensionProperties *props = calloc(count, sizeof(*props));
    if (!props)
        return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, NULL, &count, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(props[i].extensionName, name) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(props);
    return found;
}

static int find_queue_family(VkPhysicalDevice phys, VkSurfaceKHR surface, uint32_t *family)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, NULL);
    if (!count)
        return 0;
    VkQueueFamilyProperties *props = calloc(count, sizeof(*props));
    if (!props)
        return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props);
    int found = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            continue;
        VkBool32 present = VK_FALSE;
        if (vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &present) != VK_SUCCESS)
            continue;
        if (present) {
            *family = i;
            found = 1;
            break;
        }
    }
    free(props);
    return found;
}

static int device_suitable(VkPhysicalDevice phys, VkSurfaceKHR surface, uint32_t *family)
{
    if (!find_queue_family(phys, surface, family))
        return 0;
    if (!has_device_extension(phys, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        return 0;
    uint32_t formats = 0;
    uint32_t modes = 0;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &formats, NULL) != VK_SUCCESS || formats == 0)
        return 0;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &modes, NULL) != VK_SUCCESS || modes == 0)
        return 0;
    return 1;
}

static void pick_physical(App *app)
{
    VkCtx *v = &app->vk;
    uint32_t count = 0;
    VKOK(vkEnumeratePhysicalDevices(v->instance, &count, NULL), "vkEnumeratePhysicalDevices");
    if (!count)
        die("no Vulkan devices");
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices)
        die("out of memory");
    VKOK(vkEnumeratePhysicalDevices(v->instance, &count, devices), "vkEnumeratePhysicalDevices");

    int chosen = -1;
    if (app->opt.device >= 0) {
        uint32_t fam;
        if ((uint32_t)app->opt.device >= count)
            die("--device %d is out of range (%u devices)", app->opt.device, count);
        if (!device_suitable(devices[app->opt.device], v->surface, &fam))
            die("--device %d cannot present to this surface", app->opt.device);
        chosen = app->opt.device;
    } else {
        int best_score = -1;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t fam;
            if (!device_suitable(devices[i], v->surface, &fam))
                continue;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                score += 1000;
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                score += 500;
            else
                score += 100;
            if (props.vendorID == 0x8086)
                score += 100;
            if (score > best_score) {
                best_score = score;
                chosen = (int)i;
            }
        }
        if (chosen < 0)
            die("no Vulkan device can present to this surface");
    }

    v->phys = devices[chosen];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(v->phys, &props);
    fprintf(stderr, "gpu_overlay: Vulkan device %d: %s\n", chosen, props.deviceName);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
        fprintf(stderr, "gpu_overlay: warning: rendering on a CPU device\n");
    free(devices);
}

static void create_instance(App *app)
{
    VkCtx *v = &app->vk;
    VkApplicationInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "gpu_overlay";
    ai.applicationVersion = 1;
    ai.pEngineName = "none";
    ai.apiVersion = VK_API_VERSION_1_1;

    const char *exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME
    };
    VkInstanceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = 2;
    ci.ppEnabledExtensionNames = exts;
    VKOK(vkCreateInstance(&ci, NULL, &v->instance), "vkCreateInstance");
}

static void create_surface(App *app)
{
    VkCtx *v = &app->vk;
    VkXlibSurfaceCreateInfoKHR ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    ci.dpy = app->dpy;
    ci.window = app->win;
    VKOK(vkCreateXlibSurfaceKHR(v->instance, &ci, NULL, &v->surface),
         "vkCreateXlibSurfaceKHR");
}

static void create_device(App *app)
{
    VkCtx *v = &app->vk;
    uint32_t family;
    if (!find_queue_family(v->phys, v->surface, &family))
        die("no graphics+present queue family");
    v->queue_family = family;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi = {0};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;

    const char *exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = exts;
    VKOK(vkCreateDevice(v->phys, &ci, NULL, &v->device), "vkCreateDevice");
    vkGetDeviceQueue(v->device, family, 0, &v->queue);
}

static void create_command_pool(App *app)
{
    VkCtx *v = &app->vk;
    VkCommandPoolCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = v->queue_family;
    VKOK(vkCreateCommandPool(v->device, &ci, NULL, &v->cmd_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = v->cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VKOK(vkAllocateCommandBuffers(v->device, &ai, &v->cmd), "vkAllocateCommandBuffers");
}

static void create_sync(App *app)
{
    VkCtx *v = &app->vk;
    VkSemaphoreCreateInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VKOK(vkCreateSemaphore(v->device, &si, NULL, &v->sem_acquire), "vkCreateSemaphore");
    VKOK(vkCreateSemaphore(v->device, &si, NULL, &v->sem_render), "vkCreateSemaphore");

    VkFenceCreateInfo fi = {0};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VKOK(vkCreateFence(v->device, &fi, NULL, &v->fence), "vkCreateFence");
}

static void vulkan_init(App *app)
{
    VkCtx *v = &app->vk;
    create_instance(app);
    create_surface(app);
    pick_physical(app);
    create_device(app);
    create_command_pool(app);
    create_descriptor_layout(app);
    create_pipeline_layout(app);
    create_swapchain(app);
    create_pipeline(app);
    create_atlas_resources(app);
    update_descriptor(app);

    VkDeviceSize size = (VkDeviceSize)MAX_VERTICES * sizeof(Vertex);
    create_buffer(app, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &v->vbo, &v->vbo_mem);
    void *mapped = NULL;
    VKOK(vkMapMemory(v->device, v->vbo_mem, 0, size, 0, &mapped), "vkMapMemory");
    v->vbo_ptr = mapped;

    create_sync(app);
    v->ready = 1;
}

static int render_frame(App *app)
{
    VkCtx *v = &app->vk;
    if (!v->ready)
        return 0;

    VkResult r = vkWaitForFences(v->device, 1, &v->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS)
        return -1;

    uint32_t index = 0;
    r = vkAcquireNextImageKHR(v->device, v->swapchain, UINT64_MAX,
                              v->sem_acquire, VK_NULL_HANDLE, &index);
    int recreate = 0;
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain(app);
        return 0;
    }
    if (r == VK_SUBOPTIMAL_KHR)
        recreate = 1;
    else if (r != VK_SUCCESS)
        return -1;

    if (app->canvas.count > 0 && v->vbo_ptr)
        memcpy(v->vbo_ptr, app->canvas.verts,
               (size_t)app->canvas.count * sizeof(Vertex));

    VKOK(vkResetFences(v->device, 1, &v->fence), "vkResetFences");
    VKOK(vkResetCommandBuffer(v->cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKOK(vkBeginCommandBuffer(v->cmd, &bi), "vkBeginCommandBuffer");

    VkClearValue clear;
    memset(&clear, 0, sizeof(clear));
    VkRenderPassBeginInfo rp = {0};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = v->render_pass;
    rp.framebuffer = v->ss_framebuffer;
    rp.renderArea.offset.x = 0;
    rp.renderArea.offset.y = 0;
    rp.renderArea.extent = v->ss_extent;
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(v->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    if (app->canvas.count > 0) {
        vkCmdBindPipeline(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipeline);
        vkCmdBindDescriptorSets(v->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                v->pipeline_layout, 0, 1, &v->dset, 0, NULL);

        VkViewport viewport = {0.0f, 0.0f, (float)v->ss_extent.width, (float)v->ss_extent.height, 0.0f, 1.0f};
        VkRect2D scissor = {{0, 0}, v->ss_extent};
        vkCmdSetViewport(v->cmd, 0, 1, &viewport);
        vkCmdSetScissor(v->cmd, 0, 1, &scissor);

        float push[3] = {(float)v->extent.width, (float)v->extent.height, v->premult ? 1.0f : 0.0f};
        vkCmdPushConstants(v->cmd, v->pipeline_layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), push);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(v->cmd, 0, 1, &v->vbo, &offset);
        vkCmdDraw(v->cmd, (uint32_t)app->canvas.count, 1, 0, 0);
    }

    vkCmdEndRenderPass(v->cmd);

    transition_image(v->cmd, v->ss_image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
    transition_image(v->cmd, v->images[index],
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = (int32_t)v->ss_extent.width;
    blit.srcOffsets[1].y = (int32_t)v->ss_extent.height;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1].x = (int32_t)v->extent.width;
    blit.dstOffsets[1].y = (int32_t)v->extent.height;
    blit.dstOffsets[1].z = 1;
    vkCmdBlitImage(v->cmd,
                   v->ss_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   v->images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_LINEAR);

    transition_image(v->cmd, v->images[index],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VKOK(vkEndCommandBuffer(v->cmd), "vkEndCommandBuffer");

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &v->sem_acquire;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &v->cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &v->sem_render;
    VKOK(vkQueueSubmit(v->queue, 1, &submit, v->fence), "vkQueueSubmit");

    VkPresentInfoKHR pi = {0};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &v->sem_render;
    pi.swapchainCount = 1;
    pi.pSwapchains = &v->swapchain;
    pi.pImageIndices = &index;
    r = vkQueuePresentKHR(v->queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        recreate = 1;
    else if (r != VK_SUCCESS)
        return -1;

    if (recreate)
        recreate_swapchain(app);
    return 0;
}

static void vulkan_destroy(App *app)
{
    VkCtx *v = &app->vk;
    if (v->device)
        vkDeviceWaitIdle(v->device);

    if (v->vbo_ptr) {
        vkUnmapMemory(v->device, v->vbo_mem);
        v->vbo_ptr = NULL;
    }
    if (v->vbo)
        vkDestroyBuffer(v->device, v->vbo, NULL);
    if (v->vbo_mem)
        vkFreeMemory(v->device, v->vbo_mem, NULL);
    if (v->sampler)
        vkDestroySampler(v->device, v->sampler, NULL);
    if (v->atlas_view)
        vkDestroyImageView(v->device, v->atlas_view, NULL);
    if (v->atlas_image)
        vkDestroyImage(v->device, v->atlas_image, NULL);
    if (v->atlas_mem)
        vkFreeMemory(v->device, v->atlas_mem, NULL);

    destroy_swapchain_objects(app);

    if (v->dpool)
        vkDestroyDescriptorPool(v->device, v->dpool, NULL);
    if (v->dsl)
        vkDestroyDescriptorSetLayout(v->device, v->dsl, NULL);
    if (v->pipeline_layout)
        vkDestroyPipelineLayout(v->device, v->pipeline_layout, NULL);

    if (v->cmd_pool)
        vkDestroyCommandPool(v->device, v->cmd_pool, NULL);
    if (v->sem_acquire)
        vkDestroySemaphore(v->device, v->sem_acquire, NULL);
    if (v->sem_render)
        vkDestroySemaphore(v->device, v->sem_render, NULL);
    if (v->fence)
        vkDestroyFence(v->device, v->fence, NULL);
    if (v->device)
        vkDestroyDevice(v->device, NULL);
    if (v->surface)
        vkDestroySurfaceKHR(v->instance, v->surface, NULL);
    if (v->instance)
        vkDestroyInstance(v->instance, NULL);
    memset(v, 0, sizeof(*v));
}

static void signal_handler(int sig)
{
    if (sig == SIGUSR1)
        g_toggle = 1;
    else
        g_running = 0;
}

static void app_destroy(App *app)
{
    vulkan_destroy(app);
    if (app->dpy) {
        if (app->win)
            XDestroyWindow(app->dpy, app->win);
        if (app->cmap)
            XFreeColormap(app->dpy, app->cmap);
        XCloseDisplay(app->dpy);
    }
    canvas_destroy(&app->canvas);
    font_atlas_destroy(&app->font);
}

static void usage(const char *argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("  --probe              print one GPU sample and exit\n");
    printf("  --json               JSON output with --probe\n");
    printf("  --selftest           check power/RC6 arithmetic and exit\n");
    printf("  --interval SEC       refresh interval (default 1.0)\n");
    printf("  --device N           Vulkan physical device index\n");
    printf("  --output NAME        XRandR output for default placement\n");
    printf("  --x N --y N          window position\n");
    printf("  --w N --h N          window size\n");
    printf("  --no-panel           text only, no background\n");
    printf("  --no-click-through   make the window accept input\n");
    printf("  --managed            let the window manager manage the window\n");
    printf("  --draggable          drag the window with the left mouse button\n");
    printf("  --help               this text\n");
    printf("\n  SIGUSR1 toggles visibility; click tabs; Tab switches views; Ctrl-C quits.\n");
}

enum {
    OPT_PROBE = 1000,
    OPT_JSON,
    OPT_SELFTEST,
    OPT_INTERVAL,
    OPT_DEVICE,
    OPT_OUTPUT,
    OPT_X,
    OPT_Y,
    OPT_W,
    OPT_H,
    OPT_NO_PANEL,
    OPT_NO_CLICK,
    OPT_MANAGED,
    OPT_DRAGGABLE,
    OPT_HELP
};

static void parse_args(int argc, char **argv, Options *opt)
{
    static const struct option opts[] = {
        {"probe", no_argument, NULL, OPT_PROBE},
        {"json", no_argument, NULL, OPT_JSON},
        {"selftest", no_argument, NULL, OPT_SELFTEST},
        {"interval", required_argument, NULL, OPT_INTERVAL},
        {"device", required_argument, NULL, OPT_DEVICE},
        {"output", required_argument, NULL, OPT_OUTPUT},
        {"x", required_argument, NULL, OPT_X},
        {"y", required_argument, NULL, OPT_Y},
        {"w", required_argument, NULL, OPT_W},
        {"h", required_argument, NULL, OPT_H},
        {"no-panel", no_argument, NULL, OPT_NO_PANEL},
        {"no-click-through", no_argument, NULL, OPT_NO_CLICK},
        {"managed", no_argument, NULL, OPT_MANAGED},
        {"draggable", no_argument, NULL, OPT_DRAGGABLE},
        {"help", no_argument, NULL, OPT_HELP},
        {NULL, 0, NULL, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "h", opts, NULL)) != -1) {
        switch (c) {
        case OPT_PROBE: opt->probe = 1; break;
        case OPT_JSON: opt->json = 1; break;
        case OPT_SELFTEST: opt->selftest = 1; break;
        case OPT_INTERVAL:
            opt->interval = strtod(optarg, NULL);
            if (!(opt->interval >= 0.2) || opt->interval > 10.0)
                die("--interval must be between 0.2 and 10 seconds");
            break;
        case OPT_DEVICE: opt->device = atoi(optarg); break;
        case OPT_OUTPUT: opt->output = optarg; break;
        case OPT_X: opt->x = atoi(optarg); break;
        case OPT_Y: opt->y = atoi(optarg); break;
        case OPT_W: opt->w = atoi(optarg); break;
        case OPT_H: opt->h = atoi(optarg); break;
        case OPT_NO_PANEL: opt->no_panel = 1; break;
        case OPT_NO_CLICK: opt->click_through = 0; break;
        case OPT_MANAGED: opt->managed = 1; break;
        case OPT_DRAGGABLE:
            opt->draggable = 1;
            opt->click_through = 0;
            break;
        case OPT_HELP: usage(argv[0]); exit(0);
        case 'h': usage(argv[0]); exit(0);
        default: usage(argv[0]); exit(2);
        }
    }
}

int main(int argc, char **argv)
{
    Options opt;
    memset(&opt, 0, sizeof(opt));
    opt.interval = 1.0;
    opt.device = -1;
    opt.x = INT_MIN;
    opt.y = INT_MIN;
    opt.w = INT_MIN;
    opt.h = INT_MIN;
    opt.click_through = 1;
    parse_args(argc, argv, &opt);

    if (opt.selftest)
        return selftest();

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGUSR1, &sa, NULL);

    sigset_t unblock;
    sigemptyset(&unblock);
    sigaddset(&unblock, SIGINT);
    sigaddset(&unblock, SIGTERM);
    sigaddset(&unblock, SIGUSR1);
    sigprocmask(SIG_UNBLOCK, &unblock, NULL);

    App app;
    memset(&app, 0, sizeof(app));
    app.opt = opt;
    app.ngpus = gpu_discover(app.gpus, MAX_CARDS);

    if (opt.probe) {
        probe(&app, opt.json);
        return 0;
    }

    if (app.ngpus == 0)
        fprintf(stderr, "gpu_overlay: warning: no GPU telemetry cards found\n");

    x11_open(&app);
    if (!font_atlas_create(&app.font, app.dpy, app.screen))
        die("cannot load a fixed X11 font");
    if (!canvas_init(&app.canvas, &app.font))
        die("out of memory");
    vulkan_init(&app);

    fprintf(stderr, "gpu_overlay: %dx%d at %d,%d\n", app.w, app.h, app.x, app.y);

    gpu_update_all(&app);
    draw_scene(&app);
    if (render_frame(&app) != 0)
        fprintf(stderr, "gpu_overlay: first frame failed\n");

    double next_update = now_sec() + opt.interval;
    double next_raise = now_sec() + 5.0;
    int dirty = 0;

    while (g_running) {
        while (XPending(app.dpy)) {
            XEvent ev;
            XNextEvent(app.dpy, &ev);
            if (ev.type == Expose) {
                dirty = 1;
            } else if (ev.type == ButtonPress) {
                if (ev.xbutton.button == Button1) {
                    int handled = 0;
                    for (int i = 0; i < app.tab_hit_count; i++) {
                        HitRect *r = &app.tab_hits[i];
                        if (ev.xbutton.x >= r->x0 && ev.xbutton.x <= r->x1 &&
                            ev.xbutton.y >= r->y0 && ev.xbutton.y <= r->y1) {
                            int hit = app.tab_hit_tabs[i];
                            if (app.tab != hit) {
                                app.tab = hit;
                                dirty = 1;
                            }
                            app.range_open = 0;
                            handled = 1;
                            break;
                        }
                    }
                    if (!handled) {
                        for (int i = 0; i < app.metric_hit_count; i++) {
                            HitRect *r = &app.metric_hits[i];
                            if (ev.xbutton.x >= r->x0 && ev.xbutton.x <= r->x1 &&
                                ev.xbutton.y >= r->y0 && ev.xbutton.y <= r->y1) {
                                app.graph_metric = app.metric_hit_metrics[i];
                                app.tab = 0;
                                app.range_open = 0;
                                dirty = 1;
                                handled = 1;
                                break;
                            }
                        }
                    }
                    if (!handled && app.range_open) {
                        for (int i = 0; i < app.range_hit_count; i++) {
                            HitRect *r = &app.range_hits[i];
                            if (ev.xbutton.x >= r->x0 && ev.xbutton.x <= r->x1 &&
                                ev.xbutton.y >= r->y0 && ev.xbutton.y <= r->y1) {
                                app.graph_range = app.range_hit_ranges[i];
                                app.range_open = 0;
                                dirty = 1;
                                handled = 1;
                                break;
                            }
                        }
                    }
                    if (!handled) {
                        for (int i = 0; i < app.range_toggle_count; i++) {
                            HitRect *r = &app.range_toggle_hits[i];
                            if (ev.xbutton.x >= r->x0 && ev.xbutton.x <= r->x1 &&
                                ev.xbutton.y >= r->y0 && ev.xbutton.y <= r->y1) {
                                int gpu = app.range_toggle_gpus[i];
                                if (app.range_open && app.range_owner == gpu)
                                    app.range_open = 0;
                                else {
                                    app.range_open = 1;
                                    app.range_owner = gpu;
                                }
                                dirty = 1;
                                handled = 1;
                                break;
                            }
                        }
                    }
                    if (!handled) {
                        if (app.range_open) {
                            app.range_open = 0;
                            dirty = 1;
                        }
                        if (app.opt.draggable) {
                            app.dragging = 1;
                            app.drag_root_x = ev.xbutton.x_root;
                            app.drag_root_y = ev.xbutton.y_root;
                            app.drag_win_x = app.x;
                            app.drag_win_y = app.y;
                        }
                    }
                }
            } else if (ev.type == MotionNotify) {
                if (app.dragging) {
                    int nx = app.drag_win_x + (ev.xmotion.x_root - app.drag_root_x);
                    int ny = app.drag_win_y + (ev.xmotion.y_root - app.drag_root_y);
                    if (nx != app.x || ny != app.y) {
                        XMoveWindow(app.dpy, app.win, nx, ny);
                        app.x = nx;
                        app.y = ny;
                        XFlush(app.dpy);
                    }
                }
            } else if (ev.type == ButtonRelease) {
                if (ev.xbutton.button == Button1)
                    app.dragging = 0;
            } else if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.width != app.w || ev.xconfigure.height != app.h) {
                    app.w = ev.xconfigure.width;
                    app.h = ev.xconfigure.height;
                    dirty = 1;
                }
            } else if (ev.type == ClientMessage) {
                g_running = 0;
            } else if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_q || ks == XK_Escape) {
                    g_running = 0;
                } else if (ks == XK_Tab) {
                    app.tab = !app.tab;
                    app.range_open = 0;
                    dirty = 1;
                }
            }
        }

        double t = now_sec();
        if (g_toggle) {
            g_toggle = 0;
            app.hidden = !app.hidden;
            app.dragging = 0;
            if (app.hidden) {
                XUnmapWindow(app.dpy, app.win);
            } else {
                XMapRaised(app.dpy, app.win);
                if (app.opt.click_through)
                    set_click_through(app.dpy, app.win);
            }
            XFlush(app.dpy);
            fprintf(stderr, "gpu_overlay: %s\n", app.hidden ? "hidden" : "shown");
            dirty = 1;
        }

        if (t >= next_update) {
            gpu_update_all(&app);
            next_update = t + opt.interval;
            dirty = 1;
        }

        if (dirty && !app.hidden) {
            draw_scene(&app);
            if (render_frame(&app) != 0)
                g_running = 0;
            dirty = 0;
        }

        if (t >= next_raise) {
            if (!app.hidden)
                XRaiseWindow(app.dpy, app.win);
            next_raise = t + 5.0;
        }

        struct pollfd pfd;
        pfd.fd = ConnectionNumber(app.dpy);
        pfd.events = POLLIN;
        pfd.revents = 0;
        poll(&pfd, 1, 100);
    }

    app_destroy(&app);
    return 0;
}
