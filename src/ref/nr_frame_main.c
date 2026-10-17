/*
 * nr_frame - `src/ref/nr_frame.py`, the command, in C.
 *
 *     nr_frame IN.png OUT.png [--weights W] [--size HxW] [--profile P] [--style-index N]
 *              [--local-tone F] [--local-structure F] [--skin-structure F] [--auto-mask F]
 *              [--control-mask MASK.png] [--intensity F] [--intensity-ladder A,B,C]
 *              [--detail-strength F] [--colour-strength F] [--detail-radius F]
 *              [--frame-index N] [-v]
 *
 * The same flags, the same printed lines. Pictures are PNG, read and written with libpng:
 * any PNG in (8- or 16-bit, grey, palette, alpha - all reduced to 8-bit RGB), 8-bit RGB
 * out, the same `byte / 255` and `value * 255 + 0.5` as `image_io.py`. Where the Python
 * shells out to ImageMagick this does not, so `--size` is this project's own bilinear
 * resample (`nr_resize_axis`, the one the daemon uses) rather than ImageMagick's filter.
 * `--gpu`, `--resident` and `--accel` are accepted and mean nothing here: this program has
 * one backend, the resident graph in `libnr_frame`. The network runs once; the ladder
 * composes the head at each intensity as the Python does.
 *
 *     nr_frame --replay DUMP OUT [--render-scale F] [--temporal F] [--hold F] [--release F]
 *              [--cut-limit F] [--min-extent N] [--no-letterbox] [--profile P] [--intensity F] ...
 *
 * `src/bench/restill.py` without the daemon: a `nr_daemon.py --dump` capture's NNN_in.png
 * frames, in their order, through `nr_frame_live` - the daemon's frame, its letterbox, its
 * render scale and its history - into OUT as NNN_in.png and NNN_out.png, numbered from 001
 * as the daemon's own dump numbers them. The render scale is 1 unless set, as restill's is;
 * the other knobs are the daemon's defaults. It runs on whichever runtime is behind
 * libnr_frame (NR_GPU_BACKEND), so a capture can be re-rendered on Metal or Direct3D 12.
 */
#include "nr_frame.h"
#include "nr_image.h"
#include "nr_portable.h"

#include <png.h>
#include <setjmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <pthread.h>
#endif

static double now(void) { return nr_now(); }

static void die(const char *what)
{
    fprintf(stderr, "nr_frame: %s\n", what);
    exit(1);
}

/* `image_io.load`, through libpng: (H, W, 3) float32 in [0, 1]. Sixteen-bit files are
 * reduced to eight as ImageMagick's `-depth 8` reduces them; grey, palette and alpha are
 * expanded or dropped. */

/* png_create_*_struct returns NULL when the libpng the binary was compiled against is not
 * the one it is running with (libpng refuses across major.minor); the file is not the
 * problem then, the build is, and the message should say so. */
static void png_version_mismatch(void)
{
    unsigned lib = png_access_version_number();
    if (lib / 100 != PNG_LIBPNG_VER / 100) {
        fprintf(stderr, "libpng mismatch: built against %s, running with %u.%u.%u; rebuild against the "
                        "headers of the libpng that is linked\n", PNG_LIBPNG_VER_STRING, lib / 10000, (lib / 100) % 100, lib % 100);
        exit(1);
    }
}
static unsigned char *read_png_bytes(const char *path, int *height, int *width)
{
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    FILE *f = NULL;
    fopen_s(&f, path, "rb");
#else
    FILE *f = fopen(path, "rb");
#endif

    if (!f) { perror(path); exit(1); }
    unsigned char sig[8];
    if (fread(sig, 1, 8, f) != 8 || png_sig_cmp(sig, 0, 8)) { fprintf(stderr, "%s: not a PNG\n", path); exit(1); }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) png_version_mismatch();
    png_infop info = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) { fprintf(stderr, "%s: libpng could not read it\n", path); exit(1); }
    png_init_io(png, f);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);
    png_uint_32 w = png_get_image_width(png, info), h = png_get_image_height(png, info);
    int depth = png_get_bit_depth(png, info), type = png_get_color_type(png, info);
    if (type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (type == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (depth == 16) png_set_strip_16(png);
    if (type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_set_strip_alpha(png);
    png_set_packing(png);
    png_read_update_info(png, info);
    if (png_get_rowbytes(png, info) != (size_t)w * 3) { fprintf(stderr, "%s: unexpected row layout\n", path); exit(1); }
    unsigned char *raw = malloc((size_t)w * h * 3);
    png_bytep *rows = malloc(h * sizeof *rows);
    if (!raw || !rows) die("out of memory");
    for (png_uint_32 y = 0; y < h; y++) rows[y] = raw + (size_t)y * w * 3;
    png_read_image(png, rows);
    png_read_end(png, NULL);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(f);
    free(rows);
    *height = (int)h; *width = (int)w;
    return raw;
}

static float *read_png(const char *path, int *height, int *width)
{
    unsigned char *raw = read_png_bytes(path, height, width);
    size_t n = (size_t)*width * *height * 3;
    float *out = malloc(n * sizeof(float));
    if (!out) die("out of memory");
    for (size_t i = 0; i < n; i++) out[i] = (float)raw[i] / 255.0f;
    free(raw);
    return out;
}

/* `nr_image.bilinear`: the axis plan of `_axis_plan` - pixel centres mapped onto the
 * source, floor and its neighbour clamped, the fraction as the weight - then one axis at
 * a time through `nr_resize_axis`, the intermediate rounded to float32 between them. */
static float *resize(const float *source, int height, int width, int channels, int target_h, int target_w)
{
    const float *current = source;
    float *owned = NULL;
    int h = height, w = width;
    for (int axis = 0; axis < 2; axis++) {
        int extent = axis == 0 ? h : w, count = axis == 0 ? target_h : target_w;
        if (extent == count) continue;
        int32_t *low = malloc((size_t)count * sizeof *low), *high = malloc((size_t)count * sizeof *high);
        float *weight = malloc((size_t)count * sizeof *weight);
        if (!low || !high || !weight) die("out of memory");
        for (int i = 0; i < count; i++) {
            float centre = ((float)i + 0.5f) * ((float)extent / (float)count) - 0.5f;
            float floored = floorf(centre);
            int lo = (int)(floored < 0.0f ? 0.0f : floored > (float)(extent - 1) ? (float)(extent - 1) : floored);
            low[i] = lo;
            high[i] = lo + 1 > extent - 1 ? extent - 1 : lo + 1;
            float frac = centre - (float)lo;
            weight[i] = frac < 0.0f ? 0.0f : frac > 1.0f ? 1.0f : frac;
        }
        int nh = axis == 0 ? count : h, nw = axis == 0 ? w : count;
        float *out = malloc((size_t)nh * nw * channels * sizeof(float));
        if (!out) die("out of memory");
        nr_resize_axis(current, (ptrdiff_t)w * channels, channels, 1, (size_t)nh, (size_t)nw, (size_t)channels,
                       axis, low, high, weight, out);
        free(low); free(high); free(weight); free(owned);
        owned = out; current = out; h = nh; w = nw;
    }
    if (!owned) {
        owned = malloc((size_t)h * w * channels * sizeof(float));
        if (!owned) die("out of memory");
        memcpy(owned, source, (size_t)h * w * channels * sizeof(float));
    }
    return owned;
}

static float *load_image(const char *path, int *height, int *width, int resize_h, int resize_w)
{
    float *image = read_png(path, height, width);
    if (resize_h && (resize_h != *height || resize_w != *width)) {
        float *resized = resize(image, *height, *width, 3, resize_h, resize_w);
        free(image);
        image = resized;
        *height = resize_h; *width = resize_w;
    }
    return image;
}

/* An 8-bit RGB PNG of (height, width, 3) bytes. */
static void save_png_bytes(const unsigned char *raw, int height, int width, const char *path)
{
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    FILE *f = NULL;
    fopen_s(&f, path, "wb");
#else
    FILE *f = fopen(path, "wb");
#endif
    if (!f) { perror(path); exit(1); }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) png_version_mismatch();
    png_infop info = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) { fprintf(stderr, "%s: libpng could not write it\n", path); exit(1); }
    png_init_io(png, f);
    png_set_compression_level(png, 1);      /* the frames feed a video: speed over bytes */
    png_set_IHDR(png, info, (png_uint_32)width, (png_uint_32)height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    png_bytep *rows = malloc((size_t)height * sizeof *rows);
    if (!rows) die("out of memory");
    for (int y = 0; y < height; y++) rows[y] = (png_bytep)raw + (size_t)y * width * 3;
    png_write_image(png, rows);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(f);
    free(rows);
}

/* `image_io.save`: clip, `(a * 255 + 0.5)` to bytes, an 8-bit RGB PNG. */
static void save_image(const float *image, int height, int width, const char *path)
{
    size_t n = (size_t)height * width * 3;
    unsigned char *raw = malloc(n);
    if (!raw) die("out of memory");
    for (size_t i = 0; i < n; i++) {
        float v = image[i];
        v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
        raw[i] = (unsigned char)(v * 255.0f + 0.5f);
    }
    save_png_bytes(raw, height, width, path);
    free(raw);
}

static int readable(const char *path)
{
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    FILE *f = NULL;
    fopen_s(&f, path, "rb");
#else
    FILE *f = fopen(path, "rb");
#endif
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* The weights compiled into libnr_frame when the build has them (NULL asks for those), else
 * `ROOT / work / mlxw / dlssnr-logical.safetensors`: the weights are the owner's and live in
 * work/ whichever build made this executable: beside it when the Makefile put it in work/,
 * one directory up and over when CMake put it in `build/`, else relative to the caller. */
static const char *default_weights(char *buf, size_t cap)
{
    char dir[1024];
    if (nr_frame_embedded_weights_size()) return NULL;
    if (nr_dl_self_dir((const void *)&default_weights, dir, sizeof dir) == 0) {
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
        sprintf_s(buf, cap, "%s/mlxw/dlssnr-logical.safetensors", dir);
#else
        snprintf(buf, cap, "%s/mlxw/dlssnr-logical.safetensors", dir);
#endif

        if (readable(buf)) return buf;
        
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
        sprintf_s(buf, cap, "%s/../work/mlxw/dlssnr-logical.safetensors", dir);
#else
        snprintf(buf, cap, "%s/../work/mlxw/dlssnr-logical.safetensors", dir);
#endif

        if (readable(buf)) return buf;
    }
    
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    sprintf_s(buf, cap, "work/mlxw/dlssnr-logical.safetensors");
#else
    snprintf(buf, cap, "work/mlxw/dlssnr-logical.safetensors");
#endif

    return buf;
}

static const char *PROFILES[] = { "standard", "natural", "cinematic", "neutral", "vendor" };

static int profile(const char *name, nr_frame_params *p)
{
    /* `features.PROFILES` and `nr_frame.VENDOR_DEFAULTS` */
    if (!strcmp(name, "standard"))  { p->normalized_style = 0.0f;        p->local_tone = 1.0f; p->local_structure = 1.0f; return 0; }
    if (!strcmp(name, "natural"))   { p->normalized_style = 1.0f / 128;  p->local_tone = 1.0f; p->local_structure = 1.0f; return 0; }
    if (!strcmp(name, "cinematic")) { p->normalized_style = 2.0f / 128;  p->local_tone = 1.0f; p->local_structure = 1.0f; return 0; }
    if (!strcmp(name, "neutral"))   { p->normalized_style = 0.0f;        p->local_tone = 0.0f; p->local_structure = 0.0f; return 0; }
    if (!strcmp(name, "vendor"))    { p->normalized_style = 0.0f;        p->local_tone = 1.0f; p->local_structure = 1.5f; return 0; }
    return -1;
}

/* ------------------------------------------------------------------------- */
/* --replay: a --dump capture through nr_frame_live, in its order             */
/* ------------------------------------------------------------------------- */

static void join(char *out, size_t cap, const char *dir, const char *name)
{
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    sprintf_s(out, cap, "%s/%s", dir, name);
#else
    snprintf(out, cap, "%s/%s", dir, name);
#endif
}

/* The number of a daemon's NNN_in.png, or -1 for anything else in the folder. */
static long dump_number(const char *name)
{
    const char *p = name;
    if (*p < '0' || *p > '9') return -1;
    long n = 0;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; if (n > 100000000L) return -1; }
    return strcmp(p, "_in.png") ? -1 : n;
}

static int compare_long(const void *a, const void *b)
{
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The frame's own cost over the run: mean, median and extremes, and the fps they imply.
 * `elapsed_ms` is the graph time only - PNG read and write sit outside `run_ms` and are
 * not what a game pays - so this is the frame rate the model can sustain, not the replay's. */
static void report_frame_times(const double *ms, size_t count)
{
    if (!count) return;
    double *sorted = malloc(count * sizeof *sorted), total = 0;
    if (!sorted) return;
    for (size_t i = 0; i < count; i++) { sorted[i] = ms[i]; total += ms[i]; }
    qsort(sorted, count, sizeof *sorted, compare_double);
    double mean = total / (double)count;
    double median = (count & 1) ? sorted[count / 2] : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
    double p95 = sorted[(size_t)(0.95 * (double)(count - 1))];
    printf("  summary %zu frames  mean %.2f ms (%.2f fps)  median %.2f ms (%.2f fps)"
           "  min %.2f  p95 %.2f  max %.2f  graph time, excludes PNG I/O\n",
           count, mean, 1e3 / mean, median, 1e3 / median, sorted[0], p95, sorted[count - 1]);
    free(sorted);
}

/* Each entry of `dir` through `visit`; 0 when the folder could be read. */
static int list_dir(const char *dir, void (*visit)(const char *name, void *context), void *context)
{
#ifdef _WIN32
    char pattern[1200];
    join(pattern, sizeof pattern, dir, "*");
    struct _finddata_t found;
    intptr_t handle = _findfirst(pattern, &found);
    if (handle == -1) return errno == ENOENT ? 0 : -1;
    do visit(found.name, context); while (_findnext(handle, &found) == 0);
    _findclose(handle);
#else
    DIR *d = opendir(dir);
    if (!d) return -1;
    for (struct dirent *e; (e = readdir(d));) visit(e->d_name, context);
    closedir(d);
#endif
    return 0;
}

struct numbers { long *values; size_t count, cap; int others; };

static void collect(const char *name, void *context)
{
    struct numbers *n = context;
    if (!strcmp(name, ".") || !strcmp(name, "..")) return;
    long number = dump_number(name);
    if (number < 0) { n->others++; return; }
    if (n->count == n->cap) {
        n->cap = n->cap ? n->cap * 2 : 64;
        n->values = realloc(n->values, n->cap * sizeof *n->values);
        if (!n->values) die("out of memory");
    }
    n->values[n->count++] = number;
}

static void make_dir(const char *dir)
{
#ifdef _WIN32
    int rc = _mkdir(dir);
#else
    int rc = mkdir(dir, 0777);
#endif
    if (rc && errno != EEXIST) { perror(dir); exit(1); }
}

/* --realtime: keep the output on the input's clock by dropping the frames the model cannot
 * keep up with. The input plays at `input_fps`; a processed frame is due when the input's
 * own time reaches the next slot, so the number processed per second of input is the rate
 * the model sustains - `16/60` of the frames for a 60 fps source at 16 fps - and what is
 * left out is repeated (the stream) or never written (the replay). `rate` is that sustained
 * rate: 0 measures it from the running mean of each frame's wall time, a positive value
 * fixes it. The mean is of the model's wall time, not the output's, and it does not change
 * with the stride, so measuring only the frames that are actually run is enough. */
typedef struct {
    int on;
    double rate;        /* 0: measure; > 0: fixed, frames a second */
    double input_fps;
    double next_t;      /* the input time the next processed frame is due */
    double ema_ms;      /* running mean of a processed frame's wall time */
    long processed;
    long held;
} realtime_pace;

/* 1 to run the model on input frame `index` (0-based), 0 to drop it. Every frame is run
 * until a rate is known, so the first frames measure the model at stride 1. */
static int pace_process(realtime_pace *q, long index)
{
    if (!q || !q->on) return 1;
    double t = (double)index / q->input_fps;
    if (q->rate <= 0) { q->next_t = t; return 1; }      /* measuring */
    if (t + 1e-9 < q->next_t) return 0;                 /* not due yet: drop */
    q->next_t = t + 1.0 / q->rate;                      /* due: process and slot the next */
    return 1;
}

/* The wall time of a frame that was run, in ms. Folds it into the mean and, until a rate is
 * fixed, back into the rate - so an auto run follows the machine as it settles. */
static void pace_record(realtime_pace *q, double ms)
{
    if (!q || !q->on) return;
    q->processed++;
    q->ema_ms = q->ema_ms > 0 ? 0.9 * q->ema_ms + 0.1 * ms : ms;
    if (q->rate > 0) return;
    if (q->processed < 8) return;                       /* a few frames before trusting it */
    double r = 1000.0 / q->ema_ms;
    if (r > q->input_fps) r = q->input_fps;
    if (r < 0.5) r = 0.5;
    q->rate = r;
}

/* The rate the replay actually ran at, and how much it dropped. Written to stdout for the
 * tools (restill.py reads the update rate back to stitch the video on the input's clock). */
static void pace_report(const realtime_pace *q, size_t total)
{
    if (!q || !q->on || !total) return;
    double update = q->rate > 0 ? q->input_fps * (double)q->processed / (double)total : q->input_fps;
    double stride = q->processed ? (double)total / (double)q->processed : 0;
    printf("  realtime: %ld of %zu frames processed, input %.4g fps, update %.3f fps, "
           "stride %.2f, dropped %ld\n",
           q->processed, total, q->input_fps, update, stride, q->held);
    fflush(stdout);
}

/* The dump's own bytes, not a re-encode: the _in.png of the output is the input the daemon
 * saw, byte for byte. */
static int copy_file(const char *from, const char *to)
{
    FILE *in = fopen(from, "rb");
    if (!in) return 1;
    FILE *out = fopen(to, "wb");
    if (!out) { fclose(in); return 1; }
    char buf[1 << 16];
    int failed = 0;
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { failed = 1; break; }
    if (ferror(in)) failed = 1;
    fclose(in);
    if (fclose(out)) failed = 1;
    return failed;
}

/* One frame of the replay, start to finish, with the model run in the calling thread: read
 * NNN_in.png `number`, run `live`, write NNN_in.png and NNN_out.png `shown` in `out`. The
 * model's wall time in ms goes to *ms. The prompt line and the same report the sequential
 * loop prints. Used for the sequential loop and, under --pool, for the few calibration
 * frames that set an auto rate before the pipeline takes over. */
static int replay_frame(nr_frame_live *live, const nr_frame_params *p, const nr_frame_live_settings *s,
                        const char *source, const char *out, long number, long shown, double *ms)
{
    char name[64], path[1200];
#ifdef _WIN32
    sprintf_s(name, sizeof name, "%03ld_in.png", number);
#else
    snprintf(name, sizeof name, "%03ld_in.png", number);
#endif
    join(path, sizeof path, source, name);
    int height, width;
    unsigned char *rgb = read_png_bytes(path, &height, &width);
    if (!rgb) { fprintf(stderr, "%s: unreadable\n", path); return 1; }
    size_t pixels = (size_t)height * width;
    /* every swapchain behind the captures so far was B8G8R8A8, and the dumps are 8-bit,
     * so the bytes sent are the bytes the game sent */
    unsigned char *bgra = malloc(pixels * 4), *answer = malloc(pixels * 4), *shown_px = malloc(pixels * 3);
    if (!bgra || !answer || !shown_px) die("out of memory");
    for (size_t i = 0; i < pixels; i++) {
        bgra[i * 4 + 0] = rgb[i * 3 + 2]; bgra[i * 4 + 1] = rgb[i * 3 + 1];
        bgra[i * 4 + 2] = rgb[i * 3 + 0]; bgra[i * 4 + 3] = 255;
    }
    nr_frame_live_report r;
    double started = now();
    if (nr_frame_live_run(live, p, s, bgra, width, height, 1, answer, &r)) {
        fprintf(stderr, "%s: nr_frame_live_run: %s\n", path, nr_frame_error());
        free(rgb); free(bgra); free(answer); free(shown_px);
        return 1;
    }
    double elapsed = now() - started;
    if (ms) *ms = elapsed * 1e3;
    for (size_t i = 0; i < pixels; i++) {
        shown_px[i * 3 + 0] = answer[i * 4 + 2]; shown_px[i * 3 + 1] = answer[i * 4 + 1];
        shown_px[i * 3 + 2] = answer[i * 4 + 0];
    }
    char in_path[1200], out_path[1200], in_name[64], out_name[64];
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
    sprintf_s(in_name, sizeof in_name, "%03ld_in.png", shown);
    sprintf_s(out_name, sizeof out_name, "%03ld_out.png", shown);
#else
    snprintf(in_name, sizeof in_name, "%03ld_in.png", shown);
    snprintf(out_name, sizeof out_name, "%03ld_out.png", shown);
#endif
    join(in_path, sizeof in_path, out, in_name);
    join(out_path, sizeof out_path, out, out_name);
    copy_file(path, in_path);
    save_png_bytes(shown_px, height, width, out_path);
    int active_width = r.right - r.left, active_height = r.bottom - r.top;
    printf("  %s -> %s  %dx%d in %.2fs  change %.5f  %s %.4f  network %dx%d scale %g",
           name, out_name, width, height, elapsed, r.change,
           r.with_history ? "history, cut" : "no history, cut", r.cut, r.network_width,
           r.network_height, (double)s->render_scale);
    if (s->render_scale < 1.0f && r.render_width != (int)lround((double)active_width * s->render_scale))
        printf(" (runs as %.3g)", (double)r.render_width / (double)active_width);
    if (active_width != width || active_height != height)
        printf("  letterbox %dpx of rows and %dpx of columns skipped", height - active_height,
               width - active_width);
    printf("\n");
    fflush(stdout);
    free(rgb); free(bgra); free(answer); free(shown_px);
    return 0;
}

#ifndef _WIN32
/* --pool N: the offline replay's prefetch, for a conversion rather than a test. A frame is
 * one model run with a PNG read in front and a PNG write behind it, and the card idles
 * through both. Here N decoded frames wait in memory: a few threads read the selected
 * NNN_in.png files ahead of the model, the replay thread runs the model back to back with
 * the next frame already decoded, and a few threads write NNN_out.png (and copy NNN_in.png)
 * behind it. The model then waits on nothing but the card. */
enum { SLOT_FREE = 0, SLOT_DECODING, SLOT_DECODED, SLOT_MODELED, SLOT_WRITING };
#define POOL_CALIB 16       /* frames run before an auto rate is trusted */
#define POOL_DECODERS 6     /* NR_POOL_DECODERS overrides: heavier PNGs want more */
#define POOL_WRITERS 4      /* NR_POOL_WRITERS overrides */

typedef struct {
    long pos;                   /* index in the run; -1 when free */
    long number;                /* the source frame's NNN */
    int state, h, w;
    unsigned char *rgb;         /* decoded input, h*w*3, the PNG's byte order */
    unsigned char *out;         /* the answer, h*w*3 */
} pool_slot;

typedef struct {
    nr_frame_live *live;
    const nr_frame_params *p;
    const nr_frame_live_settings *s;
    const char *source, *out;
    const long *numbers;
    long count, base;           /* base: the 1-based output number of numbers[0] */
    int pool, decode_threads, write_threads;
    pool_slot *slots;
    pthread_mutex_t m;
    pthread_cond_t cv;
    long decode_pos, written;
    int failed;
    double *samples;            /* count entries */
} pool_ctx;

static int pool_slot_of(pool_ctx *c, long pos, int at_least)
{
    for (int i = 0; i < c->pool; i++)
        if (c->slots[i].pos == pos && c->slots[i].state >= at_least) return i;
    return -1;
}

/* Reads the selected in.png files, in order, into free slots as they come. */
static void *pool_decode_thread(void *arg)
{
    pool_ctx *c = arg;
    for (;;) {
        pthread_mutex_lock(&c->m);
        int idx = -1;
        long pos = -1;
        for (;;) {
            if (c->failed) { pthread_mutex_unlock(&c->m); return NULL; }
            if (c->decode_pos >= c->count) { pthread_mutex_unlock(&c->m); return NULL; }
            for (int i = 0; i < c->pool; i++)
                if (c->slots[i].state == SLOT_FREE) { idx = i; break; }
            if (idx >= 0) break;
            pthread_cond_wait(&c->cv, &c->m);
        }
        pos = c->decode_pos++;
        c->slots[idx].pos = pos;
        c->slots[idx].number = c->numbers[pos];
        c->slots[idx].state = SLOT_DECODING;
        pthread_mutex_unlock(&c->m);

        char name[64], path[1200];
#ifdef _WIN32
        sprintf_s(name, sizeof name, "%03ld_in.png", c->slots[idx].number);
#else
        snprintf(name, sizeof name, "%03ld_in.png", c->slots[idx].number);
#endif
        join(path, sizeof path, c->source, name);
        int h = 0, w = 0;
        unsigned char *rgb = read_png_bytes(path, &h, &w);
        unsigned char *out = rgb ? malloc((size_t)h * w * 3) : NULL;

        pthread_mutex_lock(&c->m);
        if (!rgb || !out) {
            c->failed = 1;
            free(rgb); free(out);
        } else {
            c->slots[idx].rgb = rgb; c->slots[idx].out = out;
            c->slots[idx].h = h; c->slots[idx].w = w;
            c->slots[idx].state = SLOT_DECODED;
        }
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->m);
    }
}

/* Writes the modeled frames behind the model; the input is copied from the dump, not
 * re-encoded. */
static void *pool_write_thread(void *arg)
{
    pool_ctx *c = arg;
    for (;;) {
        pthread_mutex_lock(&c->m);
        int idx = -1;
        for (;;) {
            if (c->failed) { pthread_mutex_unlock(&c->m); return NULL; }
            for (int i = 0; i < c->pool; i++)
                if (c->slots[i].state == SLOT_MODELED) { idx = i; break; }
            if (idx >= 0) break;
            if (c->written >= c->count) { pthread_mutex_unlock(&c->m); return NULL; }
            pthread_cond_wait(&c->cv, &c->m);
        }
        pool_slot *slot = &c->slots[idx];
        long pos = slot->pos, number = slot->number;
        int h = slot->h, w = slot->w;
        unsigned char *out = slot->out;
        slot->state = SLOT_WRITING;
        pthread_mutex_unlock(&c->m);

        long shown = c->base + pos;             /* 1-based, off the processed frames */
        char name[64], path[1200], src_name[64], src_path[1200];
#ifdef _WIN32
        sprintf_s(name, sizeof name, "%03ld_out.png", shown);
#else
        snprintf(name, sizeof name, "%03ld_out.png", shown);
#endif
        join(path, sizeof path, c->out, name);
        save_png_bytes(out, h, w, path);
#ifdef _WIN32
        sprintf_s(name, sizeof name, "%03ld_in.png", shown);
        sprintf_s(src_name, sizeof src_name, "%03ld_in.png", number);
#else
        snprintf(name, sizeof name, "%03ld_in.png", shown);
        snprintf(src_name, sizeof src_name, "%03ld_in.png", number);
#endif
        join(path, sizeof path, c->out, name);
        join(src_path, sizeof src_path, c->source, src_name);
        copy_file(src_path, path);

        pthread_mutex_lock(&c->m);
        free(slot->rgb); free(slot->out);
        slot->rgb = slot->out = NULL; slot->pos = -1; slot->state = SLOT_FREE;
        c->written++;
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->m);
    }
}

/* The replay thread: run the model on the selected frames in order, back to back. */
static int pool_run(pool_ctx *c)
{
    pthread_t *decoders = calloc(c->decode_threads, sizeof *decoders);
    pthread_t *writers = calloc(c->write_threads, sizeof *writers);
    if (!decoders || !writers) die("out of memory");
    for (int i = 0; i < c->decode_threads; i++) pthread_create(&decoders[i], NULL, pool_decode_thread, c);
    for (int i = 0; i < c->write_threads; i++) pthread_create(&writers[i], NULL, pool_write_thread, c);

    unsigned char *bgra = NULL, *answer = NULL;
    size_t cap = 0;
    for (long k = 0; !c->failed && k < c->count; k++) {
        pthread_mutex_lock(&c->m);
        int idx;
        while ((idx = pool_slot_of(c, k, SLOT_DECODED)) < 0) {
            if (c->failed) break;
            pthread_cond_wait(&c->cv, &c->m);
        }
        if (idx < 0) { pthread_mutex_unlock(&c->m); break; }
        pool_slot *slot = &c->slots[idx];
        long pos = slot->pos, number = slot->number;
        int h = slot->h, w = slot->w;
        unsigned char *rgb = slot->rgb, *out = slot->out;
        pthread_mutex_unlock(&c->m);

        size_t pixels = (size_t)h * w;
        if (pixels * 4 > cap) { cap = pixels * 4; bgra = realloc(bgra, cap); answer = realloc(answer, cap); }
        if (!bgra || !answer) die("out of memory");
        for (size_t i = 0; i < pixels; i++) {
            bgra[i * 4 + 0] = rgb[i * 3 + 2]; bgra[i * 4 + 1] = rgb[i * 3 + 1];
            bgra[i * 4 + 2] = rgb[i * 3 + 0]; bgra[i * 4 + 3] = 255;
        }
        nr_frame_live_report r;
        double started = now();
        if (nr_frame_live_run(c->live, c->p, c->s, bgra, w, h, 1, answer, &r)) {
            fprintf(stderr, "nr_frame_live_run: %s\n", nr_frame_error());
            c->failed = 1;
            break;
        }
        double elapsed = now() - started;
        if (c->samples) c->samples[k] = elapsed * 1e3;
        for (size_t i = 0; i < pixels; i++) {
            out[i * 3 + 0] = answer[i * 4 + 2]; out[i * 3 + 1] = answer[i * 4 + 1];
            out[i * 3 + 2] = answer[i * 4 + 0];
        }
        pthread_mutex_lock(&c->m);
        slot->state = SLOT_MODELED;
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->m);

        long shown = c->base + pos;
        int active_width = r.right - r.left, active_height = r.bottom - r.top;
        printf("  %03ld_in.png -> %03ld_out.png  %dx%d in %.2fs  change %.5f  %s %.4f  network %dx%d scale %g",
               number, shown, w, h, elapsed, r.change,
               r.with_history ? "history, cut" : "no history, cut", r.cut, r.network_width,
               r.network_height, (double)c->s->render_scale);
        if (c->s->render_scale < 1.0f && r.render_width != (int)lround((double)active_width * c->s->render_scale))
            printf(" (runs as %.3g)", (double)r.render_width / (double)active_width);
        if (active_width != w || active_height != h)
            printf("  letterbox %dpx of rows and %dpx of columns skipped", h - active_height, w - active_width);
        printf("\n");
        fflush(stdout);
    }

    /* let the writers finish, then wake any thread still parked on a failure */
    pthread_mutex_lock(&c->m);
    while (!c->failed && c->written < c->count) pthread_cond_wait(&c->cv, &c->m);
    if (c->failed) c->failed = 1;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->m);
    for (int i = 0; i < c->decode_threads; i++) pthread_join(decoders[i], NULL);
    for (int i = 0; i < c->write_threads; i++) pthread_join(writers[i], NULL);
    free(decoders); free(writers); free(bgra); free(answer);
    return c->failed ? 1 : 0;
}
#endif  /* !_WIN32 */

static int replay(nr_frame *frame, const nr_frame_params *p, const nr_frame_live_settings *s,
                  const char *source, const char *out, realtime_pace *pace, int pool)
{
    struct numbers frames = { 0 };
    if (list_dir(source, collect, &frames)) { perror(source); return 1; }
    if (!frames.count) { fprintf(stderr, "no NNN_in.png in %s\n", source); return 1; }
    qsort(frames.values, frames.count, sizeof *frames.values, compare_long);
    for (size_t k = 1; k < frames.count; k++)
        if (frames.values[k] != frames.values[0] + (long)k) {
            /* a gap means frames the daemon saw and this replay would not: its history differs */
            fprintf(stderr, "%s: the frames are not consecutive (%ld-%ld, %u of them)\n", source,
                    frames.values[0], frames.values[frames.count - 1], (unsigned)frames.count);
            return 1;
        }
    make_dir(out);
    struct numbers there = { 0 };
    if (list_dir(out, collect, &there)) { perror(out); return 1; }
    if (there.count || there.others) {
        fprintf(stderr, "%s is not empty: the dump is numbered from 001, as the daemon's is\n", out);
        return 1;
    }
    free(there.values);

    nr_frame_live *live = nr_frame_live_open(frame);
    if (!live) { fprintf(stderr, "nr_frame_live_open: %s\n", nr_frame_error()); return 1; }
    printf("  %s -> %s, %u frames, render scale %g, temporal %g, hold %g, release %g, cut limit %g\n",
           source, out, (unsigned)frames.count, (double)s->render_scale, (double)s->temporal, (double)s->hold,
           (double)s->release, (double)s->cut_limit);
    fflush(stdout);
    double *samples = malloc(frames.count * sizeof *samples);
    long kept = 0;                  /* the processed frames' own 1-based numbering */
    int rc = 0;

#ifndef _WIN32
    if (pool > 1) {
        /* With --realtime the selection is the frames the pace keeps; without it, every frame
         * is run and the output is the input's own rate. */
        int select = pace && pace->on;
        long start = 0;
        if (select && pace->rate <= 0) {
            /* An auto rate needs a few frames measured before it is trusted, and those frames
             * must share the pipeline's history, so they run here first, on `live`. */
            long cal = (long)frames.count < POOL_CALIB ? (long)frames.count : POOL_CALIB;
            for (long k = 0; k < cal; k++) {
                double ms = 0;
                if (replay_frame(live, p, s, source, out, frames.values[k], kept + 1, &ms)) { rc = 1; goto finish; }
                if (samples) samples[kept] = ms;
                kept++;
                pace_record(pace, ms);
            }
            double *warm = malloc((size_t)cal * sizeof *warm);
            if (warm) { for (long i = 0; i < cal; i++) warm[i] = samples[i];
                qsort(warm, (size_t)cal, sizeof *warm, compare_double);
                double med = warm[cal / 2];
                free(warm);
                pace->rate = med > 0 ? 1000.0 / med : 1.0;
                if (pace->rate > pace->input_fps) pace->rate = pace->input_fps;
                if (pace->rate < 0.5) pace->rate = 0.5;
            }
            pace->processed = cal;
            pace->held = 0;
            start = cal;
        }
        long cap = (long)frames.count - start + 1;
        long *list = malloc((size_t)cap * sizeof *list);
        long n = 0;
        if (!list) die("out of memory");
        if (select) {
            /* The frames the pace would process from `start` on: a fixed rate, so the whole
             * selection is known before a byte is read. */
            double next_t = (double)start / pace->input_fps;
            for (long k = start; k < (long)frames.count; k++) {
                double t = (double)k / pace->input_fps;
                if (t + 1e-9 < next_t) continue;
                list[n++] = frames.values[k];
                next_t = t + 1.0 / pace->rate;
            }
            pace->held += ((long)frames.count - start) - n;
            pace->processed += n;
        } else {
            for (long k = 0; k < (long)frames.count; k++) list[n++] = frames.values[k];
        }

        pool_ctx c;
        memset(&c, 0, sizeof c);
        c.live = live; c.p = p; c.s = s; c.source = source; c.out = out;
        c.numbers = list; c.count = n; c.base = kept + 1;
        c.pool = pool; c.decode_threads = POOL_DECODERS; c.write_threads = POOL_WRITERS;
        {
            const char *e = getenv("NR_POOL_DECODERS");
            if (e && atoi(e) > 0) c.decode_threads = atoi(e);
            e = getenv("NR_POOL_WRITERS");
            if (e && atoi(e) > 0) c.write_threads = atoi(e);
        }
        c.samples = samples ? samples + kept : NULL;
        c.slots = calloc((size_t)pool, sizeof *c.slots);
        if (!c.slots) die("out of memory");
        for (int i = 0; i < pool; i++) { c.slots[i].pos = -1; c.slots[i].state = SLOT_FREE; }
        pthread_mutex_init(&c.m, NULL);
        pthread_cond_init(&c.cv, NULL);
        printf("  pool: %ld frames to run, %d slots, %d decoders, %d writers\n",
               n, pool, c.decode_threads, c.write_threads);
        fflush(stdout);
        rc = pool_run(&c);
        pthread_mutex_destroy(&c.m);
        pthread_cond_destroy(&c.cv);
        free(c.slots);
        free(list);
        kept += n;
        goto finish;
    }
#else
    (void)pool;
#endif

    for (size_t k = 0; k < frames.count; k++) {
        if (pace && !pace_process(pace, (long)k)) { pace->held++; continue; }
        double ms = 0;
        if (replay_frame(live, p, s, source, out, frames.values[k], kept + 1, &ms)) { rc = 1; goto finish; }
        pace_record(pace, ms);
        if (samples) samples[kept] = ms;
        kept++;
    }

finish:
    nr_frame_live_close(live);
    if (samples) { report_frame_times(samples, (size_t)kept); free(samples); }
    pace_report(pace, frames.count);
    free(frames.values);
    return rc;
}

/* --stream WxH: raw B8G8R8A8 frames on stdin, the answers on stdout, through the same
 * nr_frame_live (history and all) as --replay - no PNG, no disk. Made for ffmpeg:
 *   ffmpeg -i in.mp4 -f rawvideo -pix_fmt bgra - | nr_frame --stream 1928x1114 |
 *   ffmpeg -f rawvideo -pix_fmt bgra -s 1928x1114 -r 60 -i - out.mp4
 * One stderr line a frame: wall ms of the run, and the running mean. */
static int stream(nr_frame *frame, const nr_frame_params *p, const nr_frame_live_settings *s,
                  int width, int height, realtime_pace *pace)
{
    size_t bytes = (size_t)width * height * 4;
    unsigned char *bgra = malloc(bytes), *answer = malloc(bytes);
    if (!bgra || !answer) die("out of memory");
    nr_frame_live *live = nr_frame_live_open(frame);
    if (!live) { fprintf(stderr, "nr_frame_live_open: %s\n", nr_frame_error()); return 1; }
    double total = 0;
    long k = 0;
    /* Always one frame out per frame in: the frames dropped under --realtime repeat the
     * last answer, so the container keeps the input's own frame rate and the stream stays
     * on the input's clock - it is the content that runs slower, not the video. */
    while (fread(bgra, 1, bytes, stdin) == bytes) {
        nr_frame_live_report r;
        double ms = 0;
        if (pace_process(pace, k)) {
            double started = now();
            if (nr_frame_live_run(live, p, s, bgra, width, height, 1, answer, &r)) {
                fprintf(stderr, "frame %ld: nr_frame_live_run: %s\n", k + 1, nr_frame_error());
                return 1;
            }
            ms = (now() - started) * 1e3;
            pace_record(pace, ms);
        } else {
            pace->held++;
        }
        total += ms;
        k++;
        if (fwrite(answer, 1, bytes, stdout) != bytes) { perror("stdout"); return 1; }
        fprintf(stderr, "frame %ld  %s  %.1f ms  mean %.1f ms of a run frame (%.2f fps)\n", k,
                ms > 0 ? "processed" : "held", ms,
                pace && pace->processed ? total / pace->processed : 0.0,
                pace && pace->processed ? 1e3 * pace->processed / total : 0.0);
    }
    fflush(stdout);
    nr_frame_live_close(live);
    free(bgra); free(answer);
    fprintf(stderr, "%ld frames, mean %.1f ms (%.2f fps)  graph time, excludes I/O\n",
            k, pace && pace->processed ? total / pace->processed : 0.0,
            pace && pace->processed ? 1e3 * pace->processed / total : 0.0);
    pace_report(pace, (size_t)k);
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: nr_frame IN OUT [--weights W] [--size HxW] [--profile standard|natural|cinematic|neutral|vendor]\n"
        "                (with no W, the weights compiled into libnr_frame; otherwise work/mlxw/dlssnr-logical.safetensors)\n"
        "                [--style-index N] [--local-tone F] [--local-structure F] [--skin-structure F]\n"
        "                [--auto-mask F] [--control-mask MASK] [--intensity F] [--intensity-ladder A,B,..]\n"
        "                [--detail-strength F] [--colour-strength F] [--detail-radius F] [--frame-index N] [-v]\n"
        "       nr_frame --replay DUMP OUT [--render-scale F] [--temporal F] [--hold F] [--release F]\n"
        "                [--cut-limit F] [--min-extent N] [--no-letterbox] [--profile P] [--intensity F] ...\n"
        "                (a nr_daemon.py --dump capture through the daemon's frame, history and all;\n"
        "                 render scale 1 unless set, the other knobs the daemon's defaults)\n"
        "       nr_frame --stream WxH [the --replay knobs]\n"
        "                (raw BGRA frames stdin -> stdout through the same live frame; per-frame ms on stderr)\n"
        "       ... --realtime [FPS] [--input-fps F]\n"
        "                (with --replay/--stream: drop frames so the output runs on the input's clock.\n"
        "                 No FPS measures the model's own rate and follows it; input-fps is 60.\n"
        "                 --replay writes one in/out per kept frame; --stream keeps the input's rate\n"
        "                 and repeats the last answer through the dropped ones)\n"
        "       ... [--pool N]\n"
        "                (with --replay --realtime: keep N decoded frames in memory, read ahead of the\n"
        "                 model and write behind it, so the model runs back to back on the card)\n"
        "       nr_frame --prepare OUT.safetensors [--weights W]\n"
        "                (the weights written with the loaders' layout work done, for faster opens;\n"
        "                 derived from NVIDIA's weights, so never committed or redistributed)\n");
    exit(2);
}

int main(int argc, char **argv)
{
    const char *input = NULL, *output = NULL, *weights = NULL, *size = NULL, *ladder = NULL, *mask_path = NULL;
    const char *profile_name = "standard";
    int verbose = 0, have_style = 0, style_index = 0, have_tone = 0, have_structure = 0;
    float local_tone = 0, local_structure = 0;
    int have_skin = 0, have_auto = 0;
    float skin = -1.0f, automatic = -1.0f;
    nr_frame_params p;
    nr_frame_defaults(&p);
    int replaying = 0;
    const char *prepare_out = NULL, *streaming = NULL;
    int realtime_on = 0;
    double realtime_rate = 0;   /* 0: measure; > 0: fixed frames a second */
    double input_fps = 60.0;
    int pool = 0;               /* --pool N: frames decoded ahead of the model */
    nr_frame_live_settings live;
    nr_frame_live_defaults(&live);

#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), (char *)0))
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--weights")) weights = NEXT();
        else if (!strcmp(a, "--size")) size = NEXT();
        else if (!strcmp(a, "--profile")) profile_name = NEXT();
        else if (!strcmp(a, "--style-index")) { have_style = 1; style_index = atoi(NEXT()); }
        else if (!strcmp(a, "--local-tone")) { have_tone = 1; local_tone = (float)atof(NEXT()); }
        else if (!strcmp(a, "--local-structure")) { have_structure = 1; local_structure = (float)atof(NEXT()); }
        else if (!strcmp(a, "--skin-structure")) { have_skin = 1; skin = (float)atof(NEXT()); }
        else if (!strcmp(a, "--auto-mask")) { have_auto = 1; automatic = (float)atof(NEXT()); }
        else if (!strcmp(a, "--control-mask")) mask_path = NEXT();
        else if (!strcmp(a, "--intensity")) p.intensity = (float)atof(NEXT());
        else if (!strcmp(a, "--intensity-ladder")) ladder = NEXT();
        else if (!strcmp(a, "--detail-strength")) p.detail_strength = (float)atof(NEXT());
        else if (!strcmp(a, "--colour-strength")) p.colour_strength = (float)atof(NEXT());
        else if (!strcmp(a, "--detail-radius")) p.detail_radius = (float)atof(NEXT());
        else if (!strcmp(a, "--frame-index")) p.frame_index = atoi(NEXT());
        else if (!strcmp(a, "--replay")) replaying = 1;
        else if (!strcmp(a, "--stream")) streaming = NEXT();
        else if (!strcmp(a, "--realtime")) {
            realtime_on = 1;
            realtime_rate = 0;          /* auto: measure the model */
            if (i + 1 < argc) {         /* an optional rate right after it */
                char *end;
                double v = strtod(argv[i + 1], &end);
                if (end != argv[i + 1] && *end == '\0' && v > 0) { realtime_rate = v; i++; }
            }
        }
        else if (!strcmp(a, "--input-fps")) input_fps = atof(NEXT());
        else if (!strcmp(a, "--pool")) pool = atoi(NEXT());
        else if (!strcmp(a, "--prepare")) prepare_out = NEXT();
        else if (!strcmp(a, "--render-scale")) live.render_scale = (float)atof(NEXT());
        else if (!strcmp(a, "--temporal")) live.temporal = (float)atof(NEXT());
        else if (!strcmp(a, "--hold")) live.hold = (float)atof(NEXT());
        else if (!strcmp(a, "--release")) live.release = (float)atof(NEXT());
        else if (!strcmp(a, "--cut-limit")) live.cut_limit = (float)atof(NEXT());
        else if (!strcmp(a, "--min-extent")) p.min_extent = atoi(NEXT());
        else if (!strcmp(a, "--no-letterbox")) live.letterbox = 0;
        else if (!strcmp(a, "--gpu") || !strcmp(a, "--resident") || !strcmp(a, "--accel"))
            ;   /* one backend here: the resident graph */
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (a[0] == '-' && a[1]) usage();
        else if (!input) input = a;
        else if (!output) output = a;
        else usage();
    }
    if (prepare_out) {
        /* no device: the weights in, the prepared file out */
        if (nr_frame_prepare(weights, prepare_out)) {
            fprintf(stderr, "nr_frame_prepare: %s\n", nr_frame_error());
            return 1;
        }
        fprintf(stderr, "prepared weights written to %s\n", prepare_out);
        return 0;
    }
    if (!streaming && (!input || !output)) usage();
    if (profile(profile_name, &p)) { fprintf(stderr, "unknown profile %s; one of", profile_name);
        for (size_t k = 0; k < sizeof PROFILES / sizeof *PROFILES; k++) fprintf(stderr, " %s", PROFILES[k]);
        fprintf(stderr, "\n"); return 2; }
    /* `nr_frame.controls`: the profile, then the explicit overrides */
    if (have_style) p.normalized_style = (float)style_index / 128.0f;
    if (have_tone) p.local_tone = local_tone;
    if (have_structure) p.local_structure = local_structure;
    if (have_skin || have_auto) { p.automatic_mask = 1; p.skin_structure = skin; p.automatic_structure = automatic; }

    char default_path[1200];
    if (!weights) weights = default_weights(default_path, sizeof default_path);

    double started = now();
    nr_frame *frame = nr_frame_open(weights);
    if (!frame) { fprintf(stderr, "nr_frame_open: %s\n", nr_frame_error()); return 1; }
    /* stdout is the frame stream under --stream */
    fprintf(streaming ? stderr : stdout, "resident backend ready in %.1fs (%s) on %s\n", now() - started,
            weights ? weights : "embedded weights", nr_frame_device(frame));
    fflush(stdout);

    if (replaying || streaming) {
        if (!(live.render_scale >= 0.05f && live.render_scale <= 1.0f)) die("--render-scale is 0.05-1");
        if (!(live.temporal >= 0.0f && live.temporal <= 1.0f) || !(live.hold >= 0.0f && live.hold <= 1.0f)
            || !(live.cut_limit >= 0.0f && live.cut_limit <= 1.0f)) die("--temporal, --hold and --cut-limit are 0-1");
        if (!(live.release >= 0.0f && live.release <= 255.0f)) die("--release is 0-255");
        if (p.min_extent < 128 || p.min_extent > 4096) die("--min-extent is 128-4096, as the daemon takes it");
        int sw, sh;
        if (streaming && (sscanf(streaming, "%dx%d", &sw, &sh) != 2 || sw <= 0 || sh <= 0))
            die("--stream takes WxH");
        realtime_pace pace;
        memset(&pace, 0, sizeof pace);
        pace.on = realtime_on;
        pace.rate = realtime_rate;
        pace.input_fps = input_fps;
        if (realtime_on) {
            if (!(input_fps > 0)) die("--input-fps must be positive");
            fprintf(stderr, "realtime: %s, input %.4g fps\n",
                    realtime_rate > 0 ? "a fixed rate" : "measuring the model", input_fps);
        }
        int rc = streaming ? stream(frame, &p, &live, sw, sh, &pace)
                           : replay(frame, &p, &live, input, output, &pace, pool);
        nr_frame_close(frame);
        return rc;
    }

    int height, width, resize_h = 0, resize_w = 0;
    if (size && sscanf(size, "%dx%d", &resize_h, &resize_w) != 2) usage();
    float *colour = load_image(input, &height, &width, resize_h, resize_w);
    printf("input %dx%d\n", width, height);
    fflush(stdout);
    float *mask = NULL;
    if (mask_path) {
        int mh, mw;
        mask = load_image(mask_path, &mh, &mw, 0, 0);
        if (mh != height || mw != width) die("the control mask must match the colour image shape");
    }

    int H, W;
    nr_frame_geometry(height, width, &H, &W);
    if (verbose) { printf("  network extent %dx%d for output %dx%d\n", W, H, width, height); fflush(stdout); }
    size_t pixels = (size_t)height * width;
    float *features = malloc((size_t)H * W * 16 * sizeof(float));
    float *wide = malloc((size_t)H * W * 4 * sizeof(float));
    float *head = malloc(pixels * 4 * sizeof(float));
    float *composed = malloc(pixels * 3 * sizeof(float));
    if (!features || !wide || !head || !composed) die("out of memory");

    started = now();
    if (nr_frame_features_masked(frame, colour, height, width, NULL, mask, &p, features)
        || nr_frame_run_features(frame, features, H, W, wide)) {
        fprintf(stderr, "%s\n", nr_frame_error());
        return 1;
    }
    for (int y = 0; y < height; y++)               /* `geometry.crop(head)` */
        memcpy(head + (size_t)y * width * 4, wide + (size_t)y * W * 4, (size_t)width * 4 * sizeof(float));
    double elapsed = now() - started;
    printf("network %.1fs\n", elapsed);
    if (verbose)
        printf("    write %.1f ms, graph %.1f ms, read %.1f ms\n", nr_frame_split(frame, 0) * 1e3,
               nr_frame_split(frame, 1) * 1e3, nr_frame_split(frame, 2) * 1e3);
    double lo = head[0], hi = head[0], sum = 0, sumsq = 0;
    for (size_t i = 0; i < pixels * 4; i++) {
        double v = head[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        sum += v; sumsq += v * v;
    }
    double mean = sum / (double)(pixels * 4);
    printf("head    min %+.4f max %+.4f sd %.4f\n", lo, hi, sqrt(sumsq / (double)(pixels * 4) - mean * mean));
    fflush(stdout);

    /* one render, one file per intensity */
    float ladder_values[64];
    int count = 0;
    if (ladder) {
        char *copy = nr_strdup(ladder), *save = NULL;
        for (char *tok = nr_strtok_r(copy, ",", &save); tok && count < 64; tok = nr_strtok_r(NULL, ",", &save))
            ladder_values[count++] = (float)atof(tok);
        free(copy);
    } else {
        ladder_values[count++] = p.intensity;
    }
    for (int k = 0; k < count; k++) {
        nr_frame_params q = p;
        q.intensity = ladder_values[k];
        if (nr_frame_compose(frame, head, colour, height, width, NULL, NULL, mask, &q, composed)) {
            fprintf(stderr, "nr_frame_compose: %s\n", nr_frame_error());
            return 1;
        }
        double total = 0, worst = 0;
        for (size_t i = 0; i < pixels * 3; i++) {
            double d = fabs((double)composed[i] - (double)colour[i]);
            total += d;
            if (d > worst) worst = d;
        }
        printf("intensity %5.2f  change mean|d| %.5f max|d| %.5f\n", q.intensity, total / (double)(pixels * 3), worst);
        char path[1200];
        if (ladder) {
            /* `stem_i{value:g}{suffix}` */
            const char *dot = strrchr(output, '.');
            const char *slash = strrchr(output, '/');
            if (dot && slash && dot < slash) dot = NULL;
            int stem = dot ? (int)(dot - output) : (int)strlen(output);

#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
            sprintf_s(path, sizeof path, "%.*s_i%g%s", stem, output, (double)q.intensity, dot ? dot : "");
#else
            snprintf(path, sizeof path, "%.*s_i%g%s", stem, output, (double)q.intensity, dot ? dot : "");
#endif
        } else {
#if defined(_WIN32) && __STDC_WANT_SECURE_LIB__
            sprintf_s(path, sizeof path, "%s", output);
#else
            snprintf(path, sizeof path, "%s", output);
#endif
        }
        save_image(composed, height, width, path);
        printf("wrote %s\n", path);
        fflush(stdout);
    }
    nr_frame_close(frame);
    free(colour); free(mask); free(features); free(wide); free(head); free(composed);
    return 0;
}
