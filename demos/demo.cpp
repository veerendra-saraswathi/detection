/*
 * Smoke demo for the RCI detection cascade.
 *
 *   cmake -S . -B build && cmake --build build
 *   ./build/rci_detect_demo
 *   ./build/rci_detect_demo frame.raw 640 480
 *   ./build/rci_detect_demo frame.raw 640 480 rgb
 *
 * With no arguments the demo paints three bright shapes on a dark field
 * and searches for Person, Car, and Truck. A raw file is tightly packed
 * gray8 or rgb8 with no header. Width and height are in pixels.
 *
 * The shapes sit on the default stride grid and the first scale matches
 * their size, so the stub pyramid can land on them. Stage lines report
 * the candidate count and the monotonic time spent in each cascade stage.
 */

#include "detect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Shape {
    int32_t id;
    const char *name;
    int x;
    int y;
    int w;
    int h;
    uint8_t value;
    float min_confidence;
};

/* Sizes match the search pyramid. Values are well above the dark field. */
const Shape kShapes[] = {
    {3, "Person", 32, 24, 24, 60, 240, 0.35f},
    {2, "Car", 128, 96, 80, 32, 210, 0.40f},
    {1, "Truck", 200, 40, 96, 48, 190, 0.45f},
};

constexpr int kSynthW = 320;
constexpr int kSynthH = 240;

uint32_t lcg(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

void fill_rect(std::vector<uint8_t> &img, int W, int x, int y, int w, int h, uint8_t value) {
    const int H = static_cast<int>(img.size() / static_cast<size_t>(W));
    for (int yy = y; yy < y + h && yy < H; ++yy) {
        if (yy < 0) continue;
        for (int xx = x; xx < x + w && xx < W; ++xx) {
            if (xx < 0) continue;
            img[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)] = value;
        }
    }
}

std::vector<uint8_t> make_synthetic() {
    std::vector<uint8_t> img(static_cast<size_t>(kSynthW) * static_cast<size_t>(kSynthH), 18);
    uint32_t state = 0xA341316Cu;
    for (uint8_t &p : img) {
        const int noise = static_cast<int>((lcg(state) >> 28) & 7u) - 3; /* -3..4 */
        int v = 18 + noise;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        p = static_cast<uint8_t>(v);
    }
    for (const Shape &s : kShapes) {
        fill_rect(img, kSynthW, s.x, s.y, s.w, s.h, s.value);
    }
    return img;
}

bool load_raw(const char *path, int64_t nbytes, std::vector<uint8_t> &out) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        std::fprintf(stderr, "cannot read %s\n", path);
        return false;
    }
    const long n = std::ftell(f);
    if (n < 0 || static_cast<int64_t>(n) != nbytes) {
        std::fclose(f);
        std::fprintf(stderr, "%s has %ld bytes, expected %lld\n", path, n,
                     static_cast<long long>(nbytes));
        return false;
    }
    if (std::fseek(f, 0, SEEK_SET) != 0) {
        std::fclose(f);
        std::fprintf(stderr, "cannot read %s\n", path);
        return false;
    }
    out.resize(static_cast<size_t>(nbytes));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    if (got != out.size()) {
        std::fprintf(stderr, "short read on %s\n", path);
        return false;
    }
    return true;
}

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [raw_path width height [gray|rgb]]\n"
                 "  no args          synthetic 320x240 gray scene\n"
                 "  raw_path W H     tightly packed gray8\n"
                 "  raw_path W H rgb tightly packed rgb8\n",
                 argv0);
}

bool center_hits(const rci_detection &d, const Shape &s) {
    if (d.type != s.id) return false;
    const float cx = d.x + d.width * 0.5f;
    const float cy = d.y + d.height * 0.5f;
    constexpr float margin = 12.f;
    return cx >= static_cast<float>(s.x) - margin &&
           cx < static_cast<float>(s.x + s.w) + margin &&
           cy >= static_cast<float>(s.y) - margin &&
           cy < static_cast<float>(s.y + s.h) + margin;
}

} /* namespace */

int main(int argc, char **argv) {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int format = RCI_PIXFMT_GRAY8;
    bool synthetic = false;

    if (argc == 1) {
        pixels = make_synthetic();
        width = kSynthW;
        height = kSynthH;
        synthetic = true;
    } else if (argc == 4 || argc == 5) {
        width = static_cast<int>(std::strtol(argv[2], nullptr, 10));
        height = static_cast<int>(std::strtol(argv[3], nullptr, 10));
        if (argc == 5) {
            if (std::strcmp(argv[4], "rgb") == 0) format = RCI_PIXFMT_RGB8;
            else if (std::strcmp(argv[4], "gray") == 0) format = RCI_PIXFMT_GRAY8;
            else {
                usage(argv[0]);
                return 2;
            }
        }
        if (width <= 0 || height <= 0) {
            usage(argv[0]);
            return 2;
        }
        const int bpp = format == RCI_PIXFMT_RGB8 ? 3 : 1;
        const int64_t nbytes =
            static_cast<int64_t>(width) * static_cast<int64_t>(height) * bpp;
        if (!load_raw(argv[1], nbytes, pixels)) return 1;
    } else {
        usage(argv[0]);
        return 2;
    }

    rci_object_def defs[3];
    std::memset(defs, 0, sizeof defs);
    for (int i = 0; i < 3; ++i) {
        defs[i].id = kShapes[i].id;
        defs[i].name = kShapes[i].name;
        defs[i].min_width = kShapes[i].w;
        defs[i].min_height = kShapes[i].h;
        defs[i].max_width = kShapes[i].w * 2;
        defs[i].max_height = kShapes[i].h * 2;
        defs[i].min_confidence = kShapes[i].min_confidence;
    }

    rci_detect_params params;
    rci_detect_params_init(&params);

    rci_image image;
    std::memset(&image, 0, sizeof image);
    image.data = pixels.data();
    image.width = width;
    image.height = height;
    image.stride = 0;
    image.format = format;

    rci_detection dets[64];
    size_t count = 0;
    rci_detect_stats stats;
    std::memset(&stats, 0, sizeof stats);
    const int rc = rci_detect_profiled(&image, defs, 3, &params, dets, 64, &count, &stats);
    if (rc != RCI_OK) {
        std::fprintf(stderr, "rci_detect failed: %d\n", rc);
        return 1;
    }

    const char *fmt_name = format == RCI_PIXFMT_RGB8 ? "rgb8" : "gray8";
    std::printf("RCI automatic object detection (cascade stub) %s\n", RCI_DETECT_VERSION);
    if (synthetic) {
        std::printf("image: synthetic %dx%d %s\n", width, height, fmt_name);
    } else {
        std::printf("image: %s %dx%d %s\n", argv[1], width, height, fmt_name);
    }
    std::printf("objects: Person, Car, Truck\n");
    std::printf("params: edge %.2f  nms %.2f  scales %d  step %.2f  stride %d\n",
                params.edge_threshold, params.nms_iou, params.scales, params.scale_step,
                params.stride);
    std::printf("stage 1 propose   %6d candidates   %8.3f ms\n", stats.candidate_count,
                stats.propose_ms);
    std::printf("stage 2 verify    %6d accepted     %8.3f ms\n", stats.verified_count,
                stats.verify_ms);
    std::printf("raw candidates before nms: %d\n", stats.verified_count);
    std::printf("stage 3 nms       %6d kept         %8.3f ms\n", stats.kept_count, stats.nms_ms);
    std::printf("detections: %zu\n\n", count);
    std::printf("%-4s %-16s %7s %7s %7s %7s %8s\n", "id", "type", "x", "y", "w", "h", "conf");
    for (size_t i = 0; i < count; ++i) {
        const rci_detection &d = dets[i];
        std::printf("%-4d %-16.16s %7.1f %7.1f %7.1f %7.1f %8.3f\n", d.id, d.type_name, d.x, d.y,
                    d.width, d.height, d.confidence);
    }

    if (!synthetic) return 0;

    int matched = 0;
    for (const Shape &s : kShapes) {
        bool hit = false;
        for (size_t i = 0; i < count; ++i) {
            if (center_hits(dets[i], s)) {
                hit = true;
                break;
            }
        }
        if (hit) ++matched;
    }
    const int nshapes = static_cast<int>(sizeof kShapes / sizeof kShapes[0]);
    std::printf("\nsynthetic check: %d/%d objects have a detection\n", matched, nshapes);
    return matched == nshapes ? 0 : 1;
}
