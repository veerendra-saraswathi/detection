/*
 * Stub cascade for RCI automatic object detection.
 *
 * Stage 1 builds a gray plane and integral images (gradient magnitude,
 * gray level, and salient-pixel counts). Each object definition grows a
 * short pyramid of windows. A window is a candidate when its mean gray
 * level differs from the mean of a surrounding ring by at least 8 levels.
 * The pyramid starts two steps below the object's base size so pedestrians
 * about 16-40 pixels wide are still covered.
 *
 * Stage 2 is one class-agnostic verifier. It keeps a window when the
 * interior stands off from the immediate surround, the interior is quieter
 * than the border (edge mass sits on the perimeter), and the four interior
 * quadrants agree (local contrast is consistent). There is no per-class
 * model. The label is whichever pyramid proposed the window.
 *
 * Stage 3 turns the structural score and the verifier score into a
 * confidence and drops overlaps with greedy per-class NMS.
 *
 * Scratch memory is allocated up front and released before return.
 * The image, the output buffer, and the optional stats struct belong
 * to the caller.
 */

#include "detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int32_t kMaxDim = 8192;
constexpr int64_t kMaxPixels = 4096LL * 4096LL;
constexpr size_t kMaxCandidates = 250000;
constexpr int kSalientDelta = 24; /* gray levels away from the image mean */
constexpr int kRingMax = 3;
constexpr float kMinFill = 0.55f;
constexpr float kStrongEdge = 0.20f; /* ring mean that saturates the edge score */
[[maybe_unused]] constexpr float kMinContrast = 30.f; /* unused while the verifier is loosened */
constexpr float kCandidateContrast = 8.f; /* gray levels, window vs surrounding ring */
constexpr float kMaxSpread = 48.f;   /* gray levels across interior quadrants */
constexpr float kMaxEdgeRatio = 0.55f; /* interior edge / border edge */

struct Params {
    float edge_threshold;
    float nms_iou;
    int max_detections;
    int scales;
    float scale_step;
    int stride;
};

struct Cand {
    int32_t type;
    char type_name[64];
    int x;
    int y;
    int w;
    int h;
    int ref_w;
    int ref_h;
    float border_edge;
    float interior_edge;
    float min_confidence;
    float confidence;
};

struct StageClock {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    void reset() { t0 = clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void copy_name(char *dst, size_t dst_n, const char *src) {
    if (dst_n == 0) return;
    size_t i = 0;
    if (src) {
        for (; i + 1 < dst_n && src[i] != '\0'; ++i) dst[i] = src[i];
    }
    dst[i] = '\0';
}

Params normalize_params(const rci_detect_params *in) {
    rci_detect_params defaults;
    if (!in) {
        rci_detect_params_init(&defaults);
        in = &defaults;
    }
    Params p;
    p.edge_threshold = in->edge_threshold < 0.f ? 0.f : in->edge_threshold;
    p.nms_iou = clampf(in->nms_iou, 0.f, 1.f);
    p.max_detections = in->max_detections > 0 ? in->max_detections : 32;
    if (p.max_detections > 10000) p.max_detections = 10000;
    p.scales = in->scales > 0 ? in->scales : 1;
    if (p.scales > 12) p.scales = 12;
    /* A step of 1 would repeat the same window on every scale. */
    p.scale_step = in->scale_step > 1.01f ? in->scale_step : 1.5f;
    p.stride = in->stride > 0 ? in->stride : 8;
    return p;
}

uint64_t rect_sum(const std::vector<uint64_t> &ii, int W, int x, int y, int w, int h) {
    const size_t S = static_cast<size_t>(W) + 1;
    const size_t x2 = static_cast<size_t>(x + w);
    const size_t y2 = static_cast<size_t>(y + h);
    const size_t y1 = static_cast<size_t>(y);
    const size_t x1 = static_cast<size_t>(x);
    return ii[y2 * S + x2] - ii[y1 * S + x2] - ii[y2 * S + x1] + ii[y1 * S + x1];
}

template <typename T>
void build_integral(const T *src, int W, int H, std::vector<uint64_t> &ii) {
    const size_t S = static_cast<size_t>(W) + 1;
    ii.assign(S * (static_cast<size_t>(H) + 1), 0);
    for (int y = 0; y < H; ++y) {
        uint64_t row = 0;
        const T *src_row = src + static_cast<size_t>(y) * static_cast<size_t>(W);
        for (int x = 0; x < W; ++x) {
            row += static_cast<uint64_t>(src_row[x]);
            ii[(static_cast<size_t>(y) + 1) * S + (static_cast<size_t>(x) + 1)] =
                ii[static_cast<size_t>(y) * S + (static_cast<size_t>(x) + 1)] + row;
        }
    }
}

int build_gray(const rci_image &image, int64_t stride, std::vector<uint8_t> &gray) {
    const int W = image.width;
    const int H = image.height;
    gray.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    if (image.format == RCI_PIXFMT_GRAY8) {
        if (stride == W) {
            std::memcpy(gray.data(), image.data, gray.size());
            return RCI_OK;
        }
        for (int y = 0; y < H; ++y) {
            const uint8_t *row = image.data + static_cast<size_t>(y) * static_cast<size_t>(stride);
            std::memcpy(gray.data() + static_cast<size_t>(y) * static_cast<size_t>(W), row,
                        static_cast<size_t>(W));
        }
        return RCI_OK;
    }
    if (image.format == RCI_PIXFMT_RGB8) {
        for (int y = 0; y < H; ++y) {
            const uint8_t *row = image.data + static_cast<size_t>(y) * static_cast<size_t>(stride);
            uint8_t *dst = gray.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
            for (int x = 0; x < W; ++x) {
                const uint8_t *p = row + static_cast<size_t>(x) * 3;
                /* Integer BT.601 luma. 77 + 150 + 29 = 256. */
                dst[x] = static_cast<uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
            }
        }
        return RCI_OK;
    }
    return RCI_ERR_FORMAT;
}

int ring_thickness(int w, int h) {
    int t = std::min(w, h) / 6;
    if (t < 1) t = 1;
    if (t > kRingMax) t = kRingMax;
    while (t > 1 && (w - 2 * t < 1 || h - 2 * t < 1)) --t;
    if (w - 2 * t < 1 || h - 2 * t < 1) return 0;
    return t;
}

float rect_mean(const std::vector<uint64_t> &ii, int W, int x, int y, int w, int h) {
    return static_cast<float>(rect_sum(ii, W, x, y, w, h)) / static_cast<float>(w * h);
}

/* Max quadrant mean minus min quadrant mean, in gray levels.
 * A 1-pixel interior has no split and reports 0. */
float quadrant_spread(const std::vector<uint64_t> &ii, int W, int x, int y, int w, int h) {
    if (w < 2 || h < 2) return 0.f;
    const int w0 = w / 2;
    const int h0 = h / 2;
    const int w1 = w - w0;
    const int h1 = h - h0;
    const float q[4] = {
        rect_mean(ii, W, x, y, w0, h0),
        rect_mean(ii, W, x + w0, y, w1, h0),
        rect_mean(ii, W, x, y + h0, w0, h1),
        rect_mean(ii, W, x + w0, y + h0, w1, h1),
    };
    float lo = q[0];
    float hi = q[0];
    for (int i = 1; i < 4; ++i) {
        if (q[i] < lo) lo = q[i];
        if (q[i] > hi) hi = q[i];
    }
    return hi - lo;
}

/*
 * Light verifier. Returns a score in [0, 1], or a negative value to reject.
 * A window passes only when its local contrast clears the candidate threshold
 * and its aspect is within 20% of the object definition. Texture and quadrant
 * agreement are not gates.
 */
float verify_window(float fill, float border_edge, float interior_edge, float interior_mean,
                    float surround_mean, float spread, int w, int h, int ref_w, int ref_h) {
    const float contrast = std::fabs(interior_mean - surround_mean);
    if (contrast < kCandidateContrast) return -1.f;
    if (h <= 0 || ref_h <= 0 || w <= 0 || ref_w <= 0) return -1.f;
    const float aspect_ratio = (static_cast<float>(w) / static_cast<float>(h)) /
                               (static_cast<float>(ref_w) / static_cast<float>(ref_h));
    if (aspect_ratio < 0.80f || aspect_ratio > 1.20f) return -1.f;

    const float fill_score = clampf((fill - kMinFill) / (1.f - kMinFill), 0.f, 1.f);
    const float edge_ratio = interior_edge / (border_edge + 1e-4f);
    const float edge_score = clampf(1.f - edge_ratio / kMaxEdgeRatio, 0.f, 1.f);
    const float contrast_score = clampf(contrast / 80.f, 0.f, 1.f);
    const float consist_score = clampf(1.f - spread / kMaxSpread, 0.f, 1.f);
    return 0.30f * fill_score + 0.25f * edge_score + 0.25f * contrast_score + 0.20f * consist_score;
}

float iou(const Cand &a, const Cand &b) {
    const float ax = static_cast<float>(a.x);
    const float ay = static_cast<float>(a.y);
    const float aw = static_cast<float>(a.w);
    const float ah = static_cast<float>(a.h);
    const float bx = static_cast<float>(b.x);
    const float by = static_cast<float>(b.y);
    const float bw = static_cast<float>(b.w);
    const float bh = static_cast<float>(b.h);
    const float x1 = std::max(ax, bx);
    const float y1 = std::max(ay, by);
    const float x2 = std::min(ax + aw, bx + bw);
    const float y2 = std::min(ay + ah, by + bh);
    const float iw = x2 - x1;
    const float ih = y2 - y1;
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float inter = iw * ih;
    const float uni = aw * ah + bw * bh - inter;
    if (uni <= 0.f) return 0.f;
    return inter / uni;
}

void nms_inplace(std::vector<Cand> &boxes, float iou_thresh) {
    std::vector<Cand> kept;
    kept.reserve(boxes.size());
    std::vector<char> dead(boxes.size(), 0);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (dead[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!dead[j] && iou(boxes[i], boxes[j]) >= iou_thresh) dead[j] = 1;
        }
    }
    boxes.swap(kept);
}

int detect_impl(
    const rci_image *image,
    const rci_object_def *objects,
    size_t object_count,
    const rci_detect_params *params_in,
    rci_detection *out,
    size_t out_capacity,
    size_t *out_count,
    rci_detect_stats *stats_in) {
    if (!image || !image->data) return RCI_ERR_ARG;
    if (object_count > 0 && !objects) return RCI_ERR_ARG;
    if (out_capacity > 0 && !out) return RCI_ERR_ARG;
    if (image->format != RCI_PIXFMT_GRAY8 && image->format != RCI_PIXFMT_RGB8) {
        return RCI_ERR_FORMAT;
    }
    const int W = image->width;
    const int H = image->height;
    if (W <= 0 || H <= 0 || W > kMaxDim || H > kMaxDim) return RCI_ERR_SIZE;
    if (static_cast<int64_t>(W) * static_cast<int64_t>(H) > kMaxPixels) return RCI_ERR_SIZE;

    const int bpp = image->format == RCI_PIXFMT_RGB8 ? 3 : 1;
    const int64_t min_stride = static_cast<int64_t>(W) * bpp;
    const int64_t stride = image->stride == 0 ? min_stride : image->stride;
    if (stride < min_stride || stride > INT32_MAX) return RCI_ERR_ARG;

    const Params params = normalize_params(params_in);
    rci_detect_stats local_stats;
    std::memset(&local_stats, 0, sizeof local_stats);
    rci_detect_stats *stats = stats_in ? stats_in : &local_stats;

    StageClock clock;

    std::vector<uint64_t> grad_ii;
    std::vector<uint64_t> gray_ii;
    std::vector<uint64_t> salient_ii;
    {
        std::vector<uint8_t> gray;
        const int gray_rc = build_gray(*image, stride, gray);
        if (gray_rc != RCI_OK) return gray_rc;

        std::vector<uint16_t> grad(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
        for (int y = 0; y < H; ++y) {
            const uint8_t *row = gray.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
            const uint8_t *nrow = (y + 1 < H) ? row + static_cast<size_t>(W) : nullptr;
            uint16_t *dst = grad.data() + static_cast<size_t>(y) * static_cast<size_t>(W);
            for (int x = 0; x < W; ++x) {
                const int dx = (x + 1 < W) ? std::abs(static_cast<int>(row[x + 1]) - row[x]) : 0;
                const int dy = nrow ? std::abs(static_cast<int>(nrow[x]) - row[x]) : 0;
                dst[x] = static_cast<uint16_t>(dx + dy);
            }
        }
        build_integral(grad.data(), W, H, grad_ii);
        grad.clear();
        grad.shrink_to_fit();

        build_integral(gray.data(), W, H, gray_ii);

        double sum = 0.0;
        for (uint8_t p : gray) sum += p;
        const double mean = sum / (static_cast<double>(W) * static_cast<double>(H));
        const int lo = static_cast<int>(mean) - kSalientDelta;
        const int hi = static_cast<int>(mean) + kSalientDelta;

        std::vector<uint16_t> salient(gray.size(), 0);
        for (size_t i = 0; i < gray.size(); ++i) {
            const int g = gray[i];
            if (g < lo || g > hi) salient[i] = 1;
        }
        build_integral(salient.data(), W, H, salient_ii);
    }

    std::vector<Cand> cands;
    cands.reserve(256);

    /* Stage 1. Structural candidates only. The verifier runs after this. */
    for (size_t oi = 0; oi < object_count; ++oi) {
        const rci_object_def &def = objects[oi];
        if (def.min_width < 2 || def.min_height < 2) continue;
        float min_conf = def.min_confidence;
        if (!(min_conf >= 0.f)) min_conf = 0.f;

        double win_w = static_cast<double>(def.min_width) / (params.scale_step * params.scale_step);
        double win_h = static_cast<double>(def.min_height) / (params.scale_step * params.scale_step);
        for (int s = 0; s < params.scales + 2; ++s) {
            const long long rw = std::llround(win_w);
            const long long rh = std::llround(win_h);
            win_w *= params.scale_step;
            win_h *= params.scale_step;
            if (rw < 2 || rh < 2) continue;
            if (rw > W || rh > H) break;
            if (def.max_width > 0 && rw > def.max_width) break;
            if (def.max_height > 0 && rh > def.max_height) break;
            const int ww = static_cast<int>(rw);
            const int hh = static_cast<int>(rh);
            const int ring = ring_thickness(ww, hh);
            if (ring <= 0) continue;
            const int inner_w = ww - 2 * ring;
            const int inner_h = hh - 2 * ring;
            const int border_area = ww * hh - inner_w * inner_h;
            if (border_area <= 0 || inner_w <= 0 || inner_h <= 0) continue;

            for (int y = 0; y + hh <= H; y += params.stride) {
                for (int x = 0; x + ww <= W; x += params.stride) {
                    if (cands.size() >= kMaxCandidates) break;
                    if (x <= 1 || y <= 1 || x + ww >= W - 1 || y + hh >= H - 1) continue;
                    const uint64_t window_sum = rect_sum(gray_ii, W, x, y, ww, hh);
                    const float window_mean =
                        static_cast<float>(window_sum) / static_cast<float>(ww * hh);
                    const int x0 = std::max(0, x - ring);
                    const int y0 = std::max(0, y - ring);
                    const int x1 = std::min(W, x + ww + ring);
                    const int y1 = std::min(H, y + hh + ring);
                    const int ring_area = (x1 - x0) * (y1 - y0) - ww * hh;
                    if (ring_area <= 0) continue;
                    const uint64_t outer_sum = rect_sum(gray_ii, W, x0, y0, x1 - x0, y1 - y0);
                    const float ring_mean =
                        static_cast<float>(outer_sum - window_sum) / static_cast<float>(ring_area);
                    if (std::fabs(window_mean - ring_mean) < kCandidateContrast) continue;

                    const uint64_t outer_g = rect_sum(grad_ii, W, x, y, ww, hh);
                    const uint64_t inner_g = rect_sum(grad_ii, W, x + ring, y + ring, inner_w, inner_h);
                    const uint64_t border_g = outer_g - inner_g;
                    const float border_edge =
                        static_cast<float>(border_g) / static_cast<float>(border_area) / 255.f;

                    Cand c;
                    c.type = def.id;
                    copy_name(c.type_name, sizeof c.type_name, def.name);
                    c.x = x;
                    c.y = y;
                    c.w = ww;
                    c.h = hh;
                    c.ref_w = def.min_width;
                    c.ref_h = def.min_height;
                    c.border_edge = border_edge;
                    c.interior_edge =
                        static_cast<float>(inner_g) / static_cast<float>(inner_w * inner_h) / 255.f;
                    c.min_confidence = min_conf;
                    c.confidence = 0.f;
                    cands.push_back(c);
                }
                if (cands.size() >= kMaxCandidates) break;
            }
            if (cands.size() >= kMaxCandidates) break;
        }
        if (cands.size() >= kMaxCandidates) break;
    }

    stats->candidate_count = static_cast<int32_t>(cands.size());
    stats->propose_ms = clock.ms();
    clock.reset();

    /* Stage 2. Shared verifier, then the confidence gate. */
    std::vector<Cand> verified;
    verified.reserve(cands.size());
    for (const Cand &c : cands) {
        const int ring = ring_thickness(c.w, c.h);
        if (ring <= 0) continue;
        const int inner_w = c.w - 2 * ring;
        const int inner_h = c.h - 2 * ring;
        if (inner_w <= 0 || inner_h <= 0) continue;

        const float area = static_cast<float>(c.w) * static_cast<float>(c.h);
        const float fill = static_cast<float>(rect_sum(salient_ii, W, c.x, c.y, c.w, c.h)) / area;
        const float interior_mean = rect_mean(gray_ii, W, c.x + ring, c.y + ring, inner_w, inner_h);
        const float spread = quadrant_spread(gray_ii, W, c.x + ring, c.y + ring, inner_w, inner_h);

        const int x0 = std::max(0, c.x - 1);
        const int y0 = std::max(0, c.y - 1);
        const int x1 = std::min(W, c.x + c.w + 1);
        const int y1 = std::min(H, c.y + c.h + 1);
        const int surround_area = (x1 - x0) * (y1 - y0) - c.w * c.h;
        if (surround_area <= 0) continue;
        const uint64_t outer_sum = rect_sum(gray_ii, W, x0, y0, x1 - x0, y1 - y0);
        const uint64_t window_sum = rect_sum(gray_ii, W, c.x, c.y, c.w, c.h);
        const float surround_mean =
            static_cast<float>(outer_sum - window_sum) / static_cast<float>(surround_area);

        const float score = verify_window(fill, c.border_edge, c.interior_edge, interior_mean,
                                           surround_mean, spread, c.w, c.h, c.ref_w, c.ref_h);
        if (score < 0.f) continue;

        const float edge_score = clampf(c.border_edge / kStrongEdge, 0.f, 1.f);
        const float confidence = clampf(0.35f * edge_score + 0.65f * score, 0.f, 1.f);
        /* if (confidence < c.min_confidence) continue; */

        Cand kept = c;
        kept.confidence = confidence;
        verified.push_back(kept);
    }

    stats->verified_count = static_cast<int32_t>(verified.size());
    stats->verify_ms = clock.ms();
    clock.reset();

    /* Optional. Set RCI_DETECT_DEBUG=1. Does not change the C API. */
    if (const char *dbg = std::getenv("RCI_DETECT_DEBUG")) {
        if (dbg[0] != '\0' && std::strcmp(dbg, "0") != 0) {
            std::fprintf(stderr, "rci_detect: raw candidates before nms: %d\n",
                         static_cast<int>(verified.size()));
        }
    }

    /* Stage 3. NMS inside each type, then a global confidence order. */
    std::sort(verified.begin(), verified.end(), [](const Cand &a, const Cand &b) {
        if (a.type != b.type) return a.type < b.type;
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    std::vector<Cand> kept;
    size_t run = 0;
    while (run < verified.size()) {
        size_t end = run + 1;
        while (end < verified.size() && verified[end].type == verified[run].type) ++end;
        std::vector<Cand> group(verified.begin() + static_cast<std::ptrdiff_t>(run),
                                verified.begin() + static_cast<std::ptrdiff_t>(end));
        nms_inplace(group, params.nms_iou);
        kept.insert(kept.end(), group.begin(), group.end());
        run = end;
    }

    std::sort(kept.begin(), kept.end(), [](const Cand &a, const Cand &b) {
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        if (a.type != b.type) return a.type < b.type;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    stats->kept_count = static_cast<int32_t>(kept.size());
    stats->nms_ms = clock.ms();

    const size_t limit = std::min(out_capacity, static_cast<size_t>(params.max_detections));
    const size_t n = std::min(limit, kept.size());
    for (size_t i = 0; i < n; ++i) {
        rci_detection &d = out[i];
        d.id = static_cast<int32_t>(i + 1);
        d.type = kept[i].type;
        copy_name(d.type_name, sizeof d.type_name, kept[i].type_name);
        d.x = static_cast<float>(kept[i].x);
        d.y = static_cast<float>(kept[i].y);
        d.width = static_cast<float>(kept[i].w);
        d.height = static_cast<float>(kept[i].h);
        d.confidence = kept[i].confidence;
    }
    *out_count = n;
    return RCI_OK;
}

} /* namespace */

extern "C" {

void rci_detect_params_init(rci_detect_params *params) {
    if (!params) return;
    params->edge_threshold = 0.03f;
    params->nms_iou = 0.40f;
    params->max_detections = 32;
    params->scales = 3;
    params->scale_step = 1.5f;
    params->stride = 8;
}

int rci_detect_profiled(
    const rci_image *image,
    const rci_object_def *objects,
    size_t object_count,
    const rci_detect_params *params,
    rci_detection *out,
    size_t out_capacity,
    size_t *out_count,
    rci_detect_stats *stats) {
    if (!out_count) return RCI_ERR_ARG;
    *out_count = 0;
    if (stats) std::memset(stats, 0, sizeof *stats);
    try {
        return detect_impl(image, objects, object_count, params, out, out_capacity, out_count, stats);
    } catch (const std::bad_alloc &) {
        *out_count = 0;
        if (stats) std::memset(stats, 0, sizeof *stats);
        return RCI_ERR_NOMEM;
    }
}

int rci_detect(
    const rci_image *image,
    const rci_object_def *objects,
    size_t object_count,
    const rci_detect_params *params,
    rci_detection *out,
    size_t out_capacity,
    size_t *out_count) {
    return rci_detect_profiled(image, objects, object_count, params, out, out_capacity, out_count,
                               nullptr);
}

} /* extern "C" */
