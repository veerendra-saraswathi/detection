/*
 * rci_detect — C API for the RCI automatic object detection cascade.
 *
 * Stub pipeline, CPU only, no third-party libraries:
 *   1. Multi-scale windows gated by border edge density on all four sides.
 *   2. One class-agnostic verifier: stand-off from the mean, interior
 *      edges quieter than the border, and consistent quadrant contrast.
 *   3. Confidence score and per-class non-maximum suppression.
 *
 * The caller owns the image buffer and the detection buffer.
 * Nothing in this library opens a socket, reads a file, or loads a model.
 *
 * Build (from the repo root):
 *   cmake -S . -B build && cmake --build build
 *   ./build/rci_detect_demo
 */

#ifndef RCI_DETECT_H
#define RCI_DETECT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RCI_DETECT_VERSION "0.1.0"

/* Gray is one byte per pixel. RGB is three bytes, R then G then B. */
typedef enum rci_pixel_format {
    RCI_PIXFMT_GRAY8 = 0,
    RCI_PIXFMT_RGB8 = 1
} rci_pixel_format;

enum {
    RCI_OK = 0,
    RCI_ERR_ARG = -1,     /* null pointer or inconsistent arguments */
    RCI_ERR_FORMAT = -2,  /* pixel format is not gray8 or rgb8 */
    RCI_ERR_SIZE = -3,    /* width, height, or pixel count is out of range */
    RCI_ERR_NOMEM = -4    /* scratch allocation failed */
};

/*
 * One class the caller wants searched.
 * min_width / min_height are the first window on that class's scale pyramid.
 * max_width / max_height cap the pyramid; zero means "no explicit cap".
 * min_confidence drops detections below this score. Negative becomes zero.
 * name is not copied until a detection is emitted, and it may be null.
 */
typedef struct rci_object_def {
    int32_t id;
    const char *name;
    int32_t min_width;
    int32_t min_height;
    int32_t max_width;
    int32_t max_height;
    float min_confidence;
} rci_object_def;

/* One detection. x, y, width, height are pixels, top-left origin. */
typedef struct rci_detection {
    int32_t id;             /* 1-based id within this result */
    int32_t type;           /* rci_object_def.id that matched */
    char type_name[64];     /* copy of the object name, always terminated */
    float x;
    float y;
    float width;
    float height;
    float confidence;       /* 0..1 */
} rci_detection;

/*
 * Image view. data is row-major and is not retained after rci_detect returns.
 * stride is bytes per row. Zero means tightly packed (width * bytes per pixel).
 * The caller guarantees the buffer holds stride * height bytes.
 */
typedef struct rci_image {
    const uint8_t *data;
    int32_t width;
    int32_t height;
    int32_t stride;
    int32_t format;         /* rci_pixel_format */
} rci_image;

/*
 * Cascade knobs. rci_detect_params_init fills in defaults that stay
 * interactive on a CPU-only Jetson (stride 8, a short scale pyramid).
 * A scale_step of 1.01 or less is replaced with 1.5 so the pyramid
 * cannot repeat the same window.
 */
typedef struct rci_detect_params {
    float edge_threshold;   /* ring |dx|+|dy| / 255 required of a candidate */
    float nms_iou;          /* drop the weaker box at or above this IoU */
    int32_t max_detections; /* cap after NMS, before the output buffer cap */
    int32_t scales;         /* windows per object, including the base size */
    float scale_step;       /* size multiplier between scales, must be > 1 */
    int32_t stride;         /* sliding-window step in pixels */
} rci_detect_params;

/* Write the default parameters. params may be null. */
void rci_detect_params_init(rci_detect_params *params);

/*
 * Run the cascade.
 * objects may be null only when object_count is 0 (result is empty).
 * params may be null to use the defaults.
 * out may be null only when out_capacity is 0.
 * On success, *out_count is the number of records written.
 * On failure, *out_count is 0 when out_count itself is non-null.
 */
int rci_detect(
    const rci_image *image,
    const rci_object_def *objects,
    size_t object_count,
    const rci_detect_params *params,
    rci_detection *out,
    size_t out_capacity,
    size_t *out_count);

/*
 * Counts and monotonic-clock times for one rci_detect_profiled call.
 * propose_ms includes the gray conversion and the integral images.
 * kept_count is the number of boxes after NMS, before the output cap.
 * The caller owns this struct.
 */
typedef struct rci_detect_stats {
    int32_t candidate_count;
    int32_t verified_count;
    int32_t kept_count;
    double propose_ms;
    double verify_ms;
    double nms_ms;
} rci_detect_stats;

/*
 * Same contract as rci_detect. stats may be null.
 * When stats is non-null it is filled on success and zeroed on failure.
 */
int rci_detect_profiled(
    const rci_image *image,
    const rci_object_def *objects,
    size_t object_count,
    const rci_detect_params *params,
    rci_detection *out,
    size_t out_capacity,
    size_t *out_count,
    rci_detect_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* RCI_DETECT_H */
