#!/usr/bin/env python3
"""Finalize a rice cutout as a layered PSD plus pure black PNG.

The preferred input is a layered PSD that contains a transparent cutout layer.
If no PSD is available, the script falls back to rembg when installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for dark-backdrop refinement
    cv2 = None

try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    from psd_tools.constants import Compression
except Exception:  # pragma: no cover - optional dependency
    PSDImage = None
    PixelLayer = None
    Compression = None


def iter_leaf_layers(container) -> Iterable:
    for layer in container:
        if getattr(layer, "is_group", lambda: False)():
            yield from iter_leaf_layers(layer)
        else:
            yield layer


def full_canvas_from_layer(psd, layer) -> Image.Image:
    rendered = layer.composite().convert("RGBA")
    x1, y1, _, _ = layer.bbox
    canvas = Image.new("RGBA", psd.size, (0, 0, 0, 0))
    canvas.alpha_composite(rendered, (x1, y1))
    return canvas


def load_cutout_from_psd(psd_path: Path) -> tuple[Image.Image, str]:
    if PSDImage is None:
        raise RuntimeError("psd-tools is not installed")

    psd = PSDImage.open(psd_path)
    candidates: list[tuple[int, int, str, object]] = []
    full_area = psd.size[0] * psd.size[1]

    for layer in iter_leaf_layers(psd):
        if not getattr(layer, "visible", True):
            continue
        name = getattr(layer, "name", "") or ""
        lower = name.lower()
        bbox = layer.bbox
        layer_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        name_score = 20 if ("cutout" in lower or "transparent" in lower or "alpha" in lower) else 0
        size_score = 10 if layer_area < full_area * 0.8 else 0
        try:
            rendered = layer.composite().convert("RGBA")
        except Exception:
            continue
        alpha_count = int((np.array(rendered)[:, :, 3] > 0).sum())
        if alpha_count == 0:
            continue
        candidates.append((name_score + size_score, alpha_count, name, layer))

    if not candidates:
        raise RuntimeError(f"No usable transparent layer found in {psd_path}")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, name, layer = candidates[0]
    return full_canvas_from_layer(psd, layer), name


def load_cutout_with_rembg(image_path: Path) -> tuple[Image.Image, str]:
    try:
        from rembg import new_session, remove
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("No PSD cutout was found and rembg is not installed") from exc
    original = Image.open(image_path).convert("RGB")
    # CoreML/ANE model compilation can stall for several minutes on some macOS
    # versions. CPU is deterministic here and faster for this one-image workflow.
    session = new_session("u2net", providers=["CPUExecutionProvider"])
    return remove(original, session=session).convert("RGBA"), "rembg-u2net-cpu"


def largest_connected_component(mask: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for dark-backdrop rice refinement")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates: list[tuple[int, int]] = []
    for index in range(1, count):
        component = labels == index
        if anchor is not None and not np.any(component & anchor):
            continue
        candidates.append((int(stats[index, cv2.CC_STAT_AREA]), index))
    if not candidates:
        return np.zeros_like(mask, dtype=bool)
    selected = max(candidates)[1]
    return labels == selected


def alpha_from_binary_mask(mask: np.ndarray, sigma: float = 0.55) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for dark-backdrop rice refinement")
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    alpha[alpha < 5] = 0
    alpha[alpha > 250] = 255
    return alpha


def fill_small_mask_holes(mask: np.ndarray, max_area: int) -> tuple[np.ndarray, int, int]:
    """Fill only small enclosed holes without sealing real gaps between leaves."""
    inverse = ~mask.astype(bool)
    labels, count = ndi.label(inverse)
    if count == 0:
        return mask.astype(bool), 0, 0

    border_labels = np.unique(
        np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
        )
    )
    areas = np.bincount(labels.ravel())
    fill_labels = np.where((areas <= max_area) & (np.arange(len(areas)) != 0))[0]
    fill_labels = np.setdiff1d(fill_labels, border_labels, assume_unique=False)
    if not len(fill_labels):
        return mask.astype(bool), 0, 0

    holes = np.isin(labels, fill_labels)
    return mask.astype(bool) | holes, int(holes.sum()), int(len(fill_labels))


def decontaminate_dark_boundary(rgba: Image.Image, core_alpha: int = 180) -> Image.Image:
    """Remove blue/black cloth colour from a narrow foreground boundary."""
    if cv2 is None:
        return rgba.convert("RGBA")
    full = np.array(rgba.convert("RGBA"))
    alpha = full[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if not len(xs):
        return rgba.convert("RGBA")

    x1, x2 = max(0, int(xs.min()) - 8), min(full.shape[1], int(xs.max()) + 9)
    y1, y2 = max(0, int(ys.min()) - 8), min(full.shape[0], int(ys.max()) + 9)
    array = full[y1:y2, x1:x2].copy()
    crop_alpha = array[:, :, 3]

    edge = (crop_alpha > 0) & (crop_alpha < 255)
    core = crop_alpha >= core_alpha
    if edge.any() and core.any():
        _, nearest = ndi.distance_transform_edt(~core, return_indices=True)
        nearest_rgb = array[:, :, :3][nearest[0], nearest[1]]
        array[:, :, :3][edge] = nearest_rgb[edge]

        hard = crop_alpha >= 128
        inward_distance = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
        inner_core = hard & (inward_distance >= 3.2)
        if inner_core.any():
            _, inner_nearest = ndi.distance_transform_edt(~inner_core, return_indices=True)
            inner_rgb = array[:, :, :3][inner_nearest[0], inner_nearest[1]]
            boundary = hard & (inward_distance < 2.6)
            current_value = array[:, :, :3].max(axis=2).astype(np.float32)
            nearest_value = inner_rgb.max(axis=2).astype(np.float32)
            blueish = (array[:, :, 2] > array[:, :, 0] + 3) & (
                array[:, :, 2] > array[:, :, 1] + 3
            )
            much_darker = (current_value < 92) & (current_value < nearest_value * 0.62)
            replace = boundary & (blueish | much_darker)
            array[:, :, :3][replace] = inner_rgb[replace]

    array[:, :, :3][crop_alpha == 0] = 0
    full[y1:y2, x1:x2] = array
    full[:, :, :3][alpha == 0] = 0
    return Image.fromarray(full).convert("RGBA")


def refine_dark_backdrop_rice(
    original: Image.Image,
    coarse_rgba: Image.Image,
    value_floor: int = 90,
) -> tuple[Image.Image, dict]:
    """Use a coarse model only for location, then rebuild rice/pot alpha from colour.

    This targets a frequent failure mode in dark-cloth phenotype photos: a soft
    saliency mask includes a large cloth halo and fills the gaps between leaves.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")

    rgb_full = np.array(original.convert("RGB"))
    coarse_alpha_full = np.array(coarse_rgba.convert("RGBA").getchannel("A"))
    coarse_hard = largest_connected_component(coarse_alpha_full >= 16)
    ys, xs = np.where(coarse_hard)
    if not len(xs):
        raise RuntimeError("coarse model did not locate a subject")

    height, width = coarse_alpha_full.shape
    scale = max(0.35, max(height, width) / 6240.0)
    margin = max(18, int(round(55 * scale)))
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(width, int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(height, int(ys.max()) + margin + 1)
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < 40 or crop_height < 80:
        raise RuntimeError("coarse subject box is too small for rice refinement")
    if crop_width > width * 0.65 or crop_height > height * 0.82:
        raise RuntimeError("coarse subject box is too broad for isolated potted-rice refinement")

    rgb = rgb_full[y1:y2, x1:x2]
    coarse_alpha = coarse_alpha_full[y1:y2, x1:x2]
    crop_h, crop_w = coarse_alpha.shape
    y_grid = np.arange(crop_h)[:, None]
    x_grid = np.arange(crop_w)[None, :]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = [rgb[:, :, index].astype(np.int16) for index in range(3)]
    excess_green = 2 * green - red - blue

    orange = (
        (hue <= 25)
        & (sat >= 45)
        & (val >= 40)
        & (red >= green + 10)
        & (red >= blue + 20)
        & (y_grid > int(crop_h * 0.56))
    )
    orange_row_counts = orange.sum(axis=1)
    orange_peak = int(orange_row_counts.max())
    wide_threshold = max(8, int(crop_w * 0.035), int(orange_peak * 0.45))
    wide_orange_rows = np.where(orange_row_counts >= wide_threshold)[0]
    if not len(wide_orange_rows):
        raise RuntimeError("no wide orange pot rim was found")
    pot_top = int(wide_orange_rows.min())
    orange &= y_grid >= pot_top

    close_size = max(3, int(round(9 * scale)))
    if close_size % 2 == 0:
        close_size += 1
    orange = cv2.morphologyEx(
        orange.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        iterations=2,
    ).astype(bool)
    pot_seed = largest_connected_component(orange)
    pot_ys, pot_xs = np.where(pot_seed)
    if not len(pot_xs):
        raise RuntimeError("no connected orange pot component was found")

    pot_margin_x = max(10, int(round(28 * scale)))
    pot_margin_y = max(8, int(round(18 * scale)))
    pot_window = (
        (x_grid >= int(pot_xs.min()) - pot_margin_x)
        & (x_grid <= int(pot_xs.max()) + pot_margin_x)
        & (y_grid >= pot_top)
        & (y_grid <= int(pot_ys.max()) + pot_margin_y)
    )
    pot = pot_window & (coarse_alpha >= 128)
    pot = cv2.morphologyEx(
        pot.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    pot = largest_connected_component(pot, orange)
    if not pot.any():
        raise RuntimeError("coarse model did not supply a continuous pot silhouette")
    pot = ndi.binary_fill_holes(pot)

    green_yellow = (
        (hue >= 14)
        & (hue <= 92)
        & (sat >= 18)
        & (val >= 38)
        & (green >= blue + 3)
        & (red >= blue - 12)
    )
    pale_sheath = (
        (hue >= 10)
        & (hue <= 58)
        & (sat >= 9)
        & (val >= 64)
        & (red >= blue + 3)
        & (green >= blue + 3)
    )
    brown_sheath = (
        (hue <= 35)
        & (sat >= 18)
        & (val >= 32)
        & (red >= blue + 5)
        & (green >= blue - 5)
    )
    plant_band = y_grid < pot_top + max(5, int(round(12 * scale)))
    # On a dark cloth backdrop, low-value green/cyan cloth folds can mimic leaf
    # hue.  A conservative value floor removes those inter-leaf strips before
    # connectivity grouping; the later alpha blur restores natural 1-2 px edges.
    dark_backdrop_value_floor = int(np.clip(value_floor, 0, 255))
    leaf_candidate = (
        plant_band
        & (val >= dark_backdrop_value_floor)
        & (green_yellow | pale_sheath | brown_sheath)
    )

    group_size = max(3, int(round(7 * scale)))
    if group_size % 2 == 0:
        group_size += 1
    grouped = cv2.dilate(
        leaf_candidate.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_size, group_size)),
        iterations=1,
    ).astype(bool)
    anchor_height = max(55, int(round(180 * scale)))
    anchor = (
        (x_grid >= int(np.percentile(pot_xs, 15)))
        & (x_grid <= int(np.percentile(pot_xs, 85)))
        & (y_grid >= pot_top - anchor_height)
        & (y_grid <= pot_top + max(5, int(round(15 * scale))))
    )
    plant_group = largest_connected_component(grouped, anchor)
    plant = leaf_candidate & plant_group
    if not plant.any():
        raise RuntimeError("colour refinement did not retain the rice canopy")

    rescue_size = max(5, int(round(13 * scale)))
    if rescue_size % 2 == 0:
        rescue_size += 1
    near_plant = cv2.dilate(
        plant.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rescue_size, rescue_size)),
        iterations=1,
    ).astype(bool)
    plant |= leaf_candidate & near_plant

    # Dark blue/black cloth can occasionally satisfy the broad green candidate
    # range in dim areas between overlapping leaves.  Removing those pixels from
    # alpha would nick thin leaves, so keep the silhouette and replace only their
    # colour from the nearest unambiguous plant pixel.  This makes white-background
    # QA expose no cloth-coloured speckles while preserving leaf and stem geometry.
    background_like = plant & (val < 45) & (
        (blue >= red + 3) | (blue >= green + 3)
    )
    cleaned_rgb = rgb.copy()
    clean_plant_reference = plant & ~background_like
    decontaminated_background_pixels = 0
    if background_like.any() and clean_plant_reference.any():
        _, nearest_clean = ndi.distance_transform_edt(
            ~clean_plant_reference, return_indices=True
        )
        nearest_clean_rgb = rgb[nearest_clean[0], nearest_clean[1]]
        cleaned_rgb[background_like] = nearest_clean_rgb[background_like]
        decontaminated_background_pixels = int(background_like.sum())

    mask = plant | pot
    alpha_crop = alpha_from_binary_mask(mask, sigma=max(0.42, min(0.75, 0.55 * scale)))
    alpha_full = np.zeros((height, width), dtype=np.uint8)
    alpha_full[y1:y2, x1:x2] = alpha_crop
    cleaned_rgb_full = rgb_full.copy()
    cleaned_rgb_full[y1:y2, x1:x2] = cleaned_rgb
    rgba_array = np.dstack([cleaned_rgb_full, alpha_full]).astype(np.uint8)
    rgba_array[alpha_full == 0, :3] = 0
    rgba = decontaminate_dark_boundary(Image.fromarray(rgba_array).convert("RGBA"))

    diagnostics = {
        "applied": True,
        "method": "coarse-localization-plus-original-colour-mask",
        "dark_backdrop_value_floor": dark_backdrop_value_floor,
        "subject_roi": [x1, y1, x2, y2],
        "pot_top_y": int(y1 + pot_top),
        "pot_bbox": [
            int(x1 + pot_xs.min()),
            int(y1 + pot_ys.min()),
            int(x1 + pot_xs.max() + 1),
            int(y1 + pot_ys.max() + 1),
        ],
        "plant_pixels": int(plant.sum()),
        "pot_pixels": int(pot.sum()),
        "foreground_pixels": int((alpha_full > 0).sum()),
        "internal_background_like_pixels": int(background_like.sum()),
        "internal_background_like_ratio": float(background_like.sum() / max(1, plant.sum())),
        "internal_background_like_pixels_decontaminated": decontaminated_background_pixels,
        "internal_background_like_pixels_remaining": int(
            background_like.sum() - decontaminated_background_pixels
        ),
        "internal_background_like_ratio_remaining": float(
            (background_like.sum() - decontaminated_background_pixels)
            / max(1, plant.sum())
        ),
    }
    return rgba, diagnostics


def refine_dark_backdrop_gray_pot_rice(
    original: Image.Image,
    coarse_rgba: Image.Image,
) -> tuple[Image.Image, dict]:
    """Recover thin rice leaves above a gray/black pot and optional white tag.

    A frequent phenotype-photo layout contains long, disconnected leaves on
    black cloth, a low-saturation gray nursery pot, a white sample tag, and a
    pale table at the bottom. Saliency models often retain the pot but lose the
    longest leaves; broad colour masks recover those leaves but also collect
    cloth lint. This branch uses the coarse model only for the pot/tag and for
    locating the plant base, then rebuilds long leaves from filtered chromatic
    components in the original image.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")

    rgb = np.array(original.convert("RGB"))
    coarse_alpha = np.array(coarse_rgba.convert("RGBA").getchannel("A"))
    height, width = coarse_alpha.shape
    y_grid = np.arange(height)[:, None]
    x_grid = np.arange(width)[None, :]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = [rgb[:, :, index].astype(np.int16) for index in range(3)]

    # Rembg sometimes fills the black cloth enclosed by arching leaves.  A
    # pure alpha-width test can then mistake the top of that false polygon for
    # the pot mouth.  Keep the alpha estimate, but corroborate it with the
    # first persistent band of bright, low-saturation pot/soil pixels.
    hard_rows = (coarse_alpha >= 192).sum(axis=1)
    peak = int(hard_rows.max())
    wide_threshold = max(int(width * 0.12), int(peak * 0.48), 220)
    wide_rows = np.where(
        (np.arange(height) > int(height * 0.55))
        & (hard_rows >= wide_threshold)
    )[0]
    if not len(wide_rows):
        raise RuntimeError("no wide high-confidence pot band was found")
    alpha_pot_top = int(wide_rows.min())

    central_x1, central_x2 = int(width * 0.18), int(width * 0.82)
    central_width = central_x2 - central_x1

    def persistent_neutral_onset(value_floor: int) -> int | None:
        neutral_rows = (
            (val[:, central_x1:central_x2] >= value_floor)
            & (sat[:, central_x1:central_x2] <= 80)
        ).sum(axis=1)
        smoothed_rows = ndi.median_filter(neutral_rows, size=81, mode="nearest")
        neutral_threshold = max(180, int(round(central_width * 0.10)))
        candidates = np.where(
            (np.arange(height) > int(height * 0.54))
            & (np.arange(height) < int(height * 0.86))
            & (smoothed_rows >= neutral_threshold)
        )[0]
        return int(candidates.min()) if len(candidates) else None

    neutral_top_80 = persistent_neutral_onset(80)
    neutral_top_90 = persistent_neutral_onset(90)
    neutral_pot_top = (
        neutral_top_90 if neutral_top_90 is not None else neutral_top_80
    )
    pot_top = alpha_pot_top
    if neutral_pot_top is not None:
        pot_top = max(pot_top, neutral_pot_top)

    # Locate the tabletop from the side margins, then estimate the pot-body
    # width from dark coarse-foreground pieces crossing that table.  White
    # labels split the body into two pieces, so combine all substantial central
    # pieces instead of keeping only one connected component.
    side_values = np.concatenate(
        [val[:, : int(width * 0.15)], val[:, int(width * 0.85) :]], axis=1
    )
    side_median = ndi.median_filter(
        np.median(side_values, axis=1), size=41, mode="nearest"
    )
    table_rows = np.where(
        (np.arange(height) > int(height * 0.72)) & (side_median >= 90)
    )[0]
    table_top = int(table_rows.min()) if len(table_rows) else int(height * 0.86)

    # Sample horizontal dark runs just below the table transition.  The pot is
    # often split into left/right runs by a white label, while hanging leaves
    # are too thin to pass the run-length threshold.  Row spans are more
    # stable than whole connected-component boxes, which can be widened by a
    # single long leaf.
    seed_boxes: list[tuple[int, int]] = []
    run_start = max(pot_top, table_top + 15)
    run_stop = min(height, table_top + 125)
    minimum_run = max(24, int(round(width * 0.010)))
    for row in range(run_start, run_stop, 5):
        row_mask = (
            (coarse_alpha[row] >= 64)
            & (val[row] < 190)
            & (sat[row] <= 80)
        )
        row_x = np.where(row_mask)[0]
        if not len(row_x):
            continue
        starts = np.r_[0, np.where(np.diff(row_x) > 1)[0] + 1]
        ends = np.r_[starts[1:] - 1, len(row_x) - 1]
        for start_index, end_index in zip(starts, ends):
            seed_x1 = int(row_x[start_index])
            seed_x2 = int(row_x[end_index]) + 1
            seed_width = seed_x2 - seed_x1
            if (
                seed_width >= minimum_run
                and seed_width <= int(width * 0.48)
                and seed_x1 < int(width * 0.72)
                and seed_x2 > int(width * 0.28)
            ):
                seed_boxes.append((seed_x1, seed_x2))
    if not seed_boxes:
        raise RuntimeError("could not estimate the gray pot body on the tabletop")
    body_x1 = min(box[0] for box in seed_boxes)
    body_x2 = max(box[1] for box in seed_boxes)
    corridor_margin = max(60, int(round(width * 0.03)))
    corridor_x1 = max(0, body_x1 - corridor_margin)
    corridor_x2 = min(width, body_x2 + corridor_margin)
    pot_corridor = (x_grid >= corridor_x1) & (x_grid < corridor_x2)

    pot_core = (
        (coarse_alpha >= 176)
        & (y_grid >= pot_top - 10)
        & pot_corridor
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        pot_core.astype(np.uint8), 8
    )
    pot = np.zeros_like(pot_core, dtype=bool)
    minimum_core_area = max(2000, int(round(width * height * 0.00016)))
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_bottom = int(
            stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT]
        )
        if area >= minimum_core_area and component_bottom >= table_top - 20:
            pot |= labels == index
    if not pot.any():
        raise RuntimeError("coarse model did not provide a gray-pot core in the body corridor")
    pot = ndi.binary_fill_holes(pot)
    near_pot = cv2.dilate(
        pot.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    pot |= (
        near_pot
        & (coarse_alpha >= 112)
        & (y_grid >= pot_top - 12)
        & pot_corridor
    )
    pot = ndi.binary_fill_holes(pot)
    pot_ys, pot_xs = np.where(pot)
    if not len(pot_xs):
        raise RuntimeError("gray-pot mask is empty")
    if int(pot_xs.max() - pot_xs.min() + 1) < int(width * 0.12):
        raise RuntimeError("bottom component is too narrow to be a nursery pot")

    # A coarse model can join the pot to a dark rectangular cloth patch on
    # either side of the rim.  Constraining only by the overall pot corridor is
    # not enough in that case: the false patch remains connected to the real
    # pot and binary hole filling preserves it.  Trace a tighter row-wise pot
    # envelope from pixels that are brighter than the local cloth baseline.
    # The baseline comes from the side margins on the same row, so the test is
    # robust to the strong bottom-of-frame flash gradient in this photo set.
    pot_core_bottom = int(pot_ys.max()) + 1
    geometry_left = np.full(height, np.nan, dtype=np.float32)
    geometry_right = np.full(height, np.nan, dtype=np.float32)
    geometry_start = max(0, pot_top - 10)
    # Only the rim/soil band needs this stricter envelope. Applying a colour
    # envelope to the lower mottled pot body would clip genuine dark plastic,
    # especially where the white label occludes it.
    geometry_stop = min(
        height,
        pot_core_bottom,
        pot_top + max(280, int(round(height * 0.085))),
    )
    geometry_delta = 12.0
    horizontal_close = max(15, int(round(width * 0.007)))
    if horizontal_close % 2 == 0:
        horizontal_close += 1
    minimum_piece = max(12, int(round(width * 0.004)))
    minimum_geometry_span = max(260, int(round(width * 0.085)))
    body_center = (body_x1 + body_x2) / 2.0

    for row in range(geometry_start, geometry_stop):
        row_evidence = (
            (coarse_alpha[row] >= 64)
            & (val[row].astype(np.float32) >= side_median[row] + geometry_delta)
            & (sat[row] <= 120)
        )
        row_evidence[:corridor_x1] = False
        row_evidence[corridor_x2:] = False
        closed_row = cv2.morphologyEx(
            row_evidence.astype(np.uint8)[None, :],
            cv2.MORPH_CLOSE,
            np.ones((1, horizontal_close), dtype=np.uint8),
        )[0].astype(bool)
        row_x = np.where(closed_row)[0]
        if not len(row_x):
            continue
        starts = np.r_[0, np.where(np.diff(row_x) > 1)[0] + 1]
        ends = np.r_[starts[1:] - 1, len(row_x) - 1]
        pieces: list[tuple[int, int]] = []
        for start_index, end_index in zip(starts, ends):
            piece_x1 = int(row_x[start_index])
            piece_x2 = int(row_x[end_index]) + 1
            if piece_x2 - piece_x1 >= minimum_piece:
                pieces.append((piece_x1, piece_x2))
        if not pieces:
            continue

        # Merge nearby texture runs across dark soil or mottled pot pixels,
        # but never bridge the much wider gap to a detached leaf or cloth lint.
        merged: list[tuple[int, int]] = []
        maximum_merge_gap = max(45, int(round(width * 0.022)))
        for piece_x1, piece_x2 in pieces:
            if merged and piece_x1 - merged[-1][1] <= maximum_merge_gap:
                merged[-1] = (merged[-1][0], piece_x2)
            else:
                merged.append((piece_x1, piece_x2))
        central_runs = [
            run
            for run in merged
            if run[0] <= body_center + width * 0.08
            and run[1] >= body_center - width * 0.08
        ]
        if not central_runs:
            continue
        run_x1, run_x2 = max(
            central_runs,
            key=lambda run: (run[1] - run[0], -abs((run[0] + run[1]) / 2 - body_center)),
        )
        if run_x2 - run_x1 < minimum_geometry_span:
            continue
        geometry_left[row] = float(run_x1)
        geometry_right[row] = float(run_x2)

    valid_geometry_rows = np.where(
        np.isfinite(geometry_left) & np.isfinite(geometry_right)
    )[0]
    if len(valid_geometry_rows) < max(35, int((geometry_stop - geometry_start) * 0.08)):
        raise RuntimeError("gray-pot colour geometry did not provide a stable body envelope")

    target_rows = np.arange(geometry_start, geometry_stop)

    # Pot sides are smooth over this short vertical band.  A free row-by-row
    # envelope can jump inward wherever dark soil separates two bright texture
    # runs, producing artificial horizontal slots.  Summarise rows in vertical
    # bins and fit one robust straight side to each edge instead.
    bin_size = max(16, int(round(height * 0.006)))
    bin_centres: list[float] = []
    bin_left: list[float] = []
    bin_right: list[float] = []
    for bin_start in range(geometry_start, geometry_stop, bin_size):
        bin_end = min(geometry_stop, bin_start + bin_size)
        rows = valid_geometry_rows[
            (valid_geometry_rows >= bin_start) & (valid_geometry_rows < bin_end)
        ]
        if len(rows) < 3:
            continue
        bin_centres.append(float(np.median(rows)))
        bin_left.append(float(np.percentile(geometry_left[rows], 20)))
        bin_right.append(float(np.percentile(geometry_right[rows], 80)))
    if len(bin_centres) < 3:
        raise RuntimeError("gray-pot geometry did not span enough vertical bins")

    def robust_linear_curve(values: list[float]) -> np.ndarray:
        centres = np.asarray(bin_centres, dtype=np.float64)
        observations = np.asarray(values, dtype=np.float64)
        coefficients = np.polyfit(centres, observations, 1)
        residuals = observations - np.polyval(coefficients, centres)
        residual_limit = max(28.0, width * 0.018)
        keep = np.abs(residuals - np.median(residuals)) <= residual_limit
        if int(keep.sum()) >= 3:
            coefficients = np.polyfit(centres[keep], observations[keep], 1)
        return np.polyval(coefficients, target_rows).astype(np.float32)

    left_curve = robust_linear_curve(bin_left)
    right_curve = robust_linear_curve(bin_right)
    if float(np.min(right_curve - left_curve)) < minimum_geometry_span:
        raise RuntimeError("gray-pot fitted body envelope became implausibly narrow")

    # Trace the visible soil/rim top as a per-column curve. A single horizontal
    # pot_top necessarily includes triangular cloth wedges beside the curved or
    # perspective-distorted rim. Isolated bright leaves above the rim are
    # suppressed by a wide percentile filter and are restored later by the
    # independent plant mask.
    top_search_stop = min(geometry_stop, pot_top + max(220, int(round(height * 0.075))))
    top_evidence = (
        (coarse_alpha >= 64)
        & (val.astype(np.float32) >= side_median[:, None] + geometry_delta)
        & (sat <= 120)
        & (y_grid >= geometry_start)
        & (y_grid < top_search_stop)
        & pot_corridor
    )
    top_evidence = cv2.morphologyEx(
        top_evidence.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    ).astype(bool)
    top_samples = np.full(width, np.nan, dtype=np.float32)
    for column in range(corridor_x1, corridor_x2):
        rows = np.where(top_evidence[geometry_start:top_search_stop, column])[0]
        if len(rows) >= 3:
            top_samples[column] = float(geometry_start + rows.min())
    valid_top_columns = np.where(np.isfinite(top_samples))[0]
    valid_top_columns = valid_top_columns[
        (valid_top_columns >= corridor_x1) & (valid_top_columns < corridor_x2)
    ]
    if len(valid_top_columns) < max(80, int((corridor_x2 - corridor_x1) * 0.20)):
        raise RuntimeError("gray-pot rim did not provide a stable top-edge curve")
    top_curve = np.interp(
        np.arange(width),
        valid_top_columns,
        top_samples[valid_top_columns],
        left=float(top_samples[valid_top_columns[0]]),
        right=float(top_samples[valid_top_columns[-1]]),
    )
    top_curve_window = max(41, int(round(width * 0.026)))
    if top_curve_window % 2 == 0:
        top_curve_window += 1
    top_curve = ndi.percentile_filter(
        top_curve, 60, size=top_curve_window, mode="nearest"
    )
    top_curve = ndi.median_filter(
        top_curve, size=top_curve_window, mode="nearest"
    ).astype(np.float32)

    geometry_margin = max(10, int(round(width * 0.005)))
    row_offsets = target_rows - geometry_start
    envelope = np.zeros_like(pot, dtype=bool)
    envelope[target_rows] = (
        (x_grid >= (left_curve[row_offsets] - geometry_margin)[:, None])
        & (x_grid < (right_curve[row_offsets] + geometry_margin)[:, None])
    )
    geometry_band = (y_grid >= geometry_start) & (y_grid < geometry_stop)
    top_envelope = y_grid >= top_curve[None, :]
    pot &= (~geometry_band) | (envelope & top_envelope)

    # Continue the smooth fitted side lines down the pot body with a wider
    # safety margin. This removes table/cloth regions enclosed by a long
    # looping leaf while preserving dark plastic and irregular dirt along the
    # genuine pot edge. The wider lower margin avoids the clipping that a
    # texture-derived row-by-row envelope caused in earlier revisions.
    if pot_core_bottom > geometry_stop:
        row_denominator = max(1, int(target_rows[-1] - target_rows[0]))
        left_slope = float(left_curve[-1] - left_curve[0]) / row_denominator
        right_slope = float(right_curve[-1] - right_curve[0]) / row_denominator
        lower_rows = np.arange(geometry_stop, pot_core_bottom)
        lower_left = left_curve[-1] + left_slope * (lower_rows - target_rows[-1])
        lower_right = right_curve[-1] + right_slope * (lower_rows - target_rows[-1])
        lower_margin = max(55, int(round(width * 0.022)))
        lower_envelope = np.zeros_like(pot, dtype=bool)
        lower_envelope[lower_rows] = (
            (x_grid >= (lower_left - lower_margin)[:, None])
            & (x_grid < (lower_right + lower_margin)[:, None])
        )
        lower_band = (y_grid >= geometry_stop) & (y_grid < pot_core_bottom)
        pot &= (~lower_band) | lower_envelope
    pot = ndi.binary_fill_holes(pot)
    pot_ys, pot_xs = np.where(pot)
    if not len(pot_xs):
        raise RuntimeError("gray-pot geometry envelope removed the pot core")

    # Soil at the open pot mouth is often as dark as the cloth.  Rembg can
    # therefore cut a notch from the mouth into the pot; binary_fill_holes
    # cannot repair it because the notch is connected to the background.  In
    # the upper part of the pot, bridge only between the leftmost and rightmost
    # pixels of the already selected pot component.  The limited vertical
    # range stops this operation before a protruding sample label could widen
    # the pot silhouette.
    mouth_fill = np.zeros_like(pot, dtype=bool)
    mouth_bottom = min(
        int(pot_ys.max()),
        pot_top + max(240, int(round(height * 0.075))),
    )
    minimum_mouth_span = max(220, int(round(width * 0.10)))
    for row in range(max(0, pot_top - 10), mouth_bottom + 1):
        row_x = np.where(pot[row])[0]
        if not len(row_x):
            continue
        left, right = int(row_x.min()), int(row_x.max())
        if right - left + 1 >= minimum_mouth_span:
            mouth_fill[row, left : right + 1] = True
    pot_before_mouth_fill = int(pot.sum())
    pot |= mouth_fill
    pot = ndi.binary_fill_holes(pot)
    pot_mouth_fill_pixels = int(pot.sum()) - pot_before_mouth_fill

    # Reconstruct a cool-white sample paddle from a strict bright core before
    # using the broader recovery rule.  In this photograph series the tabletop
    # is warm beige while the plastic label is neutral/cool white; selecting a
    # single compact core and closing it *after* component selection restores
    # dirt-covered writing without merging a detached pale table fragment.
    strict_label_seed = (
        (coarse_alpha >= 128)
        & (val >= 190)
        & (sat <= 35)
        & ((hue >= 75) | (sat <= 12))
        & (y_grid >= max(pot_top, table_top - 350))
        & pot_corridor
    )
    strict_seed_close = max(7, int(round(width * 0.003)))
    if strict_seed_close % 2 == 0:
        strict_seed_close += 1
    strict_label_seed = cv2.morphologyEx(
        strict_label_seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (strict_seed_close, strict_seed_close)
        ),
        iterations=1,
    ).astype(bool)
    strict_count, strict_labels, strict_stats, _ = cv2.connectedComponentsWithStats(
        strict_label_seed.astype(np.uint8), 8
    )
    strict_candidates: list[int] = []
    for index in range(1, strict_count):
        label_area = int(strict_stats[index, cv2.CC_STAT_AREA])
        label_width = int(strict_stats[index, cv2.CC_STAT_WIDTH])
        label_height = int(strict_stats[index, cv2.CC_STAT_HEIGHT])
        component = strict_labels == index
        if (
            label_area >= max(1200, int(round(width * height * 0.00008)))
            and label_width <= int(width * 0.42)
            and label_height <= int(height * 0.32)
            and np.any(component & pot)
        ):
            strict_candidates.append(index)

    strict_label_shape = np.zeros_like(pot, dtype=bool)
    if strict_candidates:
        strict_index = max(
            strict_candidates,
            key=lambda index: int(strict_stats[index, cv2.CC_STAT_AREA]),
        )
        strict_label_shape = strict_labels == strict_index
        strict_shape_close = max(15, int(round(width * 0.013)))
        if strict_shape_close % 2 == 0:
            strict_shape_close += 1
        strict_label_shape = cv2.morphologyEx(
            strict_label_shape.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (strict_shape_close, strict_shape_close),
            ),
            iterations=1,
        ).astype(bool)

    if strict_label_shape.any():
        label_mask = strict_label_shape.copy()
    else:
        # Fallback for warm/underexposed labels: recover a broader pale
        # component, but only next to the already trusted pot foreground.
        label_candidate = (
            (coarse_alpha >= 176)
            & (val >= 135)
            & (sat <= 100)
            & (y_grid >= max(pot_top, table_top - 350))
        )
        label_count, label_labels, label_stats, _ = cv2.connectedComponentsWithStats(
            label_candidate.astype(np.uint8), 8
        )
        label_mask = np.zeros_like(pot, dtype=bool)
        for index in range(1, label_count):
            label_area = int(label_stats[index, cv2.CC_STAT_AREA])
            label_width = int(label_stats[index, cv2.CC_STAT_WIDTH])
            label_height = int(label_stats[index, cv2.CC_STAT_HEIGHT])
            label_fill = label_area / max(1, label_width * label_height)
            component = label_labels == index
            if (
                label_area >= 500
                and label_width <= int(width * 0.55)
                and label_height <= int(height * 0.50)
                and label_fill >= 0.08
                and np.any(component & pot_corridor)
            ):
                label_mask |= component
        label_recovery_radius = max(28, int(round(width * 0.012)))
        label_recovery_kernel_size = label_recovery_radius * 2 + 1
        label_recovery_neighbourhood = cv2.dilate(
            pot.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (label_recovery_kernel_size, label_recovery_kernel_size),
            ),
            iterations=1,
        ).astype(bool)
        label_mask &= label_recovery_neighbourhood
    label_recovered_pixels = int((label_mask & ~pot).sum())
    pot |= label_mask
    pot_anchor = (
        (x_grid >= corridor_x1)
        & (x_grid < corridor_x2)
        & (y_grid >= pot_top)
        & (y_grid <= min(height - 1, table_top + 120))
    )
    pot = largest_connected_component(pot, pot_anchor)

    # A looping leaf can enclose the warm tabletop beneath a long white label,
    # making that background one connected coarse foreground component.  When
    # a reliable strict label was found, taper the fitted pot sides towards the
    # bottom, stop the body where its broad continuous silhouette ends, and
    # remove anything hanging below the label's per-column visible edge.  The
    # clean label shape is then restored explicitly.
    lower_body_bottom: int | None = None
    lower_cleanup_start: int | None = None
    if strict_label_shape.any() and pot_core_bottom > geometry_stop:
        tapered_rows = np.arange(geometry_stop, pot_core_bottom)
        tapered_left = left_curve[-1] + left_slope * (
            tapered_rows - target_rows[-1]
        )
        tapered_right = right_curve[-1] + right_slope * (
            tapered_rows - target_rows[-1]
        )
        taper_progress = np.clip(
            (tapered_rows - target_rows[-1])
            / max(1, pot_core_bottom - 1 - target_rows[-1]),
            0.0,
            1.0,
        )
        bottom_margin = max(6, int(round(width * 0.003)))
        tapered_margin = (
            lower_margin * (1.0 - taper_progress)
            + bottom_margin * taper_progress
        )
        tapered_envelope = np.zeros_like(pot, dtype=bool)
        tapered_envelope[tapered_rows] = (
            (x_grid >= (tapered_left - tapered_margin)[:, None])
            & (x_grid < (tapered_right + tapered_margin)[:, None])
        )

        body_scan_start = max(geometry_stop, table_top - 40)
        reference_body_width = float(np.median(right_curve - left_curve))
        minimum_continuous_body = max(220, int(round(reference_body_width * 0.55)))
        qualifying_rows = np.zeros(pot_core_bottom - body_scan_start, dtype=np.uint8)
        for row in range(body_scan_start, pot_core_bottom):
            row_x = np.where(pot[row] & tapered_envelope[row])[0]
            if not len(row_x):
                continue
            starts = np.r_[0, np.where(np.diff(row_x) > 1)[0] + 1]
            ends = np.r_[starts[1:] - 1, len(row_x) - 1]
            longest_run = max(
                (int(row_x[end] - row_x[start] + 1) for start, end in zip(starts, ends)),
                default=0,
            )
            qualifying_rows[row - body_scan_start] = int(
                longest_run >= minimum_continuous_body
            )
        persistent_rows = ndi.median_filter(
            qualifying_rows, size=17, mode="nearest"
        ) > 0
        persistent_indices = np.where(persistent_rows)[0]
        if len(persistent_indices):
            lower_body_bottom = body_scan_start + int(persistent_indices.max()) + 1
            # A bright cloth fold can make the side-margin tabletop detector
            # fire above the pot geometry band.  Never apply the tapered lower
            # envelope before geometry_stop, otherwise the still-zero envelope
            # would erase the entire soil/rim band as a horizontal slot.
            lower_cleanup_start = max(table_top, geometry_stop)
            lower_cleanup_band = y_grid >= lower_cleanup_start
            cleaned_lower_body = (
                pot
                & tapered_envelope
                & (y_grid < lower_body_bottom)
            )
            label_columns = np.where(strict_label_shape.any(axis=0))[0]
            for column in label_columns:
                label_rows = np.where(strict_label_shape[:, column])[0]
                if len(label_rows):
                    cleaned_lower_body[int(label_rows.max()) + 1 :, column] = False
            pot = (
                (pot & ~lower_cleanup_band)
                | (cleaned_lower_body & lower_cleanup_band)
                | strict_label_shape
            )
    pot_ys, pot_xs = np.where(pot)

    neutral_pot_fraction = float((sat[pot] <= 70).mean())
    if neutral_pot_fraction < 0.65:
        raise RuntimeError(
            "bottom component is not predominantly gray/black; use the orange-pot workflow"
        )

    green_leaf = (
        (hue >= 16)
        & (hue <= 100)
        & (sat >= 28)
        & (val >= 72)
        & (green >= blue + 6)
        & (green >= red - 24)
    )
    tan_sheath = (
        (hue <= 42)
        & (sat >= 24)
        & (val >= 78)
        & (red >= blue + 6)
        & (green >= blue - 2)
    )
    pale_sheath = (
        (val >= 105)
        & (red >= blue + 4)
        & (green >= blue + 2)
        & ((red - blue) >= 10)
        & (coarse_alpha >= 32)
    )
    bright_plant = (
        (val >= 115)
        & (red >= blue + 2)
        & (green >= blue)
        & ((np.maximum(red, green) - blue) >= 5)
    )
    neutral_highlight = (val >= 145) & (sat <= 80)
    biological_colour = green_leaf | tan_sheath | pale_sheath | bright_plant
    upper_plant_band = y_grid < pot_top + 35
    hanging_leaf_band = (
        (y_grid >= pot_top + 35)
        & (y_grid < max(pot_top + 35, table_top - 15))
    )
    candidate = (
        upper_plant_band & (biological_colour | neutral_highlight)
    ) | (
        hanging_leaf_band
        & biological_colour
        & (coarse_alpha >= 64)
    )

    # Median filtering removes single cloth fibres. Component shape filtering
    # keeps long leaf pieces while rejecting round dust specks that happen to
    # share a yellow/green colour.
    smoothed = cv2.medianBlur((candidate * 255).astype(np.uint8), 3) > 127
    smoothed = cv2.morphologyEx(
        smoothed.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    # Morphological dilation can otherwise grow the mask back into adjacent
    # dark cloth.  Intersecting with the original colour evidence preserves
    # the true leaf edge while still benefiting from the noise removal above.
    smoothed &= candidate
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        smoothed.astype(np.uint8), 8
    )
    filtered = np.zeros_like(smoothed, dtype=bool)
    kept_component_count = 0
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        long_axis = max(component_width, component_height)
        short_axis = max(1, min(component_width, component_height))
        aspect = long_axis / short_axis
        component = labels == index
        coarse_overlap = float((coarse_alpha[component] >= 160).mean())
        fill_ratio = area / max(1, component_width * component_height)
        biological_fraction = float(biological_colour[component].mean())
        component_median_saturation = float(np.median(sat[component]))
        near_horizontal_neutral_line = (
            component_width >= 140
            and component_width >= component_height * 8
            and (
                biological_fraction < 0.35
                or component_median_saturation < 90
            )
        )
        keep = (
            (area >= 60 and long_axis >= 20 and aspect >= 2.0)
            or (area >= 24 and long_axis >= 9 and coarse_overlap >= 0.55)
            or (
                area >= 120
                and coarse_overlap >= 0.05
                and (aspect >= 1.35 or fill_ratio <= 0.30)
            )
        ) and not near_horizontal_neutral_line
        if keep:
            filtered[component] = True
            kept_component_count += 1
    if not filtered.any():
        raise RuntimeError("colour reconstruction retained no elongated rice components")

    grouped = cv2.dilate(
        filtered.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    anchor = (
        (y_grid >= pot_top - 380)
        & (y_grid <= pot_top + 30)
        & (x_grid >= int(pot_xs.min()) - 160)
        & (x_grid <= int(pot_xs.max()) + 160)
    )
    group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(
        grouped.astype(np.uint8), 8
    )
    anchored_groups = [
        index
        for index in range(1, group_count)
        if np.any((group_labels == index) & anchor)
    ]
    if not anchored_groups:
        raise RuntimeError("elongated rice components did not connect to the pot mouth")
    plant_group_index = max(
        anchored_groups,
        key=lambda index: int(group_stats[index, cv2.CC_STAT_AREA]),
    )
    selected_groups = group_labels == plant_group_index
    main_group = selected_groups.copy()
    distance_to_main_group = cv2.distanceTransform(
        (~main_group).astype(np.uint8), cv2.DIST_L2, 3
    )
    rescue_distance_limit = max(80.0, width * 0.04)
    rescued_nearby_group_count = 0
    rescued_nearby_group_pixels = 0
    for index in range(1, group_count):
        if index == plant_group_index:
            continue
        group = group_labels == index
        core = filtered & group
        core_area = int(core.sum())
        if core_area == 0:
            continue
        component_width = int(group_stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(group_stats[index, cv2.CC_STAT_HEIGHT])
        long_axis = max(component_width, component_height)
        short_axis = max(1, min(component_width, component_height))
        aspect = long_axis / short_axis
        fill_ratio = core_area / max(1, component_width * component_height)
        biological_fraction = float(biological_colour[core].mean())
        coarse_overlap = float((coarse_alpha[core] >= 128).mean())
        minimum_distance = float(distance_to_main_group[group].min())
        leaf_like_shape = aspect >= 2.0 or fill_ratio <= 0.12
        nearby_leaf = (
            core_area >= 80
            and long_axis >= 45
            and leaf_like_shape
            and biological_fraction >= 0.65
            and coarse_overlap >= 0.05
            and minimum_distance <= rescue_distance_limit
        )
        strongly_supported_detached_leaf = (
            core_area >= 250
            and long_axis >= 180
            and (aspect >= 3.0 or fill_ratio <= 0.07)
            and biological_fraction >= 0.75
            and coarse_overlap >= 0.25
        )
        if nearby_leaf or strongly_supported_detached_leaf:
            selected_groups |= group
            rescued_nearby_group_count += 1
            rescued_nearby_group_pixels += core_area
    plant = filtered & selected_groups

    # Restore only tiny enclosed pinholes in leaf blades.  Large enclosed
    # regions are genuine gaps between arching leaves and remain transparent.
    plant_filled = ndi.binary_fill_holes(plant)
    small_holes = plant_filled & ~plant
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        small_holes.astype(np.uint8), 8
    )
    filled_leaf_hole_pixels = 0
    for index in range(1, hole_count):
        hole_area = int(hole_stats[index, cv2.CC_STAT_AREA])
        if hole_area <= 120:
            hole = hole_labels == index
            plant[hole] = True
            filled_leaf_hole_pixels += hole_area

    # Remove one pixel of cloth-contaminated fringe from thin leaves. The pot
    # mask overlaps the plant mask around pot_top, preventing a horizontal seam.
    plant = cv2.erode(
        plant.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1,
    ).astype(bool)
    if not plant.any():
        raise RuntimeError("gray-pot leaf cleanup removed the rice canopy")

    mask = plant | pot
    alpha = alpha_from_binary_mask(mask, sigma=0.60)
    rgba_array = np.dstack([rgb, alpha]).astype(np.uint8)
    rgba_array[alpha == 0, :3] = 0
    rgba = decontaminate_dark_boundary(Image.fromarray(rgba_array).convert("RGBA"))

    diagnostics = {
        "applied": True,
        "method": "gray-pot-hard-core-plus-elongated-original-colour-leaves",
        "pot_style": "gray_or_black_with_optional_white_tag",
        "subject_roi": [
            int(np.where(alpha > 0)[1].min()),
            int(np.where(alpha > 0)[0].min()),
            int(np.where(alpha > 0)[1].max() + 1),
            int(np.where(alpha > 0)[0].max() + 1),
        ],
        "pot_top_y": pot_top,
        "alpha_pot_top_y": alpha_pot_top,
        "neutral_pot_top_y": neutral_pot_top,
        "table_top_y": table_top,
        "pot_corridor": [corridor_x1, corridor_x2],
        "pot_geometry_y_range": [geometry_start, geometry_stop],
        "pot_geometry_left_range": [
            float(left_curve.min()),
            float(left_curve.max()),
        ],
        "pot_geometry_right_range": [
            float(right_curve.min()),
            float(right_curve.max()),
        ],
        "pot_top_curve_y_range": [
            float(top_curve[corridor_x1:corridor_x2].min()),
            float(top_curve[corridor_x1:corridor_x2].max()),
        ],
        "pot_bbox": [
            int(pot_xs.min()),
            int(pot_ys.min()),
            int(pot_xs.max() + 1),
            int(pot_ys.max() + 1),
        ],
        "neutral_pot_fraction": neutral_pot_fraction,
        "pot_mouth_fill_pixels": pot_mouth_fill_pixels,
        "label_recovered_pixels": label_recovered_pixels,
        "strict_label_pixels": int(strict_label_shape.sum()),
        "lower_cleanup_start_y": lower_cleanup_start,
        "lower_body_bottom_y": lower_body_bottom,
        "kept_colour_component_count": kept_component_count,
        "rescued_nearby_leaf_group_count": rescued_nearby_group_count,
        "rescued_nearby_leaf_pixels": rescued_nearby_group_pixels,
        "filled_leaf_hole_pixels": filled_leaf_hole_pixels,
        "plant_pixels": int(plant.sum()),
        "pot_pixels": int(pot.sum()),
        "foreground_pixels": int((alpha > 0).sum()),
        "table_pixels_retained": int(((alpha > 0) & (y_grid >= pot_ys.max() + 1)).sum()),
    }
    return rgba, diagnostics


def refine_frame_filling_dark_backdrop_rice(
    original: Image.Image,
    coarse_rgba: Image.Image,
) -> tuple[Image.Image, dict]:
    """Rebuild a large potted-rice mask when the subject touches frame edges.

    Rembg is used only as a loose spatial prior. Plant pixels are reconstructed
    from the original RGB/HSV colours, so dark cloth inside the canopy is not
    retained. The pot uses a hard rembg core constrained by a colour-located
    orange rim; this preserves its geometry without carrying the soft halo.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")

    rgb_full = np.array(original.convert("RGB"))
    coarse_alpha_full = np.array(coarse_rgba.convert("RGBA").getchannel("A"))
    coarse_hard = largest_connected_component(coarse_alpha_full >= 16)
    ys, xs = np.where(coarse_hard)
    if not len(xs):
        raise RuntimeError("coarse model did not locate a subject")

    height, width = coarse_alpha_full.shape
    scale = max(0.35, max(height, width) / 6240.0)
    margin = max(18, int(round(48 * scale)))
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(width, int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(height, int(ys.max()) + margin + 1)

    rgb = rgb_full[y1:y2, x1:x2]
    coarse_alpha = coarse_alpha_full[y1:y2, x1:x2]
    crop_h, crop_w = coarse_alpha.shape
    y_grid = np.arange(crop_h)[:, None]
    x_grid = np.arange(crop_w)[None, :]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = [rgb[:, :, index].astype(np.int16) for index in range(3)]
    excess_green = 2 * green - red - blue

    # Find the real pot rim by its wide, continuous orange rows. Narrow brown
    # sheath or insect pixels above the pot cannot satisfy the row-width test.
    orange = (
        (hue <= 28)
        & (sat >= 38)
        & (val >= 36)
        & (red >= green + 9)
        & (red >= blue + 18)
        & (y_grid > int(crop_h * 0.48))
    )
    orange_row_counts = orange.sum(axis=1)
    orange_peak = int(orange_row_counts.max())
    wide_threshold = max(12, int(crop_w * 0.055), int(orange_peak * 0.42))
    wide_orange_rows = np.where(orange_row_counts >= wide_threshold)[0]
    if not len(wide_orange_rows):
        raise RuntimeError("no wide orange pot rim was found in frame-filling mode")
    pot_top = int(wide_orange_rows.min())
    orange &= y_grid >= pot_top

    close_size = max(3, int(round(9 * scale)))
    if close_size % 2 == 0:
        close_size += 1
    orange_grouped = cv2.morphologyEx(
        orange.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        iterations=2,
    ).astype(bool)
    pot_seed = largest_connected_component(orange_grouped)
    pot_ys, pot_xs = np.where(pot_seed)
    if not len(pot_xs):
        raise RuntimeError("no connected orange pot component was found in frame-filling mode")

    # The top edge of a round pot is curved. A single horizontal pot_top value
    # retains triangular strips of black cloth above the rim near its sides.
    # Trace the first orange rim pixel in each column, interpolate occluded
    # columns behind stems, and smooth only this one-dimensional envelope.
    pot_left = int(pot_xs.min())
    pot_right = int(pot_xs.max())
    top_curve = np.full(crop_w, np.nan, dtype=np.float32)
    rim_search_bottom = min(crop_h, pot_top + max(100, int(round(300 * scale))))
    for column in range(pot_left, pot_right + 1):
        rows = np.where(orange[pot_top:rim_search_bottom, column])[0]
        if len(rows):
            top_curve[column] = float(pot_top + rows.min())
    valid_columns = np.where(np.isfinite(top_curve))[0]
    if len(valid_columns) < max(20, int((pot_right - pot_left + 1) * 0.2)):
        raise RuntimeError("orange pot rim did not provide a stable top-edge envelope")
    all_columns = np.arange(crop_w)
    top_curve = np.interp(
        all_columns,
        valid_columns,
        top_curve[valid_columns],
        left=float(top_curve[valid_columns[0]]),
        right=float(top_curve[valid_columns[-1]]),
    )
    curve_filter_size = max(9, int(round(61 * scale)))
    if curve_filter_size % 2 == 0:
        curve_filter_size += 1
    top_curve = ndi.median_filter(top_curve, size=curve_filter_size, mode="nearest")

    pot_margin_x = max(10, int(round(30 * scale)))
    pot_window = (
        (x_grid >= pot_left - pot_margin_x)
        & (x_grid <= pot_right + pot_margin_x)
        & (y_grid >= top_curve[None, :])
    )
    pot = pot_window & ((coarse_alpha >= 192) | orange_grouped)
    pot = cv2.morphologyEx(
        pot.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    pot = largest_connected_component(pot, orange)
    if not pot.any():
        raise RuntimeError("coarse model did not supply a continuous frame-filling pot silhouette")

    # Rebuild the canopy directly from pixels that are chromatically distinct
    # from the blue/neutral black cloth. Brown/tan criteria preserve sheaths,
    # insect bodies and damaged tissue attached to green leaves.
    green_yellow = (
        (hue >= 13)
        & (hue <= 78)
        & (sat >= 16)
        & (val >= 50)
        & (green >= blue + 10)
        & (excess_green >= 16)
        & (red >= blue - 14)
    )
    pale_sheath = (
        (hue >= 8)
        & (hue <= 60)
        & (sat >= 8)
        & (val >= 58)
        & (red >= blue + 2)
        & (green >= blue + 2)
    )
    brown_tan = (
        (hue <= 36)
        & (sat >= 15)
        & (val >= 30)
        & (red >= blue + 4)
        & (green >= blue - 7)
    )
    # Continue plant reconstruction through the deepest visible part of the
    # elliptical pot mouth. Otherwise the top-curve subtraction can punch
    # white notches through stems that occlude the rear rim.
    plant_limit = min(
        crop_h,
        int(np.ceil(top_curve[pot_left : pot_right + 1].max()))
        + max(12, int(round(28 * scale))),
    )
    plant_band = y_grid < plant_limit
    coarse_prior = coarse_alpha >= 4
    leaf_candidate = plant_band & coarse_prior & (green_yellow | pale_sheath | brown_tan)

    group_size = max(5, int(round(11 * scale)))
    if group_size % 2 == 0:
        group_size += 1
    grouped = cv2.dilate(
        leaf_candidate.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_size, group_size)),
        iterations=1,
    ).astype(bool)
    anchor_height = max(80, int(round(260 * scale)))
    anchor = (
        (x_grid >= int(np.percentile(pot_xs, 8)))
        & (x_grid <= int(np.percentile(pot_xs, 92)))
        & (y_grid >= pot_top - anchor_height)
        & (y_grid <= pot_top + max(5, int(round(18 * scale))))
    )
    plant_group = largest_connected_component(grouped, anchor)
    plant = leaf_candidate & plant_group
    if not plant.any():
        raise RuntimeError("frame-filling colour refinement did not retain the rice canopy")

    # Recover tiny neutral insect/lesion pixels that are enclosed by plant
    # colour, while keeping the large cloth-connected gaps transparent.
    close_leaf = max(3, int(round(3 * scale)))
    if close_leaf % 2 == 0:
        close_leaf += 1
    plant = cv2.morphologyEx(
        plant.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_leaf, close_leaf)),
        iterations=1,
    ).astype(bool)
    max_hole_area = max(64, int(round(900 * scale * scale)))
    plant, filled_hole_pixels, filled_hole_count = fill_small_mask_holes(
        plant, max_area=max_hole_area
    )

    # A two-pixel-scale rescue keeps original anti-aliased edge colours, but is
    # intentionally too narrow to refill genuine spaces between leaves.
    rescue_size = 3
    if rescue_size % 2 == 0:
        rescue_size += 1
    near_plant = cv2.dilate(
        plant.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rescue_size, rescue_size)),
        iterations=1,
    ).astype(bool)
    rescue_colour = (
        (val >= 44)
        & ((green >= blue + 7) | (red >= blue + 5))
        & ((excess_green >= 10) | (red >= green - 8))
    )
    rescued = plant_band & near_plant & rescue_colour & (coarse_alpha >= 192)
    plant |= rescued

    mask = plant | pot
    alpha_crop = alpha_from_binary_mask(mask, sigma=max(0.42, min(0.72, 0.52 * scale)))
    alpha_full = np.zeros((height, width), dtype=np.uint8)
    alpha_full[y1:y2, x1:x2] = alpha_crop
    rgba_array = np.dstack([rgb_full, alpha_full]).astype(np.uint8)
    rgba_array[alpha_full == 0, :3] = 0
    rgba = decontaminate_dark_boundary(Image.fromarray(rgba_array).convert("RGBA"))

    cloth_like = plant & (val < 55) & (
        (blue >= red + 2) | ((sat < 18) & (green <= red + 5))
    )
    diagnostics = {
        "applied": True,
        "method": "frame-filling-original-colour-mask-plus-hard-pot-core",
        "subject_roi": [x1, y1, x2, y2],
        "pot_top_y": int(y1 + pot_top),
        "pot_top_curve_y_range": [
            int(y1 + np.floor(top_curve[pot_left : pot_right + 1].min())),
            int(y1 + np.ceil(top_curve[pot_left : pot_right + 1].max())),
        ],
        "pot_bbox": [
            int(x1 + pot_xs.min()),
            int(y1 + pot_ys.min()),
            int(x1 + pot_xs.max() + 1),
            int(y1 + pot_ys.max() + 1),
        ],
        "plant_pixels": int(plant.sum()),
        "pot_pixels": int(pot.sum()),
        "foreground_pixels": int((alpha_full > 0).sum()),
        "filled_small_hole_pixels": filled_hole_pixels,
        "filled_small_hole_count": filled_hole_count,
        "internal_cloth_like_pixels": int(cloth_like.sum()),
        "internal_cloth_like_ratio": float(cloth_like.sum() / max(1, plant.sum())),
    }
    return rgba, diagnostics


def refine_frame_filling_dark_backdrop_white_pot(
    original: Image.Image,
    coarse_rgba: Image.Image,
) -> tuple[Image.Image, dict]:
    """Rebuild close-up rice stems above a white/cream pot on dark cloth.

    This phenotype-photo layout often crops stems at the top and the pot at the
    bottom. Orange-rim detection is invalid here. A wide low-saturation bright
    component locates the white pot, while plant pixels are reconstructed from
    the original colours so cloth gaps between stems remain transparent.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")

    rgb_full = np.array(original.convert("RGB"))
    coarse_alpha = np.array(coarse_rgba.convert("RGBA").getchannel("A"))
    height, width = coarse_alpha.shape
    scale = max(0.35, max(height, width) / 6240.0)
    y_grid = np.arange(height)[:, None]
    x_grid = np.arange(width)[None, :]

    hsv = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = [rgb_full[:, :, index].astype(np.int16) for index in range(3)]
    excess_green = 2 * green - red - blue

    white_strong = (
        (y_grid > int(height * 0.55))
        & (val >= 100)
        & (sat <= 110)
        & (np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue]) <= 105)
    )
    row_counts = white_strong.sum(axis=1)
    peak = int(row_counts.max())
    wide_threshold = max(int(width * 0.25), int(peak * 0.50))
    wide_rows = np.where(row_counts >= wide_threshold)[0]
    if not len(wide_rows):
        raise RuntimeError("no wide white/cream pot rim was found")
    pot_top = int(wide_rows.min())

    close_size = max(5, int(round(15 * scale)))
    if close_size % 2 == 0:
        close_size += 1
    white_grouped = cv2.morphologyEx(
        white_strong.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        iterations=2,
    ).astype(bool)
    pot_seed = largest_connected_component(white_grouped)
    seed_ys, seed_xs = np.where(pot_seed)
    if not len(seed_xs):
        raise RuntimeError("white/cream pot pixels were not connected")
    if int(seed_xs.max() - seed_xs.min() + 1) < int(width * 0.35):
        raise RuntimeError("white/cream component is too narrow to be a pot")

    pot_left = int(seed_xs.min())
    pot_right = int(seed_xs.max())
    top_curve = np.full(width, np.nan, dtype=np.float32)
    rim_search_bottom = min(height, pot_top + max(120, int(round(360 * scale))))
    for column in range(pot_left, pot_right + 1):
        rows = np.where(white_strong[pot_top:rim_search_bottom, column])[0]
        if len(rows):
            top_curve[column] = float(pot_top + rows.min())
    valid_columns = np.where(np.isfinite(top_curve))[0]
    if len(valid_columns) < max(30, int((pot_right - pot_left + 1) * 0.25)):
        raise RuntimeError("white/cream pot rim did not provide a stable top-edge envelope")
    all_columns = np.arange(width)
    top_curve = np.interp(
        all_columns,
        valid_columns,
        top_curve[valid_columns],
        left=float(top_curve[valid_columns[0]]),
        right=float(top_curve[valid_columns[-1]]),
    )
    curve_filter_size = max(11, int(round(71 * scale)))
    if curve_filter_size % 2 == 0:
        curve_filter_size += 1
    top_curve = ndi.median_filter(top_curve, size=curve_filter_size, mode="nearest")

    pot_margin_x = max(10, int(round(24 * scale)))
    pot_window = (
        (x_grid >= pot_left - pot_margin_x)
        & (x_grid <= pot_right + pot_margin_x)
        & (y_grid >= top_curve[None, :])
    )
    white_soft = (
        (val >= 52)
        & (sat <= 155)
        & (np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue]) <= 135)
    )
    pot_candidate = pot_window & (white_soft | (coarse_alpha >= 128))
    pot_candidate = cv2.morphologyEx(
        pot_candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    ).astype(bool)
    pot = largest_connected_component(pot_candidate, white_grouped)
    pot = ndi.binary_fill_holes(pot)
    if not pot.any():
        raise RuntimeError("white/cream pot silhouette could not be reconstructed")

    # Green stems can be strongly defocused in these close-up phenotype
    # photographs.  Requiring the coarse model for every green pixel makes a
    # defocused stem break into salt-and-pepper fragments.  Reconstruct the
    # green/pale stem body from original colour and reserve coarse-alpha gating
    # for the ambiguous dark brown/tan pixels (insects, roots, and scars).
    green_yellow = (
        (hue >= 13)
        & (hue <= 78)
        & (sat >= 16)
        & (val >= 38)
        & (green >= blue + 7)
        & (excess_green >= 11)
        & (red >= blue - 20)
    )
    pale_sheath = (
        (hue >= 8)
        & (hue <= 62)
        & (sat >= 7)
        & (val >= 48)
        & (red >= blue + 1)
        & (green >= blue + 1)
        & ((excess_green >= 6) | (red >= green - 6))
    )
    brown_tan = (
        (hue <= 38)
        & (sat >= 13)
        & (val >= 26)
        & (red >= blue + 3)
        & (green >= blue - 9)
    )
    plant_limit = min(
        height,
        int(np.ceil(top_curve[pot_left : pot_right + 1].max()))
        + max(16, int(round(34 * scale))),
    )
    plant_band = y_grid < plant_limit
    leaf_candidate = plant_band & (
        green_yellow | pale_sheath | (brown_tan & (coarse_alpha >= 4))
    )

    group_size = max(5, int(round(11 * scale)))
    if group_size % 2 == 0:
        group_size += 1
    grouped = cv2.dilate(
        leaf_candidate.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_size, group_size)),
        iterations=1,
    ).astype(bool)
    anchor_height = max(100, int(round(320 * scale)))
    anchor = (
        (x_grid >= pot_left)
        & (x_grid <= pot_right)
        & (y_grid >= pot_top - anchor_height)
        & (y_grid <= pot_top + max(10, int(round(30 * scale))))
    )
    plant_group = largest_connected_component(grouped, anchor)
    plant = leaf_candidate & plant_group
    if not plant.any():
        raise RuntimeError("white-pot colour refinement did not retain the rice stems")

    plant_close_size = max(3, int(round(5 * scale)))
    if plant_close_size % 2 == 0:
        plant_close_size += 1
    plant = cv2.morphologyEx(
        plant.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (plant_close_size, plant_close_size)
        ),
        iterations=1,
    ).astype(bool)
    max_hole_area = max(64, int(round(900 * scale * scale)))
    plant, filled_hole_pixels, filled_hole_count = fill_small_mask_holes(
        plant, max_area=max_hole_area
    )

    near_plant = cv2.dilate(
        plant.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    rescue_colour = (
        (val >= 34)
        & ((green >= blue + 5) | (red >= blue + 4))
        & ((excess_green >= 7) | (red >= green - 12))
    )
    plant |= plant_band & near_plant & rescue_colour & (coarse_alpha >= 128)

    mask = plant | pot
    alpha = alpha_from_binary_mask(mask, sigma=max(0.42, min(0.72, 0.52 * scale)))
    rgba_array = np.dstack([rgb_full, alpha]).astype(np.uint8)
    rgba_array[alpha == 0, :3] = 0
    rgba = decontaminate_dark_boundary(Image.fromarray(rgba_array).convert("RGBA"))

    cloth_like = plant & (val < 52) & (
        (blue >= red + 2) | ((sat < 18) & (green <= red + 5))
    )
    diagnostics = {
        "applied": True,
        "method": "frame-filling-white-pot-plus-original-colour-stem-mask",
        "pot_style": "white_or_cream",
        "subject_roi": [0, 0, width, height],
        "pot_top_y": pot_top,
        "pot_top_curve_y_range": [
            int(np.floor(top_curve[pot_left : pot_right + 1].min())),
            int(np.ceil(top_curve[pot_left : pot_right + 1].max())),
        ],
        "pot_bbox": [
            int(np.where(pot)[1].min()),
            int(np.where(pot)[0].min()),
            int(np.where(pot)[1].max() + 1),
            int(np.where(pot)[0].max() + 1),
        ],
        "plant_pixels": int(plant.sum()),
        "pot_pixels": int(pot.sum()),
        "foreground_pixels": int((alpha > 0).sum()),
        "filled_small_hole_pixels": filled_hole_pixels,
        "filled_small_hole_count": filled_hole_count,
        "internal_cloth_like_pixels": int(cloth_like.sum()),
        "internal_cloth_like_ratio": float(cloth_like.sum() / max(1, plant.sum())),
    }
    return rgba, diagnostics


def refine_frame_filling_dark_backdrop_auto(
    original: Image.Image,
    coarse_rgba: Image.Image,
) -> tuple[Image.Image, dict]:
    """Choose the white/cream-pot or orange-pot frame-filling workflow."""
    try:
        return refine_frame_filling_dark_backdrop_white_pot(original, coarse_rgba)
    except RuntimeError as white_exc:
        rgba, diagnostics = refine_frame_filling_dark_backdrop_rice(original, coarse_rgba)
        diagnostics["white_pot_mode_reason"] = str(white_exc)
        return rgba, diagnostics


def decontaminate_edge(rgba: Image.Image, core_alpha: int = 180) -> Image.Image:
    arr = np.array(rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgba.convert("RGBA")

    x1, x2 = max(0, xs.min() - 8), min(arr.shape[1], xs.max() + 9)
    y1, y2 = max(0, ys.min() - 8), min(arr.shape[0], ys.max() + 9)
    crop = arr[y1:y2, x1:x2].copy()
    crop_alpha = crop[:, :, 3]

    edge = (crop_alpha > 0) & (crop_alpha < 255)
    core = crop_alpha >= core_alpha
    if edge.any() and core.any():
        _, nearest = ndi.distance_transform_edt(~core, return_indices=True)
        nearest_rgb = crop[:, :, :3][nearest[0], nearest[1]]
        crop[:, :, :3][edge] = nearest_rgb[edge]

    crop[:, :, :3][crop_alpha == 0] = 0
    arr[y1:y2, x1:x2] = crop
    arr[:, :, :3][alpha == 0] = 0
    return Image.fromarray(arr).convert("RGBA")


def compose_on_background(rgba: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = Image.new("RGBA", rgba.size, (*color, 255))
    base.alpha_composite(rgba)
    return base.convert("RGB")


def save_layered_psd(original: Image.Image, rgba: Image.Image, psd_path: Path) -> None:
    if PSDImage is None or PixelLayer is None or Compression is None:
        raise RuntimeError("psd-tools is required to write the layered PSD")

    original_rgb = original.convert("RGB")
    cutout_rgba = rgba.convert("RGBA")
    black_rgb = Image.new("RGB", original_rgb.size, (0, 0, 0))
    mask_rgb = cutout_rgba.getchannel("A").convert("RGB")

    psd = PSDImage.new("RGB", original_rgb.size, color=0)

    for name, image, visible in [
        ("black_background_RGB_000", black_rgb, True),
        ("mask_layer_alpha", mask_rgb, False),
        ("original_hidden", original_rgb, False),
    ]:
        layer = PixelLayer.frompil(image, psd, name=name)
        layer.visible = visible

    record, channels = PixelLayer._build_layer_record_and_channels(
        cutout_rgba,
        "cutout_layer",
        0,
        0,
        Compression.RLE,
    )
    cutout_layer = PixelLayer(psd, record, channels)
    cutout_layer.visible = True
    # Insert above the black background layer for the delivery composite.
    psd.insert(1, cutout_layer)
    psd.save(psd_path)


def inspect_layered_psd(psd_path: Path) -> dict:
    if PSDImage is None:
        return {
            "exists": psd_path.exists(),
            "required_layers_present": False,
            "error": "psd-tools is not installed",
        }

    try:
        psd = PSDImage.open(psd_path)
    except Exception as exc:
        return {
            "exists": psd_path.exists(),
            "required_layers_present": False,
            "error": str(exc),
        }

    layers = [
        {
            "name": getattr(layer, "name", "") or "",
            "visible": bool(getattr(layer, "visible", False)),
            "size": list(getattr(layer, "size", (0, 0))),
        }
        for layer in psd
    ]
    names = [layer["name"] for layer in layers]

    def has_name(fragment: str) -> bool:
        return any(fragment in name for name in names)

    cutout_alpha_ok = False
    black_background_ok = False
    for layer in psd:
        name = getattr(layer, "name", "") or ""
        try:
            image = layer.topil().convert("RGBA")
        except Exception:
            continue
        arr = np.array(image)
        if name == "cutout_layer":
            alpha = arr[:, :, 3]
            cutout_alpha_ok = bool((alpha > 0).any() and (alpha == 0).any())
        if name.startswith("black_background"):
            black_background_ok = bool((arr[:, :, :3] == 0).all() and (arr[:, :, 3] == 255).all())

    return {
        "exists": psd_path.exists(),
        "layers": layers,
        "required_layers_present": all(
            [
                has_name("black_background"),
                has_name("cutout_layer"),
                has_name("mask_layer"),
                has_name("original_hidden"),
            ]
        ),
        "visible_delivery_layers_ok": all(
            [
                any(layer["name"].startswith("black_background") and layer["visible"] for layer in layers),
                any(layer["name"] == "cutout_layer" and layer["visible"] for layer in layers),
                any(layer["name"].startswith("mask_layer") and not layer["visible"] for layer in layers),
                any(layer["name"] == "original_hidden" and not layer["visible"] for layer in layers),
            ]
        ),
        "cutout_layer_alpha_ok": cutout_alpha_ok,
        "black_background_rgb_000_ok": black_background_ok,
    }


def cleanup_transient_outputs(image_path: Path, out_dir: Path) -> list[str]:
    stem = image_path.stem
    patterns = [
        f"{stem}_cutout_transparent*.png",
        f"{stem}_compare*.jpg",
        f"{stem}_corrected_zoom*.jpg",
        f"{stem}_alpha_mask*.png",
        f"{stem}_white_edge_check*.jpg",
        f"{stem}_cutout_QA_report*.json",
        f"{stem}_original_backup*",
        f"{stem}_clean_black_v*.png",
        f"{stem}_cutout_v*.psd",
    ]
    removed: list[str] = []
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def output_file_status(paths: dict[str, Path]) -> dict[str, dict]:
    status = {}
    for key, path in paths.items():
        exists = path.exists()
        status[key] = {
            "path": str(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else 0,
        }
    return status


def dark_backdrop_salient_coverage(
    original: Image.Image,
    alpha: np.ndarray,
) -> dict:
    """Estimate whether long chromatic rice structures were lost on dark cloth.

    This is intentionally independent of rembg: rembg can miss the same long,
    disconnected leaves that the final QA needs to detect.
    """
    if cv2 is None:
        return {"applicable": False, "reason": "OpenCV is not installed"}

    rgb = np.array(original.convert("RGB"))
    height, _ = alpha.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = [rgb[:, :, index].astype(np.int16) for index in range(3)]

    upper_value_median = float(np.median(val[: max(1, height // 2)]))
    if upper_value_median >= 90:
        return {
            "applicable": False,
            "reason": "upper image is not predominantly dark",
            "upper_value_median": upper_value_median,
        }

    y_grid = np.arange(height)[:, None]
    green_leaf = (
        (hue >= 16)
        & (hue <= 100)
        & (sat >= 28)
        & (val >= 72)
        & (green >= blue + 6)
        & (green >= red - 24)
    )
    tan_sheath = (
        (hue <= 42)
        & (sat >= 24)
        & (val >= 78)
        & (red >= blue + 6)
        & (green >= blue - 2)
    )
    candidate = (y_grid < int(height * 0.82)) & (green_leaf | tan_sheath)
    candidate = cv2.medianBlur((candidate * 255).astype(np.uint8), 3) > 127
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    salient = np.zeros_like(candidate, dtype=bool)
    kept_components = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        long_axis = max(component_width, component_height)
        short_axis = max(1, min(component_width, component_height))
        aspect = long_axis / short_axis
        fill_ratio = area / max(1, component_width * component_height)
        if (
            (area >= 28 and long_axis >= 16 and aspect >= 2.2)
            or (area >= 120 and fill_ratio <= 0.30)
        ):
            salient[labels == index] = True
            kept_components += 1

    salient_pixels = int(salient.sum())
    if salient_pixels < 500:
        return {
            "applicable": False,
            "reason": "too few elongated chromatic foreground pixels",
            "upper_value_median": upper_value_median,
            "salient_pixels": salient_pixels,
        }

    return {
        "applicable": True,
        "upper_value_median": upper_value_median,
        "salient_component_count": kept_components,
        "salient_pixels": salient_pixels,
        "alpha_any_coverage": float(((alpha > 0) & salient).sum() / salient_pixels),
        "alpha_hard_coverage": float(((alpha >= 128) & salient).sum() / salient_pixels),
    }


def coarse_model_retention_metrics(
    coarse_rgba: Image.Image,
    final_rgba: Image.Image,
) -> dict:
    coarse_alpha = np.array(coarse_rgba.convert("RGBA").getchannel("A"))
    final_alpha = np.array(final_rgba.convert("RGBA").getchannel("A"))
    coarse_hard = coarse_alpha >= 192
    hard_pixels = int(coarse_hard.sum())
    if hard_pixels == 0:
        return {"applicable": False, "reason": "coarse model has no hard-alpha pixels"}
    return {
        "applicable": True,
        "coarse_hard_pixels": hard_pixels,
        "retained_by_any_final_alpha": float(
            ((final_alpha > 0) & coarse_hard).sum() / hard_pixels
        ),
        "retained_by_hard_final_alpha": float(
            ((final_alpha >= 128) & coarse_hard).sum() / hard_pixels
        ),
    }


def classify_qa(report: dict) -> str:
    required_checks = [
        "source_image_exists",
        "alpha_channel_present",
        "subject_pixels_exist",
        "transparent_pixels_exist",
        "cutout_size_matches_original",
        "pure_black_background_where_alpha_zero",
        "final_outputs_exist",
        "layered_psd_required_layers_present",
        "layered_psd_visibility_ok",
        "layered_psd_cutout_alpha_ok",
        "layered_psd_black_background_ok",
    ]
    checks = report.get("qa_checks", {})
    if not all(checks.get(name) for name in required_checks):
        return "fail"
    if report.get("qa_warnings"):
        return "review_required"
    return "pass"


def qa_metrics(
    rgba: Image.Image,
    black: Image.Image,
    original: Image.Image,
    source_layer: str,
    image_path: Path,
) -> dict:
    alpha = np.array(rgba.getchannel("A"))
    black_arr = np.array(black)
    background = alpha == 0
    total_pixels = int(alpha.size)
    nonzero_alpha_pixels = int((alpha > 0).sum())
    transparent_pixels = int((alpha == 0).sum())
    semi_transparent_pixels = int(((alpha > 0) & (alpha < 255)).sum())
    foreground_ratio = nonzero_alpha_pixels / total_pixels if total_pixels else 0.0
    semi_transparent_ratio = semi_transparent_pixels / max(1, nonzero_alpha_pixels)
    ys, xs = np.where(alpha > 0)
    bbox = None
    semi_edge_distance_p95 = 0.0
    semi_edge_distance_p99 = 0.0
    semi_edge_distance_max = 0.0
    opaque_core_ratio = 0.0
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        alpha_crop = alpha[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        core = alpha_crop >= 180
        semi = (alpha_crop > 0) & (alpha_crop < 255)
        opaque_core_ratio = float(core.sum() / max(1, (alpha_crop > 0).sum()))
        if core.any() and semi.any():
            distance_from_core = ndi.distance_transform_edt(~core)[semi]
            semi_edge_distance_p95 = float(np.percentile(distance_from_core, 95))
            semi_edge_distance_p99 = float(np.percentile(distance_from_core, 99))
            semi_edge_distance_max = float(distance_from_core.max())
    bg_is_pure_black = bool((black_arr[background] == 0).all()) if background.any() else False
    qa_warnings = []
    if foreground_ratio > 0.98:
        qa_warnings.append("alpha foreground covers >98% of image; background may not be removed.")
    if 0 < foreground_ratio < 0.005:
        qa_warnings.append("alpha foreground covers <0.5% of image; subject may be mostly missing.")
    # A single insect leg, root fibre, or dirty pot-rim speck can legitimately
    # create a large maximum distance.  Warn on a broad soft fringe instead of
    # one isolated tail: either the 95th percentile is already wide, or both
    # the 99th percentile and the absolute maximum indicate persistent haze.
    if semi_edge_distance_p95 > 3.0 or (
        semi_edge_distance_p99 > 4.0 and semi_edge_distance_max > 20.0
    ):
        qa_warnings.append(
            "semi-transparent pixels extend too far from the opaque core; check for haze or cloth residue."
        )
    elif semi_transparent_ratio > 0.35 and opaque_core_ratio < 0.75:
        qa_warnings.append(
            "the subject has a large semi-transparent fraction and a weak opaque core; inspect thin structures and cloth haze."
        )

    qa_checks = {
        "source_image_exists": image_path.exists(),
        "alpha_channel_present": rgba.mode == "RGBA",
        "subject_pixels_exist": nonzero_alpha_pixels > 0,
        "transparent_pixels_exist": transparent_pixels > 0,
        "cutout_size_matches_original": rgba.size == original.size,
        "pure_black_background_where_alpha_zero": bg_is_pure_black,
        "final_outputs_exist": False,
        "layered_psd_required_layers_present": False,
        "layered_psd_visibility_ok": False,
        "layered_psd_cutout_alpha_ok": False,
        "layered_psd_black_background_ok": False,
    }

    manual_review_points = [
        "open final PSD and inspect black_background_RGB_000 plus cutout_layer at high zoom",
        "inspect clean_black PNG for cloth ghosts, bright residues, broken leaves, panicles, insect legs, antennae, and pot-bottom shadows",
        "inspect mask_layer_alpha inside PSD for leaf gaps, seedling center, pot mouth, thin leaf tips, awns, and internal dark holes",
        "temporarily inspect the cutout on white at 200%-400% zoom for black/blue cloth fringes; do not retain the white QA file",
        "confirm original_hidden is hidden and not contributing to the visible delivery composite",
    ]
    salient_coverage = dark_backdrop_salient_coverage(original, alpha)
    if (
        salient_coverage.get("applicable")
        and float(salient_coverage.get("alpha_hard_coverage", 0.0)) < 0.85
    ):
        qa_warnings.append(
            "dark-backdrop salient rice coverage is <85%; long leaves, pale sheaths, or disconnected tips may be missing."
        )

    return {
        "source_layer": source_layer,
        "source_image": str(image_path),
        "original_size": list(original.size),
        "cutout_size": list(rgba.size),
        "alpha_present": qa_checks["alpha_channel_present"],
        "bbox_alpha_gt_0": bbox,
        "total_pixels": total_pixels,
        "nonzero_alpha_pixels": nonzero_alpha_pixels,
        "transparent_pixels": transparent_pixels,
        "semi_transparent_pixels": semi_transparent_pixels,
        "foreground_alpha_ratio": foreground_ratio,
        "semi_transparent_subject_ratio": semi_transparent_ratio,
        "opaque_core_ratio": opaque_core_ratio,
        "semi_edge_distance_p95_px": semi_edge_distance_p95,
        "semi_edge_distance_p99_px": semi_edge_distance_p99,
        "semi_edge_distance_max_px": semi_edge_distance_max,
        "pure_black_background_where_alpha_zero": bg_is_pure_black,
        "dark_backdrop_salient_coverage": salient_coverage,
        "qa_checks": qa_checks,
        "qa_warnings": qa_warnings,
        "manual_review_required": True,
        "manual_review_points": manual_review_points,
        "qa_status": "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--psd", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ignored; final files are always written beside the source image.",
    )
    parser.add_argument("--force-rembg", action="store_true")
    parser.add_argument(
        "--segmentation-mode",
        choices=(
            "auto",
            "rembg",
            "dark-rice",
            "dark-rice-gray-pot",
            "dark-rice-frame",
        ),
        default="auto",
        help=(
            "auto uses existing PSD layers when available and otherwise tries gray-pot, "
            "isolated orange-pot, then frame-filling white/cream-pot or orange-pot rice "
            "colour refinement; explicit "
            "dark-rice modes require their respective refinements."
        ),
    )
    parser.add_argument(
        "--dark-rice-value-floor",
        type=int,
        default=90,
        help=(
            "Minimum HSV value retained by the isolated dark-rice colour rebuild "
            "(default: 90; calibrate from temporary white-background QA when exposure differs)."
        ),
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    ignored_output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    out_dir = image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.psd:
        psd_path = args.psd.expanduser().resolve()
    else:
        layered_delivery_psd = image_path.with_name(f"{image_path.stem}_cutout.psd")
        plain_psd = image_path.with_suffix(".psd")
        psd_path = layered_delivery_psd if layered_delivery_psd.exists() else plain_psd

    original = Image.open(image_path).convert("RGB")

    refinement_diagnostics: dict = {"applied": False}
    coarse_rgba_for_qa: Image.Image | None = None
    if psd_path.exists() and not args.force_rembg:
        rgba, source_layer = load_cutout_from_psd(psd_path)
    else:
        rgba, source_layer = load_cutout_with_rembg(image_path)
        coarse_rgba_for_qa = rgba.copy()
        if args.segmentation_mode == "dark-rice-frame":
            rgba, refinement_diagnostics = refine_frame_filling_dark_backdrop_auto(original, rgba)
            source_layer = f"{source_layer}+dark-rice-frame-colour-refinement"
        elif args.segmentation_mode == "dark-rice-gray-pot":
            rgba, refinement_diagnostics = refine_dark_backdrop_gray_pot_rice(original, rgba)
            source_layer = f"{source_layer}+dark-rice-gray-pot-colour-refinement"
        elif args.segmentation_mode == "dark-rice":
            rgba, refinement_diagnostics = refine_dark_backdrop_rice(
                original,
                rgba,
                value_floor=args.dark_rice_value_floor,
            )
            source_layer = f"{source_layer}+dark-rice-colour-refinement"
        elif args.segmentation_mode == "auto":
            try:
                rgba, refinement_diagnostics = refine_dark_backdrop_gray_pot_rice(
                    original, rgba
                )
                source_layer = f"{source_layer}+dark-rice-gray-pot-colour-refinement"
            except RuntimeError as gray_pot_exc:
                try:
                    rgba, refinement_diagnostics = refine_dark_backdrop_rice(
                        original,
                        rgba,
                        value_floor=args.dark_rice_value_floor,
                    )
                    refinement_diagnostics["gray_pot_mode_reason"] = str(gray_pot_exc)
                    source_layer = f"{source_layer}+dark-rice-colour-refinement"
                except RuntimeError as isolated_exc:
                    try:
                        rgba, refinement_diagnostics = refine_frame_filling_dark_backdrop_auto(
                            original, rgba
                        )
                        refinement_diagnostics["gray_pot_mode_reason"] = str(gray_pot_exc)
                        refinement_diagnostics["isolated_mode_reason"] = str(isolated_exc)
                        source_layer = f"{source_layer}+dark-rice-frame-colour-refinement"
                    except RuntimeError as frame_exc:
                        refinement_diagnostics = {
                            "applied": False,
                            "fallback": "rembg",
                            "gray_pot_reason": str(gray_pot_exc),
                            "isolated_reason": str(isolated_exc),
                            "frame_filling_reason": str(frame_exc),
                        }

    if rgba.size != original.size:
        raise RuntimeError(f"Cutout size {rgba.size} does not match original size {original.size}")

    rgba = decontaminate_edge(rgba)
    black = compose_on_background(rgba, (0, 0, 0))

    stem = image_path.stem
    paths = {
        "black": out_dir / f"{stem}_clean_black.png",
        "psd": out_dir / f"{stem}_cutout.psd",
    }

    black.save(paths["black"])
    save_layered_psd(original, rgba, paths["psd"])
    removed_transient_outputs = cleanup_transient_outputs(image_path, out_dir)

    report = qa_metrics(rgba, black, original, source_layer, image_path)
    report["outputs"] = {key: str(path) for key, path in paths.items()}
    report["output_policy"] = "final files are always written beside the source image"
    if ignored_output_dir is not None:
        report["ignored_output_dir"] = str(ignored_output_dir)
    report["removed_transient_outputs"] = removed_transient_outputs
    report["refinement_diagnostics"] = refinement_diagnostics
    if coarse_rgba_for_qa is not None:
        coarse_retention = coarse_model_retention_metrics(coarse_rgba_for_qa, rgba)
        report["coarse_model_retention"] = coarse_retention
        if (
            coarse_retention.get("applicable")
            and float(coarse_retention.get("retained_by_hard_final_alpha", 0.0)) < 0.80
        ):
            report["qa_warnings"].append(
                "final mask retains <80% of the coarse model's high-confidence subject; inspect pot integrity and missing leaves."
            )
    if refinement_diagnostics.get("applied"):
        dark_ratio = float(
            refinement_diagnostics.get(
                "internal_background_like_ratio_remaining",
                refinement_diagnostics.get(
                    "internal_background_like_ratio",
                    refinement_diagnostics.get("internal_cloth_like_ratio", 0.0),
                ),
            )
        )
        if dark_ratio > 0.01:
            report["qa_warnings"].append(
                "internal background-like pixels exceed 1% of the refined rice mask; inspect leaf gaps and stem bases."
            )
    report["output_file_status"] = output_file_status(paths)
    report["qa_checks"]["final_outputs_exist"] = all(
        item["exists"] and item["bytes"] > 0 for item in report["output_file_status"].values()
    )
    psd_check = inspect_layered_psd(paths["psd"])
    report["layered_psd_check"] = psd_check
    report["qa_checks"]["layered_psd_required_layers_present"] = bool(
        psd_check.get("required_layers_present")
    )
    report["qa_checks"]["layered_psd_visibility_ok"] = bool(
        psd_check.get("visible_delivery_layers_ok")
    )
    report["qa_checks"]["layered_psd_cutout_alpha_ok"] = bool(
        psd_check.get("cutout_layer_alpha_ok")
    )
    report["qa_checks"]["layered_psd_black_background_ok"] = bool(
        psd_check.get("black_background_rgb_000_ok")
    )
    report["qa_status"] = classify_qa(report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
