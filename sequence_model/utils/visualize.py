import os, re, sys, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL_COLUMNS, ATTENTION_DIR

_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONT_DEV  = os.path.join(_DIR, "NotoSansDevanagari-Regular.ttf")
_FONT_BOLD = os.path.join(_DIR, "NotoSansDevanagari-Medium.ttf")   

import matplotlib.font_manager as _fm
_FONT_LATIN = next(
    (f.fname for f in _fm.fontManager.ttflist
     if f.name == "DejaVu Sans" and "Oblique" not in f.fname and "Bold" not in f.fname),
    None,
)

_DEV_RANGE = re.compile(r'([\u0900-\u097F\u200d ]+|[^\u0900-\u097F\u200d]+)')


def _clean(text: str) -> str:
    """Strip ZWJ (U+200D) — purely typographic in Nepali, safe to remove for display."""
    return text.replace('\u200d', '')


def _pil_fonts(size: int):
    """Return (devanagari_font, latin_font) at the requested pixel size."""
    dev_path  = _FONT_DEV  if os.path.exists(_FONT_DEV)  else None
    bold_path = _FONT_BOLD if os.path.exists(_FONT_BOLD) else dev_path

    dev  = ImageFont.truetype(dev_path,   size) if dev_path  else ImageFont.load_default()
    lat  = ImageFont.truetype(_FONT_LATIN, size) if _FONT_LATIN else ImageFont.load_default()
    return dev, lat


def _text_width(text: str, dev_font, lat_font) -> int:
    """Measure pixel width of a mixed Devanagari+Latin string."""
    w = 0
    for seg in _DEV_RANGE.findall(_clean(text)):
        is_dev = bool(re.match(r'^[\u0900-\u097F ]+$', seg))
        font   = dev_font if is_dev else lat_font
        bb = font.getbbox(seg)
        w += bb[2] - bb[0]
    return w


def _draw_mixed(draw: ImageDraw.ImageDraw, xy, text: str,
                dev_font, lat_font, fill=(0, 0, 0)):
    """Draw a mixed Devanagari+Latin string at (x, y) using the correct font per segment."""
    x, y = xy
    for seg in _DEV_RANGE.findall(_clean(text)):
        is_dev = bool(re.match(r'^[\u0900-\u097F ]+$', seg))
        font   = dev_font if is_dev else lat_font
        lang   = 'ne' if is_dev else 'en'
        draw.text((x, y), seg, font=font, fill=fill, language=lang)
        bb = font.getbbox(seg)
        x += bb[2] - bb[0]


def _text_height(font) -> int:
    """Approximate line height for a given font."""
    bb = font.getbbox("अAg|")
    return bb[3] - bb[1]



def _cmap_rgba(value: float, vmin: float, vmax: float, cmap_name="YlOrRd"):
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap_name)(norm(value))
    return tuple(int(c * 255) for c in rgba[:3])


def _contrast_color(bg_rgb):
    """Return black or white depending on background luminance."""
    r, g, b = bg_rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if lum < 140 else (0, 0, 0)



def plot_attention_heatmap(
    tokens:      List[str],
    attention:   np.ndarray,
    pred_labels: List[str],
    text_id:     str = "sample",
    save_dir:    str = ATTENTION_DIR,
) -> str:
    """
    Publication-quality heatmap rendered entirely in Pillow.
    Both Devanagari tokens and Latin labels/IDs render without boxes.
    """
    attention = np.array(attention, dtype=float)
    attention = attention / (attention.sum() + 1e-9)
    CELL_H       = 72          # height of each heat cell
    PADDING      = 18          # outer margin
    TITLE_H      = 72          # space for two-line title
    CBAR_W       = 48          # colour-bar width
    TICK_H       = 90          # height below cells for rotated token labels
    TOKEN_SIZE   = 22          # token font size (px)
    VAL_SIZE     = 18          # in-cell value font size
    TITLE_SIZE   = 22
    CBAR_STEPS   = 200

    dev_tok, lat_tok   = _pil_fonts(TOKEN_SIZE)
    dev_val, lat_val   = _pil_fonts(VAL_SIZE)
    dev_ttl, lat_ttl   = _pil_fonts(TITLE_SIZE)

    tok_h    = _text_height(dev_tok)
    n        = len(tokens)
    CELL_W   = max(80, max(_text_width(t, dev_tok, lat_tok) for t in tokens) + 20)

    total_w  = PADDING * 2 + n * CELL_W + CBAR_W + 20
    total_h  = PADDING + TITLE_H + CELL_H + TICK_H + PADDING

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    vmin, vmax = attention.min(), attention.max() + 1e-9

    label_str = ", ".join(pred_labels) if pred_labels else "none"
    line1 = f"Attention Heatmap  |  id: {text_id}"
    line2 = f"Predicted: {label_str}"

    _draw_mixed(draw, (PADDING, PADDING),
                line1, dev_ttl, lat_ttl, fill=(30, 30, 30))
    _draw_mixed(draw, (PADDING, PADDING + TITLE_SIZE + 6),
                line2, dev_ttl, lat_ttl, fill=(80, 80, 80))

    cell_top = PADDING + TITLE_H
    for i, (tok, w) in enumerate(zip(tokens, attention)):
        if tok in {"।"}:
            continue
        x0 = PADDING + i * CELL_W
        x1 = x0 + CELL_W - 2
        bg = _cmap_rgba(w, vmin, vmax)
        draw.rectangle([x0, cell_top, x1, cell_top + CELL_H - 2], fill=bg)

        # Weight value centred in cell
        val_str = f"{w:.3f}"
        vw      = _text_width(val_str, dev_val, lat_val)
        vh      = _text_height(dev_val)
        vx      = x0 + (CELL_W - vw) // 2
        vy      = cell_top + (CELL_H - vh) // 2
        _draw_mixed(draw, (vx, vy), val_str, dev_val, lat_val,
                    fill=_contrast_color(bg))

        # Token label (rotated 45°) — render upright then rotate
        tok_clean = _clean(tok)
        tw = _text_width(tok_clean, dev_tok, lat_tok)
        th = _text_height(dev_tok)
        tok_img  = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
        tok_draw = ImageDraw.Draw(tok_img)
        _draw_mixed(tok_draw, (2, 2), tok_clean, dev_tok, lat_tok, fill=(40, 40, 40))
        tok_rot  = tok_img.rotate(45, expand=True, resample=Image.BICUBIC)

        # Paste below cell, centred
        px = x0 + (CELL_W - tok_rot.width) // 2
        py = cell_top + CELL_H + 4
        img.paste(tok_rot, (px, py), tok_rot)

    bar_x = PADDING + n * CELL_W + 10
    bar_y = cell_top
    bar_h = CELL_H - 2
    for step in range(CBAR_STEPS):
        v    = vmin + (vmax - vmin) * (1 - step / CBAR_STEPS)
        col  = _cmap_rgba(v, vmin, vmax)
        y0   = bar_y + int(step * bar_h / CBAR_STEPS)
        y1   = bar_y + int((step + 1) * bar_h / CBAR_STEPS)
        draw.rectangle([bar_x, y0, bar_x + 18, y1], fill=col)

    # Colour bar labels
    _, lat_sm = _pil_fonts(14)
    draw.text((bar_x + 22, bar_y),            f"{vmax:.2f}", font=lat_sm, fill=(60, 60, 60))
    draw.text((bar_x + 22, bar_y + bar_h//2), f"{(vmin+vmax)/2:.2f}", font=lat_sm, fill=(60, 60, 60))
    draw.text((bar_x + 22, bar_y + bar_h - 12), f"{vmin:.2f}", font=lat_sm, fill=(60, 60, 60))

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{text_id}_heatmap.png")
    img.save(path, dpi=(300, 300))
    return path



def plot_attention_bar(
    tokens:      List[str],
    attention:   np.ndarray,
    pred_labels: List[str],
    text_id:     str = "sample",
    save_dir:    str = ATTENTION_DIR,
    top_k:       int = 15,
) -> str:
    """
    Horizontal bar chart of top-K tokens by attention weight.
    Dual-font: Devanagari tokens render correctly alongside Latin labels.
    """
    attention = np.array(attention, dtype=float)
    attention = attention / (attention.sum() + 1e-9)

    k       = min(top_k, len(tokens))
    idx     = np.argsort(attention)[::-1][:k]
    top_w   = attention[idx]
    top_t   = [_clean(tokens[i]) for i in idx]

    BAR_H      = 36
    PADDING    = 20
    TITLE_H    = 60
    LABEL_W    = max(140, max(_text_width(t, *_pil_fonts(20)) for t in top_t) + 16)
    BAR_MAX_W  = 340
    VAL_W      = 60
    TOKEN_SIZE = 20
    TITLE_SIZE = 22

    total_w = PADDING + LABEL_W + BAR_MAX_W + VAL_W + PADDING
    total_h = PADDING + TITLE_H + k * (BAR_H + 6) + PADDING

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    dev_tok, lat_tok = _pil_fonts(TOKEN_SIZE)
    dev_ttl, lat_ttl = _pil_fonts(TITLE_SIZE)
    _, lat_val       = _pil_fonts(16)

    vmin, vmax = top_w.min(), top_w.max() + 1e-9

    label_str = ", ".join(pred_labels) if pred_labels else "none"
    _draw_mixed(draw, (PADDING, PADDING),
                f"Top-{k} Tokens by Attention  |  id: {text_id}",
                dev_ttl, lat_ttl, fill=(30, 30, 30))
    _draw_mixed(draw, (PADDING, PADDING + TITLE_SIZE + 6),
                f"Predicted: {label_str}",
                dev_ttl, lat_ttl, fill=(80, 80, 80))

    bar_x = PADDING + LABEL_W
    for rank, (tok, w) in enumerate(zip(reversed(top_t), reversed(top_w))):
        if tok in {"।"}:
            continue
        y = PADDING + TITLE_H + rank * (BAR_H + 6)

        # Token label (right-aligned)
        tw = _text_width(tok, dev_tok, lat_tok)
        tx = PADDING + LABEL_W - tw - 8
        th = _text_height(dev_tok)
        ty = y + (BAR_H - th) // 2
        _draw_mixed(draw, (tx, ty), tok, dev_tok, lat_tok, fill=(40, 40, 40))

        # Bar rectangle
        bar_len = int(w / vmax * BAR_MAX_W)
        bg = _cmap_rgba(w, vmin, vmax)
        draw.rectangle([bar_x, y + 4, bar_x + bar_len, y + BAR_H - 4],
                       fill=bg, outline=(200, 200, 200))

        # Weight label
        draw.text((bar_x + bar_len + 6, y + (BAR_H - 16) // 2),
                  f"{w:.4f}", font=lat_val, fill=(60, 60, 60))

    draw.text((bar_x + BAR_MAX_W // 2 - 50, total_h - PADDING - 4),
              "Attention Weight", font=lat_val, fill=(100, 100, 100))

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{text_id}_bar.png")
    img.save(path, dpi=(300, 300))
    return path



def plot_batch_attention(
    tokens_list: List[List[str]],
    attentions:  np.ndarray,
    pred_probs:  np.ndarray,
    ids:         List[str],
    threshold:   float = 0.5,
    save_dir:    str   = ATTENTION_DIR,
    top_k:       int   = 15,
) -> List[str]:
    saved = []
    for i in range(len(tokens_list)):
        pred_labels = [LABEL_COLUMNS[j] for j, p in enumerate(pred_probs[i]) if p >= threshold]
        tokens = tokens_list[i]
        attn   = attentions[i, :len(tokens)]
        p1 = plot_attention_heatmap(tokens, attn, pred_labels, ids[i], save_dir)
        p2 = plot_attention_bar(tokens, attn, pred_labels, ids[i], save_dir, top_k)
        saved += [p1, p2]
        print(f"  Saved: {os.path.basename(p1)}, {os.path.basename(p2)}")
    return saved



def plot_training_curves(history: dict, save_dir: str = ATTENTION_DIR) -> str:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#E07B39", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#3E8EDE", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontsize=13); axes[0].legend()

    axes[1].plot(epochs, history["val_f1_micro"], label="Val F1 (micro)", color="#5ABF7E", linewidth=2)
    axes[1].plot(epochs, history["val_f1_macro"], label="Val F1 (macro)", color="#C85AC8", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Validation F1 Scores", fontsize=13); axes[1].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved → {path}")
    return path
