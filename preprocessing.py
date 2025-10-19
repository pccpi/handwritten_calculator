from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ---------- базовые утилиты ----------

def pil_from_canvas(canvas_img: Optional[np.ndarray]) -> Image.Image:
    """
    Streamlit canvas -> PIL RGB.
    """
    if canvas_img is None:
        return Image.new("RGB", (280, 280), (255, 255, 255))
    arr = canvas_img
    if arr.max() <= 1.0001:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr[:, :, :3], mode="RGB")

def _binarize_grayscale(gray_np: np.ndarray, pct: float = 75) -> np.ndarray:
    """
    Порог: пиксели темнее thr считаем штрихом (True -> 255), фон -> 0.
    Возврат: uint8, где 255 = штрих, 0 = фон.
    """
    thr = np.percentile(gray_np, pct)
    return (gray_np < thr).astype(np.uint8) * 255

def _ink_mask(bin_np: np.ndarray) -> np.ndarray:
    # после _binarize_grayscale: 255 = штрих
    return (bin_np > 0).astype(np.uint8)

def has_ink(img_rgb: Image.Image, min_pixels: int = 30) -> bool:
    """Есть ли на изображении «чернила» (штрихи)?"""
    gray = img_rgb.convert("L")
    bin_np = _binarize_grayscale(np.array(gray), pct=75)
    return int(_ink_mask(bin_np).sum()) >= min_pixels

# ---------- MNIST-центрирование цифр ----------

def _center_to_mnist_canvas(img_l_white_on_black: Image.Image) -> Image.Image:
    """
    На входе L-картинка с БЕЛЫМИ штрихами на ЧЁРНОМ фоне.
    Возвращаем 28×28, масштаб 20px по большей стороне, сдвиг к центру масс (14,14).
    """
    arr = np.array(img_l_white_on_black)
    ys, xs = np.where(arr > 0)  # белые пиксели = штрих
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("L", (28, 28), 0)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    digit = img_l_white_on_black.crop((x_min, y_min, x_max + 1, y_max + 1))

    # лёгкий блюр сгладит зубчатости
    digit = digit.filter(ImageFilter.GaussianBlur(radius=1))

    # Масштаб: большая сторона = 20px
    w, h = digit.size
    scale = 20.0 / max(w, h) if max(w, h) > 0 else 1.0
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit = digit.resize((new_w, new_h), Image.LANCZOS)

    # Вписываем в 28×28
    canvas = Image.new("L", (28, 28), 0)
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2
    canvas.paste(digit, (ox, oy))

    # Сдвиг к центру масс
    arr = np.array(canvas, dtype=np.float32)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return canvas
    cy = ys.mean()
    cx = xs.mean()
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))

    shifted = Image.new("L", (28, 28), 0)
    shifted.paste(canvas, (shift_x, shift_y))
    return shifted

# ---------- ПРЕПРОЦЕССИНГ: цифры и оператор ----------

def preprocess_digit(img_rgb: Image.Image) -> Tuple[np.ndarray, Image.Image]:
    """
    ЦИФРЫ -> БЕЛОЕ по ЧЁРНОМУ (MNIST-стиль):
      1) bin: чёрный штрих на белом -> массив, где штрих=255, фон=0
      2) БЕЗ invert (!) — уже белое на чёрном
      3) bbox -> масштаб до 20px -> центр масс
      4) [0..1] + L2-нормализация по образцу
      5) shape (1, 28, 28)
    """
    gray = img_rgb.convert("L")

    # Штрихи (чёрные) -> 255, фон -> 0  => это уже БЕЛЫЙ штрих на ЧЁРНОМ в смысле интенсивностей
    bin_np = _binarize_grayscale(np.array(gray), pct=75)
    white_on_black = Image.fromarray(bin_np, mode="L")  # НИКАКОГО invert здесь!

    # MNIST-центрирование
    mnist_like = _center_to_mnist_canvas(white_on_black)

    # Нормализация
    v = (np.array(mnist_like).astype("float32") / 255.0).reshape(1, -1)  # (1,784)
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    x = v_norm.reshape(1, 28, 28)

    preview = mnist_like.resize((112, 112), Image.NEAREST)  # в превью должен быть ЧЁРНЫЙ фон и БЕЛАЯ цифра
    return x, preview

def preprocess_operator(img_rgb: Image.Image, target_size=(100, 100)) -> Tuple[np.ndarray, Image.Image]:
    """
    ОПЕРАТОР -> ЧЁРНОЕ по БЕЛОМУ, RGB target_size, [0..1].
    """
    gray = img_rgb.convert("L")
    bin_np = _binarize_grayscale(np.array(gray), pct=75)  # штрих=255, фон=0 (белый на чёрном в интенсивностях)

    # Для оператора нужно «чёрным по белому»:
    op_np = 255 - bin_np  # штрих=0 (чёрный), фон=255 (белый)
    op_img = Image.fromarray(op_np, mode="L").convert("RGB")

    img_resized = op_img.resize(target_size, Image.LANCZOS)
    x = np.array(img_resized).astype("float32") / 255.0
    x = x.reshape(1, target_size[0], target_size[1], 3)
    preview = img_resized
    return x, preview
