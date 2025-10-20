# app.py
# ------------------------------------------------------------
# Рукописный калькулятор (финальная версия)
# ------------------------------------------------------------

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas

from preprocessing import (
    pil_from_canvas, has_ink, preprocess_digit, preprocess_operator
)
from inference import load_models_cached
from labels import DEFAULT_OPERATOR_LABELS

# ---------- Константы интерфейса ----------
CANVAS_W = 200
CANVAS_H = 200
STROKE_WIDTH = 15
BG_COLOR = "#FFFFFF"
STROKE_COLOR = "#000000"

DIGITS_MODEL_PATH = "digits_model.keras"
SYMBOLS_MODEL_PATH = "symbols_cnn_model.keras"
OP_LABELS = DEFAULT_OPERATOR_LABELS  # например ['+', '/', '*', '-']

st.set_page_config(page_title="Рукописный калькулятор", layout="wide")

# Заголовок
st.markdown("<h3 style='margin:0'>Рукописный калькулятор</h3>", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------- Очистка ----------
if "reset" not in st.session_state:
    st.session_state["reset"] = 0

def clear_all():
    st.session_state["reset"] += 1
    try:
        st.rerun()
    except Exception:
        pass

# ---------- Строка 1: холсты ----------
cA1, cA2, cOp, cB1, cB2, cCtrl = st.columns([1, 1, 1, 1, 1, 1.2])

with cA1:
    st.markdown("**Цифра A1**")
    canvas_a1 = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W, height=CANVAS_H,
        drawing_mode="freedraw",
        key=f"canvas_a1_{st.session_state['reset']}",
        update_streamlit=True,
    )

with cA2:
    st.markdown("**Цифра A2**")
    canvas_a2 = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W, height=CANVAS_H,
        drawing_mode="freedraw",
        key=f"canvas_a2_{st.session_state['reset']}",
        update_streamlit=True,
    )

with cOp:
    st.markdown("**Оператор**")
    canvas_op = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W, height=CANVAS_H,
        drawing_mode="freedraw",
        key=f"canvas_op_{st.session_state['reset']}",
        update_streamlit=True,
    )

with cB1:
    st.markdown("**Цифра B1**")
    canvas_b1 = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W, height=CANVAS_H,
        drawing_mode="freedraw",
        key=f"canvas_b1_{st.session_state['reset']}",
        update_streamlit=True,
    )

with cB2:
    st.markdown("**Цифра B2**")
    canvas_b2 = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W, height=CANVAS_H,
        drawing_mode="freedraw",
        key=f"canvas_b2_{st.session_state['reset']}",
        update_streamlit=True,
    )

with cCtrl:
    # без заголовка, только кнопка
    st.markdown("&nbsp;", unsafe_allow_html=True)
    clear_clicked = st.button("Очистить всё", use_container_width=True)

# ---------- Загрузка моделей ----------
digits_model, symbols_model, load_errors = load_models_cached(
    DIGITS_MODEL_PATH, SYMBOLS_MODEL_PATH
)
for err in load_errors:
    st.error(err)
ready = (digits_model is not None) and (symbols_model is not None)

# ---------- Инференс ----------
pred_info = {"a1": None, "a2": None, "b1": None, "b2": None, "op": None}
A_value = None
B_value = None
op_label = None
calc_result = None

def _predict_digit(img):
    x, prev = preprocess_digit(img)
    probs = digits_model.predict(x, verbose=0)[0]
    idxs = probs.argsort()[-3:][::-1]
    return {
        "top1": int(idxs[0]),
        "conf": float(probs[idxs[0]]),
        "top3": [(int(i), float(probs[i])) for i in idxs],
        "prev": prev
    }

def _predict_operator(img):
    x, prev = preprocess_operator(img, target_size=(100, 100))
    probs = symbols_model.predict(x, verbose=0)[0]
    idxs = probs.argsort()[-3:][::-1]
    def lab(i):
        return OP_LABELS[i] if i < len(OP_LABELS) else f"class_{int(i)}"
    return {
        "top1": lab(int(idxs[0])),
        "conf": float(probs[idxs[0]]),
        "top3": [(lab(int(i)), float(probs[i])) for i in idxs],
        "prev": prev
    }

something_drawn = any(
    img is not None and getattr(img, "size", 0) != 0
    for img in [
        canvas_a1.image_data,
        canvas_a2.image_data,
        canvas_op.image_data,
        canvas_b1.image_data,
        canvas_b2.image_data,
    ]
)



if ready and something_drawn:
    imgs = {
        "a1": pil_from_canvas(canvas_a1.image_data) if canvas_a1.image_data is not None else None,
        "a2": pil_from_canvas(canvas_a2.image_data) if canvas_a2.image_data is not None else None,
        "op": pil_from_canvas(canvas_op.image_data) if canvas_op.image_data is not None else None,
        "b1": pil_from_canvas(canvas_b1.image_data) if canvas_b1.image_data is not None else None,
        "b2": pil_from_canvas(canvas_b2.image_data) if canvas_b2.image_data is not None else None,
    }

    # предсказания
    for key in imgs:
        if imgs[key] is not None and has_ink(imgs[key]):
            if key == "op":
                pred_info[key] = _predict_operator(imgs[key])
            else:
                pred_info[key] = _predict_digit(imgs[key])

    # собрать числа
    def compose(a1, a2):
        if a1 and a2:
            return a1["top1"] * 10 + a2["top1"]
        if a1: return a1["top1"]
        if a2: return a2["top1"]
        return None

    A_value = compose(pred_info["a1"], pred_info["a2"])
    B_value = compose(pred_info["b1"], pred_info["b2"])
    op_label = pred_info["op"]["top1"] if pred_info["op"] else None

    def safe_eval(a, op, b):
        if a is None or b is None or op is None:
            return None
        try:
            if op == "+": return a + b
            if op == "-": return a - b
            if op in ["*", "×", "x"]: return a * b
            if op in ["/", "÷"]: return "деление на ноль" if b == 0 else a / b
            return f"неизвестный оператор: {op}"
        except Exception as e:
            return f"ошибка: {e}"

    calc_result = safe_eval(A_value, op_label, B_value)

# разделитель
st.markdown("<div style='height:10px; border-top:1px solid #e6e6e6;'></div>", unsafe_allow_html=True)

# ---------- Строка 2: распознавание под каждым холстом ----------
rA1, rA2, rOp, rB1, rB2, rCalc = st.columns([1, 1, 1, 1, 1, 1.2])

def render_digit_block(slot, title):
    st.markdown(f"**{title}**")
    info = pred_info[slot]
    if not ready:
        st.write("Нет моделей.")
        return
    if not info:
        st.write("—")
        return
    st.image(info["prev"].resize((84, 84)))
    st.write(f"Ответ: {info['top1']} ({info['conf'] * 100:.1f}%)")
    st.write("Топ-3:")
    for cls, p in info["top3"]:
        st.text(f"{cls} — {p * 100:.1f}%")

with rA1:
    render_digit_block("a1", "Распознавание A1")
with rA2:
    render_digit_block("a2", "Распознавание A2")

with rOp:
    st.markdown("**Распознавание оператора**")
    if not ready:
        st.write("Нет моделей.")
    elif not pred_info["op"]:
        st.write("—")
    else:
        info = pred_info["op"]
        st.image(info["prev"].resize((84, 84)))
        st.write(f"Ответ: `{info['top1']}`")
        st.write("Топ-3:")
        for sym, p in info["top3"]:
            st.text(f"{sym} — {p * 100:.1f}%")

with rB1:
    render_digit_block("b1", "Распознавание B1")
with rB2:
    render_digit_block("b2", "Распознавание B2")

with rCalc:
    if clear_clicked:
        clear_all()
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    # увеличил размер результата
    if not ready:
        st.write("—")
    elif calc_result is None:
        st.write("Нарисуйте оба числа и оператор.")
    else:
        st.markdown(
            f"<div style='font-size:34px; font-weight:700; padding:10px 0;'>"
            f"{A_value} {op_label} {B_value} = {calc_result}"
            f"</div>",
            unsafe_allow_html=True
        )
