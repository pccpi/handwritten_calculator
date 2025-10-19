# app.py
# ------------------------------------------------------------
# Рукописный калькулятор (Streamlit) — компактный интерфейс
# - Три канваса: белый фон, чёрное перо (толщина 15)
# - Распознавание всегда "на лету", без кнопки "Рассчитать"
# - Кнопка "Очистить всё" узкая, справа от холстов
# - Под холстами: слева распознавание (топ-3 для цифр и оператора, %),
#                 справа вычисление
# - Без сайдбара и без эмодзи/подсказок
# ------------------------------------------------------------

import streamlit as st
import numpy as np

from streamlit_drawable_canvas import st_canvas

from preprocessing import pil_from_canvas, has_ink, preprocess_digit, preprocess_operator
from inference import load_models_cached, predict_all
from labels import DEFAULT_OPERATOR_LABELS

# ---------- Константы интерфейса ----------
CANVAS_W = 240
CANVAS_H = 240
STROKE_WIDTH = 15
BG_COLOR = "#FFFFFF"
STROKE_COLOR = "#000000"

DIGITS_MODEL_PATH = "digits_model.keras"
SYMBOLS_MODEL_PATH = "symbols_cnn_model.keras"
OP_LABELS = DEFAULT_OPERATOR_LABELS  # например: ['+', '/', '*', '-']

st.set_page_config(page_title="Рукописный калькулятор", layout="wide")

# Заголовок компактно
st.markdown("<h3 style='margin:0'>Рукописный калькулятор</h3>", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------- Ряд с тремя канвасами + узкая кнопка справа ----------
cA, cOp, cB, cBtn = st.columns([1, 1, 1, 0.25])

with cA:
    st.markdown("**Число A**")
    canvas_a = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W,
        height=CANVAS_H,
        drawing_mode="freedraw",
        key="canvas_a",
    )
with cOp:
    st.markdown("**Оператор**")
    canvas_op = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W,
        height=CANVAS_H,
        drawing_mode="freedraw",
        key="canvas_op",
    )
with cB:
    st.markdown("**Число B**")
    canvas_b = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        width=CANVAS_W,
        height=CANVAS_H,
        drawing_mode="freedraw",
        key="canvas_b",
    )
with cBtn:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    clear_clicked = st.button("Очистить всё", use_container_width=True)

if clear_clicked:
    st.experimental_rerun()

# ---------- Загрузка моделей ----------
digits_model, symbols_model, load_errors = load_models_cached(
    DIGITS_MODEL_PATH, SYMBOLS_MODEL_PATH
)
for err in load_errors:
    st.error(err)

ready = (digits_model is not None) and (symbols_model is not None)

# ---------- Инференс "на лету" ----------
something_drawn = any([
    canvas_a.image_data is not None,
    canvas_op.image_data is not None,
    canvas_b.image_data is not None
])

pred = None
top3_digits_a = None
top3_digits_b = None
top3_ops = None

if ready and something_drawn:
    img_a = pil_from_canvas(canvas_a.image_data)
    img_op = pil_from_canvas(canvas_op.image_data)
    img_b = pil_from_canvas(canvas_b.image_data)

    ink_a = has_ink(img_a)
    ink_op = has_ink(img_op)
    ink_b = has_ink(img_b)

    if ink_a and ink_op and ink_b:
        # Основной пайплайн (получим best классы и предпросмотры)
        pred = predict_all(
            img_a=img_a,
            img_op=img_op,
            img_b=img_b,
            digits_model=digits_model,
            symbols_model=symbols_model,
            operator_labels=OP_LABELS
        )

        # Дополнительно посчитаем топ-3 вероятности вручную здесь,
        # чтобы точно иметь распределения и для цифр, и для оператора.
        xa, _ = preprocess_digit(img_a)
        xb, _ = preprocess_digit(img_b)
        xop, _ = preprocess_operator(img_op, target_size=(100, 100))

        probs_a = digits_model.predict(xa, verbose=0)[0]
        probs_b = digits_model.predict(xb, verbose=0)[0]
        probs_op = symbols_model.predict(xop, verbose=0)[0]

        idxs_a = probs_a.argsort()[-3:][::-1]
        idxs_b = probs_b.argsort()[-3:][::-1]
        idxs_op = probs_op.argsort()[-3:][::-1]

        top3_digits_a = [(int(i), float(probs_a[i])) for i in idxs_a]
        top3_digits_b = [(int(i), float(probs_b[i])) for i in idxs_b]
        top3_ops = [(OP_LABELS[i] if i < len(OP_LABELS) else f"class_{int(i)}",
                     float(probs_op[i])) for i in idxs_op]

# ---------- Блок под холстами: слева распознавание, справа вычисление ----------
st.markdown("<div style='height:10px; border-top:1px solid #e6e6e6;'></div>", unsafe_allow_html=True)
left, right = st.columns([1.6, 1])

with left:
    st.markdown("**Распознавание**")
    if not ready:
        st.write("Добавьте в каталог приложения файлы моделей: "
                 "`digits_model.keras`, `symbols_cnn_model.keras`.")
    elif pred is None:
        st.write("Нарисуйте число A, оператор и число B.")
    else:
        conf_a = f"{pred.prob_digit_a * 100:.1f}%"
        conf_b = f"{pred.prob_digit_b * 100:.1f}%"

        gA, gOp, gB = st.columns([1, 1, 1])

        with gA:
            st.image(pred.prev_a_img.resize((96, 96)), caption=f"A = {pred.digit_a}   ({conf_a})")
            if top3_digits_a is not None:
                st.write("Топ-3 по A:")
                for cls, p in top3_digits_a:
                    st.write(f"{cls} — {p*100:.1f}%")

        with gOp:
            st.image(pred.prev_op_img.resize((96, 96)), caption=f"Оператор = {pred.operator}")
            if top3_ops is not None:
                st.write("Топ-3 по оператору:")
                for sym, p in top3_ops:
                    st.write(f"{sym} — {p*100:.1f}%")

        with gB:
            st.image(pred.prev_b_img.resize((96, 96)), caption=f"B = {pred.digit_b}   ({conf_b})")
            if top3_digits_b is not None:
                st.write("Топ-3 по B:")
                for cls, p in top3_digits_b:
                    st.write(f"{cls} — {p*100:.1f}%")

with right:
    st.markdown("**Вычисление**")
    if pred is not None:
        r = pred.eval
        st.markdown(
            f"<div style='font-size:26px; font-weight:600; "
            f"padding:6px 0'>{r.a} {r.op} {r.b} = {r.value}</div>",
            unsafe_allow_html=True
        )
    else:
        st.write("—")
