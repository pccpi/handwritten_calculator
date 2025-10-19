# app.py
# ------------------------------------------------------------
# Рукописный калькулятор (Streamlit), компактный 2-строчный layout:
#  ┌─────────────┬─────────────┬─────────────┬───────────────┐
#  │  Холст A    │  Холст Op   │  Холст B    │  Очистить     │  ← строка 1
#  ├─────────────┼─────────────┼─────────────┼───────────────┤
#  │  Распозн. A │  Распозн. Op│  Распозн. B │  Вычисление   │  ← строка 2
#  └─────────────┴─────────────┴─────────────┴───────────────┘
# - Белый фон, чёрное перо (толщина 15)
# - Распознавание всегда "на лету"
# - Топ-3 для цифр и оператора с процентами
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
OP_LABELS = DEFAULT_OPERATOR_LABELS  # например ['+', '/', '*', '-']

st.set_page_config(page_title="Рукописный калькулятор", layout="wide")

# Заголовок без эмодзи
st.markdown("<h3 style='margin:0'>Рукописный калькулятор</h3>", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------- Механизм очистки без experimental_rerun ----------
if "reset" not in st.session_state:
    st.session_state["reset"] = 0

def clear_all():
    # меняем ключи канвасов через счётчик — это гарантированно очищает их
    st.session_state["reset"] += 1
    # безопасный rerun (новые версии: st.rerun)
    try:
        st.rerun()
    except Exception:
        # на старых версиях можно было бы вызвать experimental_rerun,
        # но если его нет — rerun уже попытались
        pass

# ---------- Строка 1: три холста + кнопка "Очистить" ----------
cA, cOp, cB, cCtrl = st.columns([1, 1, 1, 0.5])

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
        key=f"canvas_a_{st.session_state['reset']}",
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
        key=f"canvas_op_{st.session_state['reset']}",
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
        key=f"canvas_b_{st.session_state['reset']}",
    )

with cCtrl:
    st.markdown("**Действия**")
    if st.button("Очистить всё", use_container_width=True):
        clear_all()

# ---------- Загрузка моделей ----------
digits_model, symbols_model, load_errors = load_models_cached(
    DIGITS_MODEL_PATH, SYMBOLS_MODEL_PATH
)
for err in load_errors:
    st.error(err)
ready = (digits_model is not None) and (symbols_model is not None)

# ---------- Инференс на лету ----------
pred = None
top3_digits_a = top3_digits_b = top3_ops = None

something_drawn = any([
    canvas_a.image_data is not None,
    canvas_op.image_data is not None,
    canvas_b.image_data is not None
])

if ready and something_drawn:
    img_a = pil_from_canvas(canvas_a.image_data)
    img_op = pil_from_canvas(canvas_op.image_data)
    img_b = pil_from_canvas(canvas_b.image_data)

    if has_ink(img_a) and has_ink(img_op) and has_ink(img_b):
        # Основной пайплайн (даёт предикты и предпросмотры)
        pred = predict_all(
            img_a=img_a,
            img_op=img_op,
            img_b=img_b,
            digits_model=digits_model,
            symbols_model=symbols_model,
            operator_labels=OP_LABELS
        )
        # Вычислим распределения для топ-3 отдельно (чтобы показать проценты)
        xa, _ = preprocess_digit(img_a)
        xb, _ = preprocess_digit(img_b)
        xop, _ = preprocess_operator(img_op, target_size=(100, 100))

        pa = digits_model.predict(xa, verbose=0)[0]
        pb = digits_model.predict(xb, verbose=0)[0]
        po = symbols_model.predict(xop, verbose=0)[0]

        ia = pa.argsort()[-3:][::-1]
        ib = pb.argsort()[-3:][::-1]
        io = po.argsort()[-3:][::-1]

        top3_digits_a = [(int(i), float(pa[i])) for i in ia]
        top3_digits_b = [(int(i), float(pb[i])) for i in ib]
        top3_ops = [(OP_LABELS[i] if i < len(OP_LABELS) else f"class_{int(i)}",
                     float(po[i])) for i in io]

# Разделитель между строками
st.markdown("<div style='height:10px; border-top:1px solid #e6e6e6;'></div>", unsafe_allow_html=True)

# ---------- Строка 2: распознавание под каждым холстом + вычисление справа ----------
rA, rOp, rB, rCalc = st.columns([1, 1, 1, 0.5])

with rA:
    st.markdown("**Распознавание A**")
    if not ready:
        st.write("Нет моделей.")
    elif pred is None:
        st.write("—")
    else:
        st.image(pred.prev_a_img.resize((96, 96)))
        st.write(f"Ответ: {pred.digit_a}  ({pred.prob_digit_a * 100:.1f}%)")
        if top3_digits_a:
            st.write("Топ-3:")
            for cls, p in top3_digits_a:
                st.write(f"{cls} — {p * 100:.1f}%")

with rOp:
    st.markdown("**Распознавание оператора**")
    if not ready:
        st.write("Нет моделей.")
    elif pred is None:
        st.write("—")
    else:
        st.image(pred.prev_op_img.resize((96, 96)))
        st.write(f"Ответ: {pred.operator}")
        if top3_ops:
            st.write("Топ-3:")
            for sym, p in top3_ops:
                st.write(f"{sym} — {p * 100:.1f}%")

with rB:
    st.markdown("**Распознавание B**")
    if not ready:
        st.write("Нет моделей.")
    elif pred is None:
        st.write("—")
    else:
        st.image(pred.prev_b_img.resize((96, 96)))
        st.write(f"Ответ: {pred.digit_b}  ({pred.prob_digit_b * 100:.1f}%)")
        if top3_digits_b:
            st.write("Топ-3:")
            for cls, p in top3_digits_b:
                st.write(f"{cls} — {p * 100:.1f}%")

with rCalc:
    st.markdown("**Вычисление**")
    if pred is not None:
        r = pred.eval
        st.markdown(
            f"<div style='font-size:26px; font-weight:600; padding:6px 0'>"
            f"{r.a} {r.op} {r.b} = {r.value}"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.write("—")
