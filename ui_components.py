from dataclasses import dataclass
from typing import List
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

@dataclass
class Settings:
    bg_color: str
    stroke_color: str
    stroke_width: int
    realtime: bool
    digits_model_path: str
    symbols_model_path: str
    operator_labels: List[str]

def topbar():
    st.title("🖊️ Рукописный калькулятор")

def sidebar_settings(default_operator_labels: str) -> Settings:
    with st.sidebar:
        st.markdown("### Настройки")
        bg = st.color_picker("Цвет фона всех канвасов", value="#FFFFFF")
        stroke = st.color_picker("Цвет пера", value="#000000")
        width = st.slider("Толщина пера", 8, 40, 22)
        realtime = st.toggle("Распознавать на лету", value=True)

        st.markdown("---")
        st.markdown("**Пути к моделям**")
        digits_path = st.text_input("Digits model (.keras)", "digits_model.keras")
        symbols_path = st.text_input("Symbols model (.keras)", "symbols_cnn_model.keras")

        st.markdown("---")
        labels_str = st.text_input("Метки операторов (по индексам)", default_operator_labels)
        labels = [s.strip() for s in labels_str.split(",") if s.strip()]

    return Settings(
        bg_color=bg,
        stroke_color=stroke,
        stroke_width=width,
        realtime=realtime,
        digits_model_path=digits_path,
        symbols_model_path=symbols_path,
        operator_labels=labels
    )

def _canvas(title: str, key: str, bg_color: str, stroke_color: str, stroke_width: int):
    st.subheader(title)
    return st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=key,
    )

def three_canvases(bg_color: str, stroke_color: str, stroke_width: int):
    col_a, col_op, col_b = st.columns(3)
    with col_a:
        c_a = _canvas("Число A", "canvas_a", bg_color, stroke_color, stroke_width)
    with col_op:
        c_op = _canvas("Оператор", "canvas_op", bg_color, stroke_color, stroke_width)
    with col_b:
        c_b = _canvas("Число B", "canvas_b", bg_color, stroke_color, stroke_width)
    return c_a, c_op, c_b

def results_block(pred_pack):
    import numpy as np
    st.markdown("---")
    st.subheader("Результаты распознавания")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(pred_pack.prev_a_img, caption=f"A (белое по чёрному) → {pred_pack.digit_a}")
        st.progress(pred_pack.prob_digit_a)
        st.caption("уверенность модели")
    with c2:
        st.image(pred_pack.prev_op_img, caption=f"Оператор (чёрное по белому) → {pred_pack.operator}")
        if pred_pack.operator_probs is not None:
            st.caption("Топ по операторам:")
            # покажем top-3
            probs = np.array(pred_pack.operator_probs)
            top3 = probs.argsort()[-3:][::-1]
            for i in top3:
                st.write(f"- idx {i}: {float(probs[i]):.3f}")
    with c3:
        st.image(pred_pack.prev_b_img, caption=f"B (белое по чёрному) → {pred_pack.digit_b}")
        st.progress(pred_pack.prob_digit_b)
        st.caption("уверенность модели")

    st.markdown("---")
    st.subheader("Вычисление")
    r = pred_pack.eval
    st.markdown(f"### ➜ {r.a} {r.op} {r.b} = **{r.value}**")

def help_block():
    with st.expander("Справка / советы"):
        st.markdown(
            """
- Все три канваса **одного цвета фона**, правильная полярность достигается *предобработкой*.
- Цифры → **белое по чёрному** (28×28, 1ch), оператор → **чёрное по белому** (100×100, 3ch).
- Если у модели операторов другой порядок классов — поменяй строку меток в сайдбаре.
- Файлы `digits_model.keras` и `symbols_cnn_model.keras` помести рядом с приложением или укажи путь.
- Кнопка «Очистить всё» перезапускает приложение и сбрасывает канвасы.
"""
        )
