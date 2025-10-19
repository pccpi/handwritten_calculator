# app.py
# --------------------------------------------
# Streamlit-приложение: рукописный калькулятор
# Три канваса одного цвета.
# Цифры → БЕЛОЕ по ЧЁРНОМУ (MNIST-стиль, 28x28)
# Оператор → ЧЁРНОЕ по БЕЛОМУ (RGB, 100x100)
# Пустые холсты не распознаются.
# --------------------------------------------

import streamlit as st
from ui_components import three_canvases, sidebar_settings, results_block, help_block, topbar
from inference import load_models_cached, predict_all
from preprocessing import pil_from_canvas, has_ink
from labels import DEFAULT_OPERATOR_LABELS

st.set_page_config(page_title="Рукописный калькулятор", page_icon="🖊️", layout="wide")
topbar()

# --- Sidebar / settings (дефолтный порядок меток операторов берём из labels.py)
settings = sidebar_settings(default_operator_labels=",".join(DEFAULT_OPERATOR_LABELS))

# --- Три канваса (одинаковые цвета фона/пера)
c_a, c_op, c_b = three_canvases(
    bg_color=settings.bg_color,
    stroke_color=settings.stroke_color,
    stroke_width=settings.stroke_width
)

# --- Загрузка моделей (кэшируется в процессе работы)
digits_model, symbols_model, load_errors = load_models_cached(
    settings.digits_model_path, settings.symbols_model_path
)
for err in load_errors:
    st.error(err)

# --- Кнопки
col1, col2 = st.columns(2)
with col1:
    do_predict = st.button("Рассчитать", use_container_width=True)
with col2:
    if st.button("Очистить всё", use_container_width=True):
        st.experimental_rerun()

ready = (digits_model is not None) and (symbols_model is not None)

# --- Логика запуска инференса
# - Либо нажали кнопку,
# - Либо включён "Распознавать на лету" и что-то нарисовано.
something_drawn = any([
    c_a.image_data is not None,
    c_op.image_data is not None,
    c_b.image_data is not None
])
should_run = ready and (do_predict or (settings.realtime and something_drawn))

if should_run:
    # Достаём изображения из канвасов
    img_a = pil_from_canvas(c_a.image_data)
    img_op = pil_from_canvas(c_op.image_data)
    img_b = pil_from_canvas(c_b.image_data)

    # Фильтр пустых холстов (иначе модель "угадывает" любимые классы вроде 6/9/5/0)
    ink_a = has_ink(img_a)
    ink_op = has_ink(img_op)
    ink_b = has_ink(img_b)

    if not (ink_a and ink_op and ink_b):
        missing = []
        if not ink_a: missing.append("A")
        if not ink_op: missing.append("оператор")
        if not ink_b: missing.append("B")
        st.info("Нарисуйте: " + ", ".join(missing) + ". Пустые поля не распознаются.")
    else:
        # Предсказания + вычисление
        pred = predict_all(
            img_a=img_a,
            img_op=img_op,
            img_b=img_b,
            digits_model=digits_model,
            symbols_model=symbols_model,
            operator_labels=settings.operator_labels  # например ['+', '/', '*', '-']
        )
        results_block(pred)

else:
    if ready:
        st.info("Нарисуйте число A, оператор и число B. Нажмите «Рассчитать» или включите ‘Распознавать на лету’.")
    else:
        st.warning("Загрузите модели в сайдбаре или положите файлы рядом с приложением.")

# --- Справка
help_block()