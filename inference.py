from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_digit, preprocess_operator

@dataclass
class SafeEvalResult:
    a: int
    op: str
    b: int
    value: str

@dataclass
class PredPack:
    digit_a: int
    digit_b: int
    operator: str
    prob_digit_a: float
    prob_digit_b: float
    operator_probs: Optional[List[float]]  # может быть None, если размер меток не совпал
    prev_a_img: "Image.Image"
    prev_op_img: "Image.Image"
    prev_b_img: "Image.Image"
    eval: SafeEvalResult

@tf.keras.utils.register_keras_serializable()
def load_model(path: str):
    return tf.keras.models.load_model(path)

@tf.keras.utils.register_keras_serializable()
def _cache():
    pass

@tf.keras.utils.register_keras_serializable()
def _noop():
    pass

@tf.keras.utils.register_keras_serializable()
def _identity(x): return x

# Кэш загрузки (streamlit)
_loaded = {}

def load_models_cached(digits_path: str, symbols_path: str):
    errors = []
    digits_model = None
    symbols_model = None
    try:
        if not os.path.exists(digits_path):
            raise FileNotFoundError(f"Не найден файл модели цифр: {digits_path}")
        if digits_path not in _loaded:
            _loaded[digits_path] = tf.keras.models.load_model(digits_path)
        digits_model = _loaded[digits_path]
    except Exception as e:
        errors.append(f"Ошибка загрузки модели цифр: {e}")

    try:
        if not os.path.exists(symbols_path):
            raise FileNotFoundError(f"Не найден файл модели операторов: {symbols_path}")
        if symbols_path not in _loaded:
            _loaded[symbols_path] = tf.keras.models.load_model(symbols_path)
        symbols_model = _loaded[symbols_path]
    except Exception as e:
        errors.append(f"Ошибка загрузки модели операторов: {e}")

    return digits_model, symbols_model, errors

def _safe_eval(a: int, op: str, b: int) -> SafeEvalResult:
    try:
        if op == "+":
            val = a + b
        elif op == "-":
            val = a - b
        elif op in ["*", "×", "x"]:
            val = a * b
        elif op in ["/", "÷"]:
            if b == 0:
                return SafeEvalResult(a, op, b, "деление на ноль")
            val = a / b
        else:
            return SafeEvalResult(a, op, b, f"неизвестный оператор: {op}")
        return SafeEvalResult(a, op, b, str(val))
    except Exception as e:
        return SafeEvalResult(a, op, b, f"ошибка вычисления: {e}")

def predict_all(img_a, img_op, img_b, digits_model, symbols_model, operator_labels: List[str]) -> PredPack:
    # подготовка
    x_a, prev_a = preprocess_digit(img_a)
    x_b, prev_b = preprocess_digit(img_b)
    x_op, prev_op = preprocess_operator(img_op, target_size=(100, 100))

    # предсказания цифр (допускаем выход [B,10] или [B,10] при (B,H,W))
    pred_a = digits_model.predict(x_a, verbose=0)[0]
    pred_b = digits_model.predict(x_b, verbose=0)[0]
    a_idx = int(np.argmax(pred_a))
    b_idx = int(np.argmax(pred_b))
    a_conf = float(np.max(pred_a))
    b_conf = float(np.max(pred_b))

    # предсказание оператора
    pred_op = symbols_model.predict(x_op, verbose=0)[0]
    op_idx = int(np.argmax(pred_op))
    if len(operator_labels) == len(pred_op):
        op_label = operator_labels[op_idx]
        op_probs = pred_op.tolist()
    else:
        op_label = f"class_{op_idx}"
        op_probs = None

    eval_res = _safe_eval(a_idx, op_label, b_idx)

    return PredPack(
        digit_a=a_idx,
        digit_b=b_idx,
        operator=op_label,
        prob_digit_a=a_conf,
        prob_digit_b=b_conf,
        operator_probs=op_probs,
        prev_a_img=prev_a,
        prev_op_img=prev_op,
        prev_b_img=prev_b,
        eval=eval_res
    )
