# -*- coding: utf-8 -*-
import numpy as np
import cv2


def predict_type_of_road_markings(image: np.ndarray) -> str:
    """
    Функция, определяющая вид дорожной разметки
    Входные данные: BGR изображение, прочитанное
    Выходные данные: строка с названием вида разметки

    Формат вывода: текстовая строка, принимающая одно из четырёх значений: "first", "second", "third", "fourth".

    Примеры вывода: "first"
                    "second"
                    "third"
                    "fourth"
    """
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация для выделения белых линий (подходит для чёрного фона)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Измерение штрихов и промежутков
    dash_lengths = []
    gaps = []
    if len(contours) >= 2:
        # Сортировка контуров по координате x для корректного измерения промежутков
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # Длина штриха как максимальная сторона прямоугольника
            dash_length = max(w, h)
            dash_lengths.append(dash_length)

            # Измерение промежутков между соседними штрихами
            if i < len(contours) - 1:
                x_next, y_next, w_next, h_next = cv2.boundingRect(contours[i + 1])
                gap = min(abs(x_next - (x + w)), abs(y_next - (y + h)))
                gaps.append(gap)

    # Средние значения
    avg_dash_length = np.mean(dash_lengths) if dash_lengths else 0
    avg_gap = np.mean(gaps) if gaps else 0

    # Классификация на основе измеренных параметров (на основе анализа изображений)
    # Тип 1: короткие штрихи (~20 px), промежуток ~50 px
    # Тип 2: короткие штрихи (~20 px), промежуток ~70 px
    # Тип 3: длинные штрихи (~40 px), промежуток ~30 px
    # Тип 4: короткие штрихи (~20 px), промежуток ~100 px
    if 15 < avg_dash_length <= 25 and 40 < avg_gap <= 60:
        return "first"
    elif 15 < avg_dash_length <= 25 and 60 < avg_gap <= 80:
        return "second"
    elif 25 < avg_dash_length <= 45 and avg_gap <= 40:
        return "third"
    elif 15 < avg_dash_length <= 25 and avg_gap > 80:
        return "fourth"
    else:
        return "first"  # Фallback на случай некорректных данных

    return "first"  # Фallback (должен быть недостижим)