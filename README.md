# SCGVAC

**S**ociocultural **C**haracteristics of **G**roups via **V**ideo-recorded **A**ctions of **C**rowds — інтелектуальна система нейромережевого визначення психологічних та соціокультурних характеристик угрупувань (вік, стать, емоції) за відеофіксацією дій натовпу. Базується на детекторі облич **YOLO** (Ultralytics) та трьох незалежно натренованих моделях класифікації **Vision Transformer (ViT)**. Має графічний інтерфейс на базі **PyQt6**.

## Функціональні можливості

- Сучасний графічний інтерфейс на базі **PyQt6**.
- Три джерела вхідних даних: жива камера, відеофайл (.mp4 / .avi / .mov / .mkv / .gif), статичне фото (.jpg / .png / .webp).
- Автоматичне детектування облич у кадрі моделлю **YOLO** (Ultralytics) з розширенням рамки для збереження контексту.
- Класифікація **емоцій** на 7 класів: *angry, disgust, fear, happy, neutral, sad, surprise*.
- Класифікація **віку** на 9 діапазонів: *0–2, 3–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60–69, 70+*.
- Класифікація **статі**: *woman / man*.
- Гнучке керування параметрами у реальному часі: мінімальний розмір обличчя, максимальна кількість облич у кадрі, ширина оброблюваного кадру.
- Автоматичне обчислення метрик якості (Accuracy, Precision, Recall, F1-score, ROC-AUC) на тренувальних та тестових вибірках.
- Генерація візуальних звітів: матриця помилок (звичайна та нормалізована), ROC-крива (one-vs-rest з macro-середнім), стовпчаста діаграма метрик по класах.

## Архітектура

Конвеєр складається з двох етапів:

1. **Детектор облич** — модель **YOLO** (Ultralytics), яка локалізує обличчя у кадрі та повертає bounding-box'и.
2. **Класифікація** — три окремі моделі **ViT-base-patch16-224** з простим лінійним класифікатором поверх `[CLS]`-токену, натреновані на окремих датасетах:

| Задача | Класів | Датасет | Файл ваг |
|---|---|---|---|
| Face detection | — | `aklimarimi/8-facial-expressions-for-yolo` | `yolo26n.pt` |
| Emotion | 7 | `dilkushsingh/facial-emotion-dataset` | `best.pt` |
| Age | 9 | `jangedoo/utkface-new` | `best1.pt` |
| Gender | 2 | `jangedoo/utkface-new` | `best (2).pt` |

Файли ваг ViT (~340 МБ кожен) та YOLO (`yolo26n.pt`) не зберігаються у репозиторії. ViT-ваги треба натренувати локально через ноутбук `Face_ViT_SingleTask_and_YOLO.ipynb`; YOLO-ваги автоматично завантажуються бібліотекою Ultralytics при першому запуску `main.py`.

## Структура репозиторію

```
.
├── main.py                                # PyQt6 GUI: камера / відео / фото
├── evaluate.py                            # Оцінювання моделей з метриками і графіками
├── prepare_data.py                        # Підготовка train/test з Kaggle
├── Face_ViT_SingleTask_and_YOLO.ipynb     # Ноутбук тренування
├── report.py                              # Допоміжний скрипт звітування
├── requirements.txt                       # Python-залежності
├── run.sh / run.bat                       # Запуск повного циклу
└── README.md
```

## Технологічний стек

- **Python 3.10+**
- **Глибоке навчання**: PyTorch, HuggingFace Transformers (Vision Transformer), Ultralytics (YOLO для детекції облич)
- **Комп'ютерний зір**: OpenCV
- **Графічний інтерфейс**: PyQt6
- **Метрики та візуалізація**: scikit-learn, Matplotlib, Seaborn
- **Обробка даних**: Pandas, NumPy, Pillow, tqdm, kagglehub

## Швидкий старт

### 1. Клонувати репозиторій

```bash
git clone https://github.com/AlexFRUZ/Diplom.git
cd Diplom
```

### 2. Встановити залежності

```bash
python3 -m venv venv
source venv/bin/activate           # Linux/macOS
# або: venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Отримати натреновані ваги

**ViT-ваги** (3 файли): натренувати власні, прогнавши `Face_ViT_SingleTask_and_YOLO.ipynb` (потрібен Kaggle API ключ — `~/.kaggle/kaggle.json`), або помістити готові `.pt`-файли у корінь проєкту:
- `best.pt` (emotion, 7 виходів)
- `best1.pt` (age, 9 виходів)
- `best (2).pt` (gender, 2 виходи)

Скрипти `main.py` та `evaluate.py` автоматично визначають, який файл відповідає якій задачі за розміром класифікаційної голови — імена файлів не критичні.

**YOLO-ваги** (`yolo26n.pt`): завантажуються автоматично бібліотекою Ultralytics при першому запуску `main.py`. Кешуються у домашній директорії користувача.

### 4. Запустити графічний інтерфейс

```bash
python3 main.py
```

У вікні SCGVAC натиснути **Load models**, обрати джерело — **Камера**, **Відео** або **Фото** — і виконати розпізнавання характеристик облич у кадрі.

## Оцінювання моделей

### Підготувати тестові датасети (відтворює сплі­ти з ноутбука, `random_state=42`)

```bash
python3 prepare_data.py --task all --split both
```

Створить `data/{emotion,age,gender}/{train,test}/<class>/...` через символьні посилання (не дублюючи ~3 ГБ зображень).

### Запустити оцінювання

```bash
# Все одразу (emotion + age + gender × train + test):
python3 evaluate.py

# Або окремо:
python3 evaluate.py --task gender --split test
```

На виході — у поточну теку:
- `confusion_matrix_<task>_<split>.png` (+ `_normalized`)
- `roc_curve_<task>_<split>.png`
- `per_class_metrics_<task>_<split>.png`
- `metrics_<task>_<split>.json`
- `summary.json` — зведення всіх запусків

### Або один скрипт на все

```bash
./run.sh           # підготовка даних + оцінювання
./run.sh cam       # + запуск GUI з вебкамерою
./run.sh eval      # лише оцінювання
./run.sh prep      # лише підготовка
```

## Аргументи командного рядка

### `main.py`

```bash
python3 main.py [OPTIONS]
```

| Параметр | Опис |
|---|---|
| `--emotion-weights PATH` | Шлях до ваг моделі емоцій |
| `--age-weights PATH` | Шлях до ваг моделі віку |
| `--gender-weights PATH` | Шлях до ваг моделі статі |
| `--backbone NAME` | HuggingFace ID бекбону (за замовч. `google/vit-base-patch16-224-in21k`) |
| `--camera N` | Індекс камери (за замовч. 0) |
| `--width N` | Ширина кадру для прискорення |
| `--min-face N` | Мінімальний розмір обличчя для детекції |
| `--max-faces N` | Максимум облич у кадрі |

### `evaluate.py`

```bash
python3 evaluate.py [OPTIONS]
```

| Параметр | Опис |
|---|---|
| `--task {emotion,age,gender,all}` | Яку задачу оцінювати (за замовч. `all`) |
| `--split {train,test,both}` | Який спліт (за замовч. `both`) |
| `--data-root PATH` | Корінь даних (за замовч. `./data`) |
| `--weights PATH` | Явні ваги (інакше автодетект) |
| `--batch-size N` | Розмір батчу (за замовч. 32) |
| `--out-dir PATH` | Куди класти графіки/JSON |

