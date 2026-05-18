"""
Відтворює train/test-сплити з Face_ViT_SingleTask_and_YOLO.ipynb і викладає їх
у форматі ImageFolder, який очікує evaluate.py.

Датасети (ті самі, що в ноутбуці, через kagglehub):
  - Emotion (7 класів): dilkushsingh/facial-emotion-dataset
        У ньому вже є готові train_dir/ та test_dir/ з підпапками-класами.
  - Age (9 бінів)  + Gender (woman/man): jangedoo/utkface-new
        Файли мають імена `<age>_<gender>_..._.jpg`. Спліт у ноутбуці:
            train_test_split(test_size=0.2, random_state=42, stratify=<target>)
        Цей самий seed + stratify -> точно ті самі train/test, що бачила модель.

Вихід (за замовчуванням `--out-dir ./data`):

  data/
    emotion/
      train/<class>/*.jpg
      test/<class>/*.jpg
    age/
      train/<bin>/*.jpg
      test/<bin>/*.jpg
    gender/
      train/{woman,man}/*.jpg
      test/{woman,man}/*.jpg

За замовчуванням використовуються symlink-и (--mode symlink), щоб не дублювати ~3 GB зображень.
Для роботи на Windows-FS без прав на symlink-и передай `--mode copy`.

Приклади:
  python3 prepare_data.py --task all
  python3 prepare_data.py --task gender --split test --out-dir ./data
  python3 prepare_data.py --task emotion --mode copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError


# -------------------------------------------------
# Константи з ноутбука (один-в-один)
# -------------------------------------------------
EMO_CLASSES_7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

AGE_BINS_9: List[Tuple[int, int]] = [
    (0, 2), (3, 9), (10, 19), (20, 29), (30, 39),
    (40, 49), (50, 59), (60, 69), (70, 120),
]
# evaluate.py використовує "70+" як ім'я останнього класу — підтримуємо обидва варіанти.
AGE_NAMES_RANGE = [f"{a}-{b}" for a, b in AGE_BINS_9]
AGE_NAMES_FOLDER = AGE_NAMES_RANGE[:-1] + ["70+"]  # узгоджено з evaluate.py: AGE_CLASSES

# UTK у файлі: 0 = male, 1 = female. У ноутбуці зроблено перемапу -> 0 = woman, 1 = man.
UTK_GENDER_TO_ID = {0: 1, 1: 0}
GENDER_NAMES = ["woman", "man"]

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def age_to_bin(age: int) -> int:
    for i, (a0, a1) in enumerate(AGE_BINS_9):
        if a0 <= age <= a1:
            return i
    return len(AGE_BINS_9) - 1


def _is_decodable(path: str) -> bool:
    """PIL.verify() — швидко (читає лише заголовки) фільтрує биті JPEG'и
    (як `filter_bad_decode` у ноутбуці, але без повного декодування)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return False


def filter_broken(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    """Прибирає рядки, де image_path не декодується."""
    keep = []
    bad = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        if _is_decodable(row["image_path"]):
            keep.append(row)
        else:
            bad += 1
    print(f"  → битих/нечитабельних: {bad}, залишилось: {len(keep)}/{len(df)}")
    return pd.DataFrame(keep).reset_index(drop=True)


# -------------------------------------------------
# Завантаження датасетів
# -------------------------------------------------
def download_emotion() -> Path:
    import kagglehub
    print("[INFO] Завантаження dilkushsingh/facial-emotion-dataset ...")
    root = kagglehub.dataset_download("dilkushsingh/facial-emotion-dataset")
    print(f"[INFO] emo_root: {root}")
    return Path(root)


def download_utk() -> Path:
    import kagglehub
    print("[INFO] Завантаження jangedoo/utkface-new ...")
    root = kagglehub.dataset_download("jangedoo/utkface-new")
    print(f"[INFO] utk_root: {root}")
    return Path(root)


# -------------------------------------------------
# Збір DataFrame'ів — точно як у ноутбуці
# -------------------------------------------------
def collect_emotion_split(split_dir: Path) -> pd.DataFrame:
    """Eмо-датасет уже структурований як ImageFolder (train_dir/<class>/...)."""
    rows = []
    for cls_idx, cls in enumerate(EMO_CLASSES_7):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            print(f"[WARN] Відсутня папка класу: {cls_dir}")
            continue
        for dp, _, files in os.walk(cls_dir):
            for fn in files:
                if fn.lower().endswith(IMG_EXT):
                    rows.append({
                        "image_path": str(Path(dp) / fn),
                        "label": cls_idx,
                        "class_name": cls,
                    })
    return pd.DataFrame(rows)


def collect_utk(root: Path) -> pd.DataFrame:
    """UTKFace: ім'я файлу `<age>_<gender>_<race>_<datetime>.jpg`."""
    rows = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(IMG_EXT):
                continue
            base = fn.split(".")[0]
            parts = base.split("_")
            if len(parts) < 2:
                continue
            try:
                age = int(parts[0])
                g_raw = int(parts[1])
                if g_raw not in UTK_GENDER_TO_ID:
                    continue
                rows.append({
                    "image_path": str(Path(dp) / fn),
                    "age_bin": age_to_bin(age),
                    "gender": UTK_GENDER_TO_ID[g_raw],
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


# -------------------------------------------------
# Викладання на диск
# -------------------------------------------------
def place_file(src: Path, dst: Path, mode: str) -> None:
    """Покласти файл `src` у `dst`. mode: symlink | copy | hardlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:  # copy
        shutil.copy2(src, dst)


def dump_split(
    df: pd.DataFrame,
    out_split_dir: Path,
    class_names: List[str],
    label_col: str,
    mode: str,
    split_name: str,
) -> None:
    """Розкладає рядки df по підпапках out_split_dir/<class_name>/..."""
    for class_name in class_names:
        (out_split_dir / class_name).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in class_names}
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        src = Path(row["image_path"])
        cls_idx = int(row[label_col])
        if not (0 <= cls_idx < len(class_names)):
            continue
        class_name = class_names[cls_idx]
        # Унікальне ім'я: cls_idx-original-basename, щоб не було колізій
        dst_name = f"{cls_idx:02d}_{src.name}"
        place_file(src, out_split_dir / class_name / dst_name, mode)
        counts[class_name] += 1

    print(f"  → {split_name}: {sum(counts.values())} файлів | по класах: {counts}")


# -------------------------------------------------
# Обробка задач
# -------------------------------------------------
def prepare_emotion(out_dir: Path, mode: str, splits: List[str]) -> None:
    emo_root = download_emotion()
    task_dir = out_dir / "emotion"

    splits_map = {"train": emo_root / "train_dir", "test": emo_root / "test_dir"}
    for split in splits:
        src_split = splits_map[split]
        if not src_split.is_dir():
            print(f"[ERROR] Не знайдено: {src_split}")
            continue
        df = collect_emotion_split(src_split)
        print(f"[INFO] emotion {split}: {len(df)} зразків — перевірка цілісності ...")
        df = filter_broken(df, f"clean emotion {split}")
        dump_split(df, task_dir / split, EMO_CLASSES_7, "label", mode, f"emotion/{split}")


def prepare_utk(task: str, out_dir: Path, mode: str, splits: List[str]) -> None:
    """task: 'age' або 'gender'. Спліт відтворюється з seed=42 + stratify."""
    utk_root = download_utk()
    utk_df = collect_utk(utk_root)
    print(f"[INFO] UTK total: {len(utk_df)} — перевірка цілісності ...")
    utk_df = filter_broken(utk_df, "clean UTK")

    if task == "age":
        df = utk_df[["image_path", "age_bin"]].rename(columns={"age_bin": "label"})
        stratify = utk_df["age_bin"]
        class_names = AGE_NAMES_FOLDER
        task_dir = out_dir / "age"
    elif task == "gender":
        df = utk_df[["image_path", "gender"]].rename(columns={"gender": "label"})
        stratify = utk_df["gender"]
        class_names = GENDER_NAMES
        task_dir = out_dir / "gender"
    else:
        raise ValueError(task)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify
    )
    print(f"[INFO] {task} train/test: {len(train_df)}/{len(test_df)} (random_state=42)")

    if "train" in splits:
        dump_split(train_df, task_dir / "train", class_names, "label", mode, f"{task}/train")
    if "test" in splits:
        dump_split(test_df, task_dir / "test", class_names, "label", mode, f"{task}/test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="all",
                        choices=["emotion", "age", "gender", "all"],
                        help="Яку задачу готувати. За замовчуванням — all.")
    parser.add_argument("--split", default="both",
                        choices=["train", "test", "both"],
                        help="Які сплі­ти викладати. За замовчуванням — both.")
    parser.add_argument("--out-dir", default="./data", type=Path,
                        help="Куди класти ImageFolder-структури.")
    parser.add_argument("--mode", default="symlink",
                        choices=["symlink", "hardlink", "copy"],
                        help="Як класти файли. За замовчуванням symlink (економить ~3 GB).")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["emotion", "age", "gender"] if args.task == "all" else [args.task]

    for t in tasks:
        print("\n" + "=" * 60)
        print(f"  Готую {t} (splits={splits}, mode={args.mode})")
        print("=" * 60)
        if t == "emotion":
            prepare_emotion(args.out_dir, args.mode, splits)
        else:
            prepare_utk(t, args.out_dir, args.mode, splits)

    print("\n[OK] Готово. Структура у:", args.out_dir.resolve())
    print("\nПриклад запуску evaluate.py:")
    print(f"  python3 evaluate.py --task gender --data-dir {args.out_dir}/gender/test")
    print(f"  python3 evaluate.py --task age    --data-dir {args.out_dir}/age/test")
    print(f"  python3 evaluate.py --task emotion --data-dir {args.out_dir}/emotion/test")


if __name__ == "__main__":
    main()
