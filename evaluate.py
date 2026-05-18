"""
Оцінювання натренованих ViT-моделей: classification report, confusion matrix, ROC-curve.

Швидкі запуски:
    # Все одразу (emotion+age+gender, train+test):
    python3 evaluate.py
    # Тільки тест для всіх задач:
    python3 evaluate.py --split test
    # Тільки одна задача, обидва спліти:
    python3 evaluate.py --task gender --split both
    # Стара поведінка з явним --data-dir:
    python3 evaluate.py --task emotion --data-dir ./data/emotion/test

За замовчуванням data-шляхи беруться як <data-root>/<task>/<split>
  (--data-root за замовчуванням ./data — це структура, яку створює prepare_data.py).

Виведе у консоль метрики для кожної (задача, спліт) комбінації.
Збереже в `--out-dir` (за замовч. .):
    - confusion_matrix_<task>_<split>.png  (+ _normalized)
    - roc_curve_<task>_<split>.png
    - per_class_metrics_<task>_<split>.png
    - metrics_<task>_<split>.json
    - summary.json — зведена таблиця всіх запусків
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# -------------------------
# Labels (мають збігатися з тренувальними)
# -------------------------
EMO_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
# Порядок має збігатися з тренуванням: GENDER_NAMES = ["woman", "man"] (index 0 = woman, 1 = man)
GEN_CLASSES = ["woman", "man"]

TASK_CLASSES = {
    "emotion": EMO_CLASSES,
    "age":     AGE_CLASSES,
    "gender":  GEN_CLASSES,
}

# Дефолтні ваги визначаються автоматично — за розміром head у checkpoint'і.
# (Імена файлів best.pt / best1.pt / best (2).pt не несуть однозначної інформації про задачу.)
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_WEIGHTS = [
    SCRIPT_DIR / "best.pt",
    SCRIPT_DIR / "best1.pt",
    SCRIPT_DIR / "best (1).pt",
    SCRIPT_DIR / "best (2).pt",
]

# task -> очікувана кількість виходів моделі
TASK_N_CLASSES = {
    "emotion": 7,
    "age":     9,
    "gender":  2,
}


def _peek_head_out_features(ckpt_path: Path) -> Optional[int]:
    """Дивиться у state_dict і повертає число виходів head, не вантажучи всю модель."""
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as e:
        print(f"[WARN] Не можу прочитати {ckpt_path}: {type(e).__name__}: {e}")
        return None
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    # Підтримуємо різні префікси / схеми збереження
    for k in ("head.weight", "classifier.weight", "fc.weight",
              "module.head.weight", "_orig_mod.head.weight"):
        if k in state and hasattr(state[k], "shape"):
            return int(state[k].shape[0])
    # fallback: будь-який ключ, що схожий на head
    for k, v in state.items():
        if k.endswith(".weight") and ("head" in k or "classifier" in k) and hasattr(v, "shape"):
            return int(v.shape[0])
    return None


def autodetect_default_weights() -> Dict[str, Path]:
    """Сканує всі знайдені .pt у теці скрипта і мапить task -> файл за розміром head."""
    n_to_task = {n: t for t, n in TASK_N_CLASSES.items()}  # 7->emotion, 9->age, 2->gender
    mapping: Dict[str, Path] = {}
    for p in CANDIDATE_WEIGHTS:
        if not p.exists():
            continue
        n = _peek_head_out_features(p)
        if n is None:
            continue
        task = n_to_task.get(n)
        if task is None:
            print(f"[WARN] {p.name}: head має {n} виходів — невідома задача, пропускаю.")
            continue
        if task in mapping:
            print(f"[WARN] {p.name}: '{task}' уже заматчений з {mapping[task].name} — лишаю перший.")
            continue
        mapping[task] = p
        print(f"[INFO] Автодетект: {p.name} → {task} ({n} класів)")
    return mapping


DEFAULT_WEIGHTS: Dict[str, Path] = {}  # буде заповнено в main() через autodetect_default_weights()


# -------------------------
# Модель (така ж, як у навчанні)
# -------------------------
class ViTClassifier(nn.Module):
    """Архітектура збігається з тренувальною з Face_ViT_SingleTask_and_YOLO.ipynb:
    простий лінійний шар поверх [CLS]-токену ViT.
    """

    def __init__(self, backbone: str, n_classes: int):
        super().__init__()
        self.vit = AutoModel.from_pretrained(backbone)
        hidden = self.vit.config.hidden_size
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]  # [CLS] токен
        return self.head(cls)


# -------------------------
# ImageFolder із жорстко заданим порядком класів (а не алфавітним)
# -------------------------
class FixedClassImageFolder(datasets.ImageFolder):
    """ImageFolder, де class_to_idx визначається переданим списком, а не алфавітом.

    Це критично для задач, де порядок класів у тренуванні не алфавітний
    (наприклад gender: ["woman", "man"] — алфавіт дав би {"man":0, "woman":1}).

    Додатково фільтрує биті/недокачані зображення (як у Face_ViT_SingleTask_and_YOLO.ipynb)
    через PIL.Image.verify().
    """

    def __init__(self, root: str, class_names: List[str], transform=None):
        self._fixed_class_names = list(class_names)
        super().__init__(root, transform=transform)
        self._prune_broken()

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        found = sorted(d.name for d in os.scandir(directory) if d.is_dir())
        missing = [c for c in self._fixed_class_names if c not in found]
        if missing:
            raise FileNotFoundError(
                f"У '{directory}' відсутні очікувані підпапки-класи: {missing}. "
                f"Потрібні (в саме такому порядку індексів): {self._fixed_class_names}. "
                f"Знайдено: {found}"
            )
        extra = [c for c in found if c not in self._fixed_class_names]
        if extra:
            print(f"[WARN] Зайві підпапки в '{directory}' будуть проігноровані: {extra}")
        class_to_idx = {name: i for i, name in enumerate(self._fixed_class_names)}
        return self._fixed_class_names, class_to_idx

    def _prune_broken(self) -> None:
        """Швидко перевіряє кожен файл через PIL.verify() (читає лише заголовки)
        і видаляє биті/непрочитувані з self.samples."""
        good: List[Tuple[str, int]] = []
        bad = 0
        for path, lbl in tqdm(self.samples, desc="  перевірка цілісності", leave=False):
            try:
                with Image.open(path) as im:
                    im.verify()
                good.append((path, lbl))
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                bad += 1
        if bad:
            print(f"[WARN] Пропущено {bad} битих/непрочитаних файлів (залишилось {len(good)}/{len(self.samples)}).")
        self.samples = good
        self.imgs = good
        self.targets = [lbl for _, lbl in good]


def get_loader(data_dir: str, processor, class_names: List[str], batch_size: int = 32):
    """data_dir має структуру: data_dir/<class_name>/<images>."""
    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("height", 224)

    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    ds = FixedClassImageFolder(data_dir, class_names=class_names, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return ds, loader


# -------------------------
# Інференс — збирає прогнози і ймовірності
# -------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    return labels, preds, probs


# -------------------------
# 1. Confusion matrix
# -------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, tag, out_dir: Path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Частка" if normalize else "Кількість"})
    plt.xlabel("Прогноз моделі", fontsize=12, fontweight="bold")
    plt.ylabel("Істинний клас", fontsize=12, fontweight="bold")
    plt.title(f"Матриця помилок — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = out_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname.name}")


# -------------------------
# 2. ROC-curve (One-vs-Rest для кожного класу)
# -------------------------
def plot_roc_curve(y_true, y_prob, class_names, tag, out_dir: Path):
    n_classes = len(class_names)
    # label_binarize для n_classes==2 повертає (N,1), а не (N,2) — будуємо one-hot руками.
    if n_classes == 2:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_bin = np.zeros((len(y_true_arr), 2), dtype=int)
        y_bin[np.arange(len(y_true_arr)), y_true_arr] = 1
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr
        auc_dict[cls] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    # Macro-середнє
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in class_names]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in class_names:
        mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color="navy", lw=3, linestyle="--",
             label=f"macro-середнє (AUC = {macro_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    plt.title(f"ROC-крива — {tag}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = out_dir / f"roc_curve_{tag}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname.name}")
    return auc_dict, macro_auc


# -------------------------
# 3. Метрики по класах (стовпчаста діаграма)
# -------------------------
def plot_per_class_metrics(y_true, y_pred, class_names, tag, out_dir: Path):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)

    x = np.arange(len(class_names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision",
           color="#5C6BC0", edgecolor="black")
    ax.bar(x, recall, width, label="Recall",
           color="#66BB6A", edgecolor="black")
    ax.bar(x + width, f1, width, label="F1-score",
           color="#FFB74D", edgecolor="black")

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.01, f"{p:.2f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(i, r + 0.01, f"{r:.2f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(i + width, f + 0.01, f"{f:.2f}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Значення метрики", fontsize=12, fontweight="bold")
    ax.set_title(f"Метрики по класах — {tag}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = out_dir / f"per_class_metrics_{tag}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname.name}")


# -------------------------
# Завантаження ваг у модель
# -------------------------
def load_model(backbone: str, weights_path: str, n_classes: int, device: torch.device) -> ViTClassifier:
    model = ViTClassifier(backbone, n_classes=n_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# -------------------------
# Один прогон: (task, split, data_dir)
# -------------------------
def evaluate_one(
    task: str,
    split: str,
    data_dir: Path,
    model: ViTClassifier,
    processor,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
) -> dict:
    class_names = TASK_CLASSES[task]
    tag = f"{task}_{split}"

    print("\n" + "=" * 60)
    print(f"  ▶ {tag.upper()}  (data: {data_dir})")
    print("=" * 60)

    print("Завантаження датасету...")
    ds, loader = get_loader(str(data_dir), processor, class_names, batch_size)
    print(f"Зразків: {len(ds)} | class_to_idx: {ds.class_to_idx}")

    print("Запуск інференсу...")
    y_true, y_pred, y_prob = predict(model, loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy:              {accuracy:.4f}")
    print(f"Macro Precision:       {macro_p:.4f}")
    print(f"Macro Recall:          {macro_r:.4f}")
    print(f"Macro F1-score:        {macro_f1:.4f}")
    print(f"Weighted Precision:    {weighted_p:.4f}")
    print(f"Weighted Recall:       {weighted_r:.4f}")
    print(f"Weighted F1-score:     {weighted_f1:.4f}")

    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   digits=4, zero_division=0)
    print(report)

    print("\nСтворення графіків:")
    plot_confusion_matrix(y_true, y_pred, class_names, tag, out_dir, normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names, tag + "_normalized", out_dir, normalize=True)
    auc_per_class, macro_auc = plot_roc_curve(y_true, y_prob, class_names, tag, out_dir)
    plot_per_class_metrics(y_true, y_pred, class_names, tag, out_dir)

    metrics_dict = {
        "task": task,
        "split": split,
        "data_dir": str(data_dir),
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "macro_auc": float(macro_auc),
        "per_class_auc": {k: float(v) for k, v in auc_per_class.items()},
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0)
    }
    out_path = out_dir / f"metrics_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {out_path.name}")

    return metrics_dict


# -------------------------
# Головна функція
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="all",
                        choices=["emotion", "age", "gender", "all"],
                        help="Яку задачу оцінювати. За замовч.: all.")
    parser.add_argument("--split", default="both",
                        choices=["train", "test", "both"],
                        help="Який спліт оцінювати. За замовч.: both.")
    parser.add_argument("--data-root", default="./data", type=Path,
                        help="Корінь даних. Шляхи беруться як <data-root>/<task>/<split>. "
                             "Структуру створює prepare_data.py.")
    parser.add_argument("--data-dir", default=None,
                        help="(опційно) Явний шлях до однієї теки ImageFolder. "
                             "Якщо вказано, потребує single --task і single --split.")
    parser.add_argument("--weights", default=None,
                        help="Шлях до .pt. Сенс лише для single --task. "
                             "Якщо не вказано — підставляться дефолти.")
    parser.add_argument("--backbone", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default=".", type=Path,
                        help="Куди класти графіки/JSON. За замовч.: поточна тека.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Розгортаємо список задач і сплітів
    tasks = ["emotion", "age", "gender"] if args.task == "all" else [args.task]
    splits = ["train", "test"] if args.split == "both" else [args.split]

    # Валідація --data-dir / --weights проти кількості комбінацій
    if args.data_dir is not None and (len(tasks) > 1 or len(splits) > 1):
        parser.error("--data-dir можна використовувати лише з одною задачею і одним сплітом.")
    if args.weights is not None and len(tasks) > 1:
        parser.error("--weights можна використовувати лише з одною задачею (для all — використовуй дефолти).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    # Автодетект дефолтних ваг (потрібно, тільки якщо --weights не задано)
    if args.weights is None:
        print("\nАвтодетект дефолтних ваг ...")
        DEFAULT_WEIGHTS.update(autodetect_default_weights())

    print(f"\nЗавантаження backbone: {args.backbone} ...")
    processor = AutoImageProcessor.from_pretrained(args.backbone)

    summary: List[dict] = []

    for task in tasks:
        class_names = TASK_CLASSES[task]

        # Ваги
        weights = args.weights
        if weights is None:
            default = DEFAULT_WEIGHTS.get(task)
            if default is None or not default.exists():
                print(f"[WARN] Пропускаю {task}: дефолтні ваги відсутні: {default}")
                continue
            weights = str(default)
            print(f"\n[INFO] {task}: ваги = {weights}")

        print(f"[INFO] {task}: завантаження моделі ...")
        model = load_model(args.backbone, weights, len(class_names), device)

        for split in splits:
            # Дата-шлях
            if args.data_dir is not None:
                data_dir = Path(args.data_dir)
            else:
                data_dir = args.data_root / task / split

            if not data_dir.is_dir():
                print(f"[WARN] Пропускаю {task}/{split}: тека не існує: {data_dir}")
                continue

            try:
                m = evaluate_one(
                    task=task,
                    split=split,
                    data_dir=data_dir,
                    model=model,
                    processor=processor,
                    device=device,
                    batch_size=args.batch_size,
                    out_dir=args.out_dir,
                )
                summary.append(m)
            except FileNotFoundError as e:
                print(f"[ERROR] {task}/{split}: {e}")

        # звільняємо GPU перед наступною задачею
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Зведення
    if summary:
        print("\n" + "=" * 60)
        print("  ПІДСУМОК")
        print("=" * 60)
        hdr = f"{'task':<10}{'split':<7}{'N':>8}{'acc':>10}{'macroF1':>10}{'macroAUC':>11}"
        print(hdr)
        print("-" * len(hdr))
        for m in summary:
            print(f"{m['task']:<10}{m['split']:<7}{m['n_samples']:>8}"
                  f"{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['macro_auc']:>11.4f}")

        sum_path = args.out_dir / "summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ {sum_path}")

    print("\nГотово!")


if __name__ == "__main__":
    main()
