# === 0) Завантаження датасетів (якщо змінних ще немає) ===
import kagglehub

if "emo_root" not in globals():
    emo_root = kagglehub.dataset_download("dilkushsingh/facial-emotion-dataset")

if "utk_root" not in globals():
    utk_root = kagglehub.dataset_download("jangedoo/utkface-new")

print("emo_root :", emo_root)
print("utk_root :", utk_root)

# === 1) Побудова тестових наборів + classification_report для best.pt і best1.pt ===
import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoImageProcessor, AutoModel

# Класи емоцій
EMO_CLASSES_7 = ["angry","disgust","fear","happy","neutral","sad","surprise"]
EMO_TO_ID = {c:i for i,c in enumerate(EMO_CLASSES_7)}

# Вікові інтервали
AGE_BINS_9 = [(0,2),(3,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,120)]
AGE_NAMES_9 = [f"{a}-{b}" for a,b in AGE_BINS_9]

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")
BACKBONE = globals().get("backbone", "google/vit-base-patch16-224-in21k")

def age_to_bin(age: int) -> int:
    for i,(a0,a1) in enumerate(AGE_BINS_9):
        if a0 <= age <= a1:
            return i
    return len(AGE_BINS_9)-1

def filter_bad_decode(df: pd.DataFrame, col="image_path") -> pd.DataFrame:
    good = []
    for _, row in df.iterrows():
        p = row[col]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            good.append(row)
    return pd.DataFrame(good).reset_index(drop=True)

def collect_emotion_test(emo_root: str) -> pd.DataFrame:
    test_dir = os.path.join(emo_root, "test_dir")
    rows = []
    for cls in EMO_CLASSES_7:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        cls_id = EMO_TO_ID[cls]
        for dp, _, files in os.walk(cls_dir):
            for fn in files:
                if fn.lower().endswith(IMG_EXT):
                    rows.append({"image_path": os.path.join(dp, fn), "label": cls_id})
    df = pd.DataFrame(rows)
    return filter_bad_decode(df)

def collect_age_test(utk_root: str) -> pd.DataFrame:
    rows = []
    for dp, _, files in os.walk(utk_root):
        for fn in files:
            if not fn.lower().endswith(IMG_EXT):
                continue
            base = fn.split(".")[0]
            parts = base.split("_")
            if len(parts) < 2:
                continue
            try:
                age = int(parts[0])
                rows.append({"image_path": os.path.join(dp, fn), "label": age_to_bin(age)})
            except:
                continue

    utk_df = pd.DataFrame(rows)
    utk_df = filter_bad_decode(utk_df)

    _, age_test = train_test_split(
        utk_df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=utk_df["label"]
    )
    return age_test.reset_index(drop=True)

def read_rgb_safe(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class SingleTaskImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, label_col: str = "label"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        for _ in range(5):
            row = self.df.iloc[idx]
            img = read_rgb_safe(row["image_path"])
            if img is not None:
                x = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
                y = int(row[self.label_col])
                return x, y
            idx = np.random.randint(0, len(self.df))

        dummy = np.zeros((224,224,3), dtype=np.uint8)
        x = self.processor(images=dummy, return_tensors="pt")["pixel_values"].squeeze(0)
        return x, 0

def collate_single(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys

class ViTClassifier(nn.Module):
    def __init__(self, backbone: str, n_classes: int):
        super().__init__()
        self.vit = AutoModel.from_pretrained(backbone)
        h = self.vit.config.hidden_size
        self.head = nn.Linear(h, n_classes)

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)

@torch.no_grad()
def evaluate_report(weights_path: str, test_df: pd.DataFrame, class_names: list[str],
                    batch_size=128, workers=2):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Не знайдено файл ваг: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(BACKBONE, use_fast=True)

    ds = SingleTaskImageDataset(test_df, processor, label_col="label")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_single)

    model = ViTClassifier(BACKBONE, n_classes=len(class_names)).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    print("\nФайл ваг:", weights_path)
    print("Точність:", round(float(acc) * 100, 2), "%")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

# Побудова тестових вибірок
emo_test_df = collect_emotion_test(emo_root)
age_test = collect_age_test(utk_root)

print("Емоції test:", len(emo_test_df))
print("Вік test   :", len(age_test))

# Звіти
evaluate_report("best.pt", emo_test_df, EMO_CLASSES_7)
evaluate_report("best1.pt", age_test, AGE_NAMES_9)