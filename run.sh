#!/usr/bin/env bash
# Один запуск, що робить усе:
#   1) (за потреби) активує локальний venv
#   2) готує дані (prepare_data.py) — пропускається, якщо тека ./data вже є
#   3) проганяє evaluate.py для всіх задач (emotion+age+gender) і обох сплітів (train+test)
#
# Опційно:
#   ./run.sh cam    — після цього ще запустить main.py з вебкамерою
#   ./run.sh prep   — лише підготувати дані
#   ./run.sh eval   — лише оцінювання (припускає, що ./data вже існує)
#   ./run.sh all    — (за замовч.) prep + eval
#
# Перед першим запуском:
#   python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
#   (а також налаштувати ~/.kaggle/kaggle.json для kagglehub)

set -e

cd "$(dirname "$0")"

# Активуємо venv, якщо є
if [ -f "venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

MODE="${1:-all}"

run_prep() {
    if [ -d "data" ] && [ -d "data/emotion" ] && [ -d "data/age" ] && [ -d "data/gender" ]; then
        echo "[run.sh] ./data вже існує — пропускаю prepare_data.py"
    else
        echo "[run.sh] Готую дані ..."
        python3 prepare_data.py --task all --split both --mode symlink
    fi
}

run_eval() {
    echo "[run.sh] Оцінюю всі задачі × всі спліти ..."
    python3 evaluate.py --task all --split both --data-root ./data
}

run_cam() {
    echo "[run.sh] Запускаю main.py (вебкамера) ..."
    # Реальний розподіл за розміром head:
    #   best.pt   → emotion (7 класів)
    #   best1.pt  → age (9 класів)
    #   best (2).pt → gender (2 класів)
    # (main.py також авто-перерозподілить, якщо переплутати.)
    python3 main.py \
        --emotion-weights "best.pt" \
        --age-weights "best1.pt" \
        --gender-weights "best (2).pt"
}

case "$MODE" in
    prep)  run_prep ;;
    eval)  run_eval ;;
    cam)   run_prep; run_eval; run_cam ;;
    all|"") run_prep; run_eval ;;
    *) echo "Невідомий режим: $MODE. Допустимі: prep | eval | cam | all"; exit 2 ;;
esac

echo "[run.sh] Готово."
