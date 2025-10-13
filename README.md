# WINSTARS.AI 2025 — Tasks 1 & 2

Bilingual documentation (English / Русский) for training and running the two tasks in this repository:
- Task 1: MNIST digit classification with 3 algorithms (RandomForest, Feed-Forward NN, CNN) in TensorFlow + scikit-learn.
- Task 2: Simple multimodal pipeline — NER (HuggingFace Transformers) + image classification (ResNet18 in PyTorch) on Animals-10 demo. The pipeline returns a boolean if a detected entity matches the predicted image class.

Each task has its own detailed README:
- [Task 1 README](src/task1/README.md)
- [Task 2 README](src/task2/README.md)

Contents:
- English
  - Overview
  - Quickstart
  - Task 1: MNIST Classification
  - Task 2: Multimodal Pipeline (NER + Image)
  - Project structure, configuration, testing, troubleshooting
- Русский
  - Обзор
  - Быстрый старт
  - Задача 1: Классификация MNIST
  - Задача 2: Мультимодальный конвейер (NER + Изображение)
  - Структура проекта, конфигурация, тестирование, устранение неполадок

---

## English

### 1) Overview
This repository contains two self-contained tasks:
- Task 1 (src/task1): Train MNIST classification models and report accuracy.
- Task 2 (src/task2): Train an image classifier (ResNet18) and run a small pipeline that extracts entities from text and classifies an image; the result is a boolean whether any entity matches the predicted class.

Included demo data and model:
- Demo dataset for Task 2: data/animals10_demo/ (10 classes, 2 images per class)
- Pretrained demo model: models/image/resnet18.pth (can be used for pipeline inference)

Helper scripts (Windows):
- setup_env.bat — create venv and install dependencies
- run_mnist_train.bat — train MNIST CNN (example)
- run_pipeline.bat — example on how to call the pipeline (paths and class order may need adjustment; see below)

Requirements (see requirements.txt):
- Python 3.8+
- TensorFlow 2.9+
- PyTorch + torchvision
- transformers, datasets, seqeval (for NER)
- scikit-learn, numpy, pandas, Pillow, jupyter, pytest, tqdm, joblib

Note: The first NER run downloads the model dslim/bert-base-NER from HuggingFace.


### 2) Quickstart

- Windows (CMD):
  ```bat
  setup_env.bat
  venv\Scripts\activate
  ```

- Linux/macOS:
  ```bash
  bash setup_env.sh
  source venv/bin/activate
  ```

If you prefer manual setup:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU notes:
- PyTorch with CUDA: install a CUDA-enabled build from https://pytorch.org/get-started/locally/
- TensorFlow GPU on Windows requires compatible CUDA/cuDNN; otherwise CPU will be used automatically.


### 3) Task 1 — MNIST Classification

Main entry: src/task1/train_mnist.py
- Algorithms: rf (RandomForest), nn (dense NN), cnn (simple ConvNet)
- Data: tf.keras.datasets.mnist (downloaded automatically)
- Output: prints validation accuracy on 20% hold-out split

CLI:
```bash
python src/task1/train_mnist.py \
  --algo {rf|nn|cnn} \
  --epochs <int> \
  --limit <int> \
  --config <path_to_json>
```
- --algo: choose model (default from config or cnn)
- --epochs: for nn/cnn training (default 3)
- --limit: limit dataset size for quick runs (optional)
- --config: JSON config file with keys like {"algorithm": "cnn", "training": {"epochs": 3}, "limit": 2000}

Examples:
```bash
# Fast CNN run
python src/task1/train_mnist.py --algo cnn --epochs 3 --limit 2000

# RandomForest (no epochs argument is needed for RF)
python src/task1/train_mnist.py --algo rf --limit 5000

# Load parameters from JSON
python src/task1/train_mnist.py --config config/config.json
```

Minimal architecture/flow:
```
MNIST (70k) -> normalize -> split(80/20)
    |                         |
    | train                   | evaluate
    v                         v
[RF or NN or CNN] --------> accuracy printed
```

Relevant files:
- src/task1/mnist_wrapper.py — factory wrapper for algorithms
- src/task1/mnist_rf.py — RandomForestClassifier
- src/task1/mnist_nn.py — Keras dense network
- src/task1/mnist_cnn.py — Keras CNN
- src/task1/train_mnist.py — CLI training script


### 4) Task 2 — Multimodal Pipeline (NER + Image)

Goal: Decide if a named entity in the text matches the predicted image class.
- NER: HuggingFace pipeline with model dslim/bert-base-NER
- Image classification: ResNet18 (torchvision) fine-tuned on Animals-10 demo
- Demo data: data/animals10_demo/<class>/<image>.jpg
- Demo model: models/image/resnet18.pth

Train the image model (optional — a demo model is already included):
```bash
python src/task2/train_image.py \
  --data_dir data/animals10_demo \
  --output_dir models/image
```
This will save models/image/resnet18.pth.

Run the pipeline:
```bash
python src/task2/pipeline.py \
  --text "There is a zebra in the picture" \
  --image data/animals10_demo/zebra/zebra_1.jpg \
  --image_model models/image/resnet18.pth \
  --classes bear cat cow dog elephant giraffe horse monkey sheep zebra
```
Notes:
- --classes must match the exact class order the model was trained with. For ImageFolder, this is alphabetical by folder name. For the provided demo data, the order is:
  bear, cat, cow, dog, elephant, giraffe, horse, monkey, sheep, zebra
- The first pipeline run will download the NER model from HuggingFace.
- The pipeline prints extracted entities, image prediction and probability, and the final boolean Match.

Minimal architecture/flow:
```
Text  -> [HuggingFace NER] -> [entities]
Image -> [ResNet18 classifier] -> [class, prob]
                         |           |
                         \--------- compare (substring match) ---> Boolean
```

Relevant files:
- src/task2/train_image.py — train ResNet18 on ImageFolder
- src/task2/infer_image.py — utility to predict a single image
- src/task2/infer_ner.py — HuggingFace NER wrapper
- src/task2/pipeline.py — glue code combining NER + image classifier

Helper (Windows) examples:
- run_pipeline.bat shows a sample call; ensure the image path exists (e.g., data/animals10_demo/cow/cow_1.jpg) and pass the correct classes order as documented above.


### 5) Project structure
```
config/
  config.json
  project_config copy.json
data/
  animals10_demo/
    <10 classes>/<2 images each>
docs/USAGE.md (legacy; see this README for up-to-date commands)
models/
  image/resnet18.pth (demo weights)
notebooks/
  task1_mnist_demo.ipynb
  task2_pipeline_demo.ipynb
show_result/
  test1.png
src/
  task1/ (MNIST models and trainer)
  task2/ (image trainer, NER, pipeline)
  utils/config_loader.py
tests/
  test_imports.py
Dockerfile
requirements.txt
setup_env.bat, setup_env.sh
run_mnist_train.bat, run_pipeline.bat
```


### 6) Configuration
Some scripts support a JSON config via --config. Example (config/config.json provided):
```json
{
  "project": "winstars_internship_v10",
  "ner": {
    "model": "dslim/bert-base-NER",
    "epochs": 3,
    "batch_size": 8
  },
  "image": {
    "arch": "resnet18",
    "epochs": 5,
    "batch_size": 16
  },
  "training": { "seed": 42 }
}
```
Note: Each training script only reads the fields it knows about (see the script’s argparse and config parsing logic).


### 7) Testing
A lightweight smoke test verifies imports:
```bash
pytest -k test_imports -q
```


### 8) Troubleshooting
- NER download hangs: ensure internet connectivity and that transformers can access HuggingFace.
- CPU vs GPU: both TF and PyTorch fall back to CPU. Training will be slower on CPU but should still run on the demo dataset.
- Class order mismatch: if predictions look wrong, confirm --classes matches the alphabetical order of folders used during training.
- Inference model path: the pipeline should be given a trained model via --image_model; otherwise the image classifier will behave randomly.
- Windows path issues: prefer forward slashes or escape backslashes in quoted strings, e.g., "data/animals10_demo/cow/cow_1.jpg".


---

## Русский

### 1) Обзор
Репозиторий содержит две задачи:
- Задача 1 (src/task1): Обучение моделей классификации MNIST и вывод точности.
- Задача 2 (src/task2): Обучение классификатора изображений (ResNet18) и запуск простого пайплайна: извлечение сущностей из текста (NER) + классификация изображения; результат — булево значение, совпадает ли какая-либо сущность с предсказанным классом изображения.

В комплекте:
- Демо-данные для задачи 2: data/animals10_demo/ (10 классов, по 2 изображения)
- Демо-модель: models/image/resnet18.pth (можно сразу использовать в пайплайне)

Скрипты-помощники (Windows):
- setup_env.bat — создание окружения и установка зависимостей
- run_mnist_train.bat — пример запуска обучения MNIST (CNN)
- run_pipeline.bat — пример вызова пайплайна (пути и порядок классов при необходимости поправьте; см. ниже)


### 2) Быстрый старт

- Windows (CMD):
  ```bat
  setup_env.bat
  venv\Scripts\activate
  ```

- Linux/macOS:
  ```bash
  bash setup_env.sh
  source venv/bin/activate
  ```

Либо вручную:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Примечания по GPU:
- Для PyTorch с CUDA установите сборку с поддержкой CUDA: https://pytorch.org/get-started/locally/
- Для TensorFlow на Windows нужна совместимая версия CUDA/cuDNN; иначе используется CPU.


### 3) Задача 1 — Классификация MNIST

Точка входа: src/task1/train_mnist.py
- Алгоритмы: rf (RandomForest), nn (полносвязная сеть), cnn (сверточная сеть)
- Данные: tf.keras.datasets.mnist (скачиваются автом��тически)
- Вывод: печать точности на валидации (20% отложенное множество)

CLI:
```bash
python src/task1/train_mnist.py \
  --algo {rf|nn|cnn} \
  --epochs <int> \
  --limit <int> \
  --config <путь_к_json>
```
- --algo: выбор модели (по умолчанию из конфига или cnn)
- --epochs: число эпох для nn/cnn (по умолчанию 3)
- --limit: ограничение размера выборки для быстрых прогонов (опционально)
- --config: JSON с параметрами, например {"algorithm": "cnn", "training": {"epochs": 3}, "limit": 2000}

Примеры:
```bash
# Быстрый прогон CNN
python src/task1/train_mnist.py --algo cnn --epochs 3 --limit 2000

# RandomForest (epochs не требуется)
python src/task1/train_mnist.py --algo rf --limit 5000

# Загрузка параметров из JSON
python src/task1/train_mnist.py --config config/config.json
```

Схема/поток выполнения:
```
MNIST (70k) -> нормализация -> разбиение(80/20)
    |                               |
    | обучение                      | оценка
    v                               v
[RF или NN или CNN] -----------> п��чать accuracy
```

Ключевые файлы:
- src/task1/mnist_wrapper.py — фабрика алгоритмов
- src/task1/mnist_rf.py — RandomForestClassifier
- src/task1/mnist_nn.py — Keras полносвязная сеть
- src/task1/mnist_cnn.py — Keras CNN
- src/task1/train_mnist.py — CLI-скрипт обучения


### 4) Задача 2 — Мультимодальный пайплайн (NER + Изображение)

Цель: проверить, совпадает ли именная сущность в тексте с предсказанным классом изображения.
- NER: HuggingFace pipeline с моделью dslim/bert-base-NER
- Классификация изображений: ResNet18 (torchvision) на Animals-10 demo
- Демо-данные: data/animals10_demo/<класс>/<изображение>.jpg
- Демо-модель: models/image/resnet18.pth

Обучение модели изображений (опционально — демо-веса уже есть):
```bash
python src/task2/train_image.py \
  --data_dir data/animals10_demo \
  --output_dir models/image
```
Модель будет сохранена в models/image/resnet18.pth.

Запуск пайплайна:
```bash
python src/task2/pipeline.py \
  --text "На картинке изображена зебра" \
  --image data/animals10_demo/zebra/zebra_1.jpg \
  --image_model models/image/resnet18.pth \
  --classes bear cat cow dog elephant giraffe horse monkey sheep zebra
```
Примечания:
- --classes должен соответствовать порядку классов, с которым обучалась модель. Для ImageFolder порядок — алфавит имен папок. Для демо: 
  bear, cat, cow, dog, elephant, giraffe, horse, monkey, sheep, zebra
- При первом запуске NER происходит скачивание модели с HuggingFace.
- Скрипт печатает извлеченные сущности, предсказание класса изображения и вероятность, затем булево значение Match.

Схема/поток выполнения:
```
Текст  -> [HuggingFace NER] -> [список сущностей]
Изобр. -> [ResNet18 классификатор] -> [класс, prob]
                         |                  |
                         \------ сравнение (подстрока) ----> Булево
```

Ключевые файлы:
- src/task2/train_image.py — обучение ResNet18 на ImageFolder
- src/task2/infer_image.py — утилита предсказания для одного изображения
- src/task2/infer_ner.py — обертка NER на HuggingFace
- src/task2/pipeline.py — объединение NER + классификатор изображений

Подсказка (Windows):
- run_pipeline.bat содержит пример вызова; убедитесь, что путь к изображению существует (например, data/animals10_demo/cow/cow_1.jpg) и передан корректный порядок классов, как указано выше.


### 5) Структура проекта
См. краткую схему в английской части. Важные каталоги и файлы:
- config/ — конфигурации
- data/animals10_demo/ — демо-изображения по 10 классам
- models/image/resnet18.pth — демо-веса
- notebooks/ — ноутбуки для демонстрации
- src/task1, src/task2 — код задач 1 и 2
- tests/test_imports.py — минимальный тест


### 6) Конфигурация
Некоторые скрипты принимают JSON через --config. Пример в config/config.json. Учтите, что каждый скрипт читает только релевантные поля.


### 7) Тестирование
Запустить минимальный тест:
```bash
pytest -k test_imports -q
```


### 8) Частые проблемы
- Скачивание NER: проверьте интерне�� и доступ к HuggingFace.
- CPU/GPU: при отсутствии GPU обучение пройдет на CPU (медленнее, но для демо-данных достаточно).
- Порядок классов: если предсказания кажутся некорректными, проверьте — соответствует ли --classes алфавитному порядку папок.
- Путь к модели: для пайплайна обязательно укажите --image_model с весами обученной модели.
- Пути Windows: используйте прямые слэши или экранируйте обратные.

---

This README supersedes docs/USAGE.md where they diverge. Use the commands in this file for the current codebase.