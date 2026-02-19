# iSTFTNet Vocoder Project

Проект по синтезу речи на основе вокодера iSTFTNet

## Установка

1. Клонируйте репозиторий и перейдите в ветку `sound-hw3`:
```bash
git clone <https://github.com/Lzora7/Study.git>
cd HW_3
git checkout sound-hw3
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. (Опционально) Загрузите предобученный чекпойнт вокодера для синтеза без обучения:
```bash
python download_checkpoint.py
```
Чекпойнт сохранится в `saved/demo_run/model_best.pth`. 
Альтернатива: чекпойнт автоматически скачивается при последовательном запуске ячеек в `demo.ipynb`.

## Обучение модели

Для обучения вокодера используйте:
```bash
python train.py
```

Для обучения с другим датасетом или конфигурацией:
```bash
python train.py datasets=custom_dir datasets.custom_dir.train.root_dir=/path/to/your/data
```

Основные параметры задаются через Hydra (см. раздел «Конфигурация»).

## Синтез (Synthesis)

Скрипт `synthesize.py` преобразует аудио в mel-спектрограммы и генерирует волновые формы с помощью обученного вокодера. Результат сохраняется в указанную директорию.

**Обязательные параметры (через Hydra):**
- `checkpoint` — путь к чекпойнту модели (например, `saved/2nd_try/model_best.pth`)
- `input_dir` — директория с входными аудиофайлами (`.wav`, `.flac`, `.mp3`, `.m4a`)
- `output_dir` — директория для сохранения сгенерированных аудиофайлов

**Пример:**
```bash
python synthesize.py \
    checkpoint=saved/2nd_try/model_best.pth \
    input_dir=/path/to/audio \
    output_dir=/path/to/output
```

Сгенерированные файлы сохраняются как `{original_name}_synthesized.wav` в `output_dir`. Параметры mel-спектрограммы (sample rate, hop_length и т.д.) берутся из чекпойнта или из конфига.

## Подготовка данных для обучения

### Датасет RUSLAN (по умолчанию)

Используется конфиг `datasets=ruslan`. Аудио загружается в `data/datasets/ruslan/audio` (при необходимости — автоматическая загрузка, если в конфиге указано `download: true`).

### Кастомный датасет

1. Подготовьте данные в формате:
```
NameOfTheDirectory
├── audio
│   ├── utterance1.wav   # допустимы .flac, .mp3, .m4a
│   ├── utterance2.wav
│   └── ...
└── transcriptions       # опционально
    ├── utterance1.txt
    ├── utterance2.txt
    └── ...
```

2. Запустите обучение:
```bash
python train.py datasets=custom_dir \
    datasets.custom_dir.train.root_dir=/path/to/NameOfTheDirectory
```

Для синтеза в `synthesize.py` в качестве `input_dir` можно указать либо корень (с подпапкой `audio/`), либо сразу папку `audio/`.

## Демонстрация

Для интерактивной демонстрации откройте `demo.ipynb` в Jupyter Notebook или Google Colab.

Ноутбук включает:
1. Клонирование репозитория и установку зависимостей
2. Загрузку весов предобученного вокодера
3. Загрузку семпла RUSLAN
4. Ресинтез: Audio -> Mel -> Vocoder -> Audio
5. Рекомендации по MOS (Mean Opinion Score)
6. Анализ качества вокодера на обучающих и внешних данных

## Структура проекта

```
HW_3/
├── src/
│   ├── configs/           # Конфигурации Hydra
│   │   ├── datasets/       # ruslan.yaml, custom_dir.yaml
│   │   ├── model/         # istftnet.yaml
│   │   └── ...
│   ├── datasets/          # RUSLANDataset, CustomDirDataset
│   ├── model/             # iSTFTNet, HiFiGAN-дискриминатор
│   ├── loss/              # iSTFTLoss (mel + adversarial + feature matching)
│   ├── metrics/           # Метрики обучения
│   ├── trainer/           # GANTrainer, base_trainer
│   ├── transforms/        # Mel-спектрограмма, аугментации
│   └── logger/            # Comet, WandB
├── train.py               # Скрипт обучения (GAN)
├── synthesize.py          # Скрипт синтеза по чекпойнту
├── download_checkpoint.py # Скрипт загрузки предобученного чекпойнта
├── demo.ipynb             # Демонстрационный ноутбук
```

## Конфигурация

Проект использует Hydra. Базовый конфиг — `src/configs/baseline.yaml`.

- **Модель:** генератор iSTFTNet, дискриминатор HiFiGAN (multi-period). Конфиг модели: `src/configs/model/istftnet.yaml`.
- **Датасеты:** `datasets=ruslan` (по умолчанию) или `datasets=custom_dir` с указанием `datasets.custom_dir.train.root_dir` и при необходимости `datasets.custom_dir.test.root_dir`.
- **Логирование:** в конфиге задаётся `writer` (например, `cometml` или `wandb`). Для Comet/WandB при необходимости настройте переменные окружения (см. `.env`).
