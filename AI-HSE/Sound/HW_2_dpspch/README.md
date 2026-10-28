# Распознавание речи — Deep Speech 2

Пайплайн преобразования аудио в текст: подготовка данных, обучение модели, декодирование и оценка качества.

## Задачи и реализация

- Модель распознавания речи на основе Deep Speech 2.
- Обучение с CTC loss.
- Аугментации аудио и спектрограмм.
- Инференс на собственных аудиофайлах.
- Оценка качества по WER и CER.
- Конфигурации нескольких размеров модели.

## Технологии

Python, PyTorch, torchaudio, Hydra, OmegaConf, Comet ML / Weights & Biases.

## Установка

Из корня репозитория:

```bash
cd AI-HSE/Sound/HW_2_dpspch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Распознавание собственных аудиофайлов

Подготовьте каталог:

```text
my-data/
├── audio/
│   ├── sample-01.wav
│   └── sample-02.wav
└── transcriptions/
    ├── sample-01.txt
    └── sample-02.txt
```

Транскрипции нужны для оценки качества. Имена текстовых и аудиофайлов должны совпадать.

Подготовьте чекпойнт обученной модели и выберите соответствующую ему конфигурацию архитектуры. Например, для `deep_speech2_small`:

```bash
python inference.py \
    model=deep_speech2_small \
    datasets=custom_dataset \
    datasets.test.audio_dir=/absolute/path/my-data/audio \
    datasets.test.transcription_dir=/absolute/path/my-data/transcriptions \
    inferencer.from_pretrained=/absolute/path/model_best.pth \
    inferencer.save_path=custom_results
```

Предсказания сохраняются в `data/saved/custom_results/test/`. Для запуска без эталонных текстов не задавайте `datasets.test.transcription_dir`; WER/CER в таком сценарии не характеризуют качество распознавания.

## Оценка качества

```bash
python calc_metrics.py \
    data/saved/custom_results/test \
    /absolute/path/my-data/transcriptions \
    --show-samples 5 \
    --output metrics.json
```

WER измеряет ошибки на уровне слов, CER — на уровне символов. Для обеих метрик меньшее значение означает лучшее качество.

## Обучение

```bash
python train.py model=deep_speech2_small
```

Перед запуском проверьте конфигурацию датасета, параметры обучения и настройки выбранного сервиса логирования. Для обучения рекомендуется GPU. Конфигурации находятся в `src/configs/`.

## Демонстрация и результаты

[demo.ipynb](demo.ipynb) содержит примеры инференса, аугментаций и декодирования. Для выполнения необходимы веса модели. Численные WER/CER следует сравнивать на одной тестовой выборке с указанием конфигурации модели и чекпойнта.

Основа проекта — учебный шаблон ASR. Лицензия: [LICENSE](LICENSE).

[Все проекты Sound](../README.md)
