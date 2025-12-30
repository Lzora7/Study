# Deep Speech 2 ASR Project

Проект по распознаванию речи на основе архитектуры Deep Speech 2.

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository_url>
cd HW_2_dpspch
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Обучение модели

Для обучения модели используйте:
```bash
python train.py
```

Для обучения конкретной модели:
```bash
python train.py model=deep_speech2_small
```

## Инференс (Inference)

### Использование с предустановленными датасетами

Для запуска inference на предустановленных датасетах (например, LibriSpeech):
```bash
python inference.py inferencer.from_pretrained=saved/<model_name>/model_best.pth
```

### Использование с кастомным датасетом

1. Подготовьте датасет в следующем формате:
```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav  # может быть .flac или .mp3
│   ├── UtteranceID2.wav
│   └── ...
└── transcriptions  # опционально - ground truth транскрипции
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    └── ...
```

2. Запустите inference:
```bash
python inference.py \
    datasets=custom_dataset \
    datasets.custom_dataset.test.audio_dir=/path/to/audio \
    datasets.custom_dataset.test.transcription_dir=/path/to/transcriptions \
    inferencer.from_pretrained=saved/<model_name>/model_best.pth \
    inferencer.save_path=custom_dataset_results
```

Предсказания будут сохранены в директории `data/saved/custom_dataset_results/test/` в виде файлов `{UtteranceID}.txt`, где каждый файл содержит предсказанный текст для соответствующего аудио файла.

## Подсчет метрик WER/CER

Для подсчета метрик Word Error Rate (WER) и Character Error Rate (CER) используйте скрипт `calc_metrics.py`:

```bash
python calc_metrics.py \
    /path/to/predictions \
    /path/to/ground_truth_transcriptions \
    --show-samples 5 \
    --output metrics.json
```

**Параметры:**
- `predictions_dir`: Директория с предсказаниями (файлы `{UtteranceID}.txt`)
- `ground_truth_dir`: Директория с ground truth транскрипциями (файлы `{UtteranceID}.txt`)
- `--show-samples N`: Количество примеров для отображения (по умолчанию: 5)
- `--output PATH`: Опционально: путь для сохранения детальных метрик в JSON формате

**Пример использования:**
```bash
# После запуска inference на кастомном датасете
python calc_metrics.py \
    data/saved/custom_dataset_results/test \
    data/custom_dataset/transcriptions \
    --show-samples 10 \
    --output results/metrics.json
```

Скрипт выведет:
- Средние метрики WER и CER
- Количество обработанных примеров
- Примеры предсказаний (если указан `--show-samples`)

## Демонстрация

Для интерактивной демонстрации использования проекта откройте `demo.ipynb` в Jupyter Notebook или Google Colab.

Ноутбук включает:
1. Клонирование репозитория и установку зависимостей
2. Загрузку весов модели
3. Запуск inference.py
4. Работу с собственными данными
5. Демонстрацию ASR-аугментаций
6. Демонстрацию hand-crafted beam search

## Структура проекта

```
HW_2_dpspch/
├── src/
│   ├── configs/          # Конфигурации Hydra
│   ├── datasets/         # Классы датасетов
│   │   └── custom_dir_audio_dataset.py  # CustomDirAudioDataset
│   ├── model/            # Архитектуры моделей
│   ├── metrics/          # Метрики (WER, CER)
│   ├── trainer/          # Тренировка и инференс
│   └── transforms/       # Аугментации
├── train.py              # Скрипт обучения
├── inference.py          # Скрипт инференса
├── calc_metrics.py       # Скрипт подсчета метрик
├── demo.ipynb           # Демонстрационный ноутбук
└── README.md            # Этот файл
```

## Конфигурация

Проект использует Hydra для управления конфигурациями. Основные конфигурации находятся в `src/configs/`.

Для кастомного датасета создан конфиг `src/configs/datasets/custom_dataset.yaml`, который можно использовать через:
```bash
python inference.py datasets=custom_dataset
```

