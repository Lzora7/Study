# Нейронный вокодер iSTFTNet

Восстановление звуковой волны по mel-спектрограмме с использованием генератора iSTFTNet и GAN-обучения.

Демонстрационный сценарий: **исходное аудио → mel-спектрограмма → вокодер → восстановленное аудио**.

## Задачи и реализация

- Генератор с блоками увеличения временного разрешения и восстановлением сигнала через inverse STFT.
- Multi-Period Discriminator для GAN-обучения.
- Обучение с mel, adversarial и feature matching loss.
- Подготовка данных RUSLAN и собственных аудиозаписей.
- Загрузка чекпойнта и пакетный ресинтез аудиофайлов.
- Анализ качества на обучающих и внешних данных в демонстрационном ноутбуке.

## Технологии

Python, PyTorch, torchaudio, librosa, NumPy, SciPy, SoundFile, Hydra, OmegaConf, Comet ML / Weights & Biases.

## Установка

Из корня репозитория:

```bash
cd AI-HSE/Sound/HW_3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

В текущем `requirements.txt` указаны `torch==2.2.0` и `numpy>=2.0,<3`. Совместимость этого набора зависимостей и полный запуск в чистом окружении не проверены; при ошибках NumPy/PyTorch потребуется согласовать версии.

## Ресинтез без обучения

Загрузите предобученный чекпойнт:

```bash
python download_checkpoint.py
```

По умолчанию он сохраняется в `saved/demo_run/model_best.pth`. Загрузка использует внешнюю ссылку Google Drive из `download_checkpoint.py`; файл должен быть доступен для скачивания. Собственный источник можно задать через `--url`.

Поместите исходные аудиофайлы в отдельную папку и выполните:

```bash
python synthesize.py \
    checkpoint=saved/demo_run/model_best.pth \
    input_dir=/absolute/path/audio \
    output_dir=/absolute/path/results
```

Результаты сохраняются как `{original_name}_synthesized.wav`. Параметры mel-преобразования берутся из чекпойнта или конфигурации.

Это восстановление аудио по спектрограмме. Для генерации речи из текста потребуется отдельная модель, преобразующая текст в mel-спектрограмму.

## Обучение

На RUSLAN:

```bash
python train.py
```

Базовая конфигурация включает загрузку данных RUSLAN в `data/datasets/ruslan/audio`.

На собственных данных с подпапкой `audio/`:

```bash
python train.py \
    datasets=custom_dir \
    datasets.train.root_dir=/absolute/path/my-data \
    datasets.test.root_dir=/absolute/path/my-data
```

Конфигурация `custom_dir` использует разделение на train/test: 90% для обучения и 10% для проверки с фиксированным seed. Перед обучением настройте сервис логирования; по умолчанию выбран Comet ML. Для обучения рекомендуется GPU.

## Демонстрация и результаты

В [demo.ipynb](demo.ipynb) представлены ресинтез аудио и анализ качества на разных данных. Для прослушивания сравнивайте оригинал и восстановленный сигнал; численные результаты оценки следует приводить вместе с описанием выборки и протокола.

Чекпойнты хранятся отдельно от Git. Предобученная модель загружается через `download_checkpoint.py`.

## Конфигурация и код

- `src/configs/baseline.yaml` — базовые параметры обучения и синтеза.
- `src/configs/datasets/` — RUSLAN и собственные данные.
- `src/model/` — генератор iSTFTNet и дискриминатор.
- `src/loss/` — функции потерь GAN и mel-реконструкции.
- `src/transforms/mel_spectrogram.py` — извлечение mel-признаков.
- `train.py` — обучение; `synthesize.py` — ресинтез по чекпойнту.

Лицензия: [LICENSE](LICENSE).

[Все проекты Sound](../README.md)
