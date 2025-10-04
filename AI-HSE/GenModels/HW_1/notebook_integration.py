"""
Интеграция с Jupyter Notebook
Примеры кода для использования в вашем homework notebook
"""

import pandas as pd
import random
from audio_processor import AudioProcessor, load_urls_from_file

def get_sample_transcriptions(num_samples=10, random_seed=42):
    """
    Получить транскрипции для случайных аудиофайлов
    
    Args:
        num_samples: Количество образцов
        random_seed: Seed для воспроизводимости
        
    Returns:
        pd.DataFrame: Результаты транскрипции
    """
    # Загружаем URL
    urls = load_urls_from_file("urls_normalized.tsv")
    
    if not urls:
        print("Ошибка: не удалось загрузить URL из файла")
        return pd.DataFrame()
    
    # Создаем процессор
    processor = AudioProcessor()
    
    # Выбираем случайные URL
    random.seed(random_seed)
    selected_urls = random.sample(urls, min(num_samples, len(urls)))
    
    results = []
    
    print(f"Обрабатываем {len(selected_urls)} аудиофайлов...")
    
    for i, url in enumerate(selected_urls):
        print(f"Файл {i+1}/{len(selected_urls)}: {url[:50]}...")
        
        # Транскрибируем
        transcription = processor.process_audio_from_url(url)
        
        results.append({
            'url': url,
            'transcription': transcription if transcription else "Ошибка транскрипции"
        })
    
    return pd.DataFrame(results)

def display_transcription_results(df):
    """
    Красиво отобразить результаты транскрипции
    
    Args:
        df: DataFrame с результатами
    """
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ ТРАНСКРИПЦИИ АУДИОФАЙЛОВ")
    print("=" * 80)
    
    for idx, row in df.iterrows():
        print(f"\nФайл {idx + 1}:")
        print(f"URL: {row['url']}")
        print(f"Транскрипция: {row['transcription']}")
        print("-" * 80)

# Пример использования в notebook:
"""
# В вашем notebook выполните:

# 1. Импортируйте функции
from notebook_integration import get_sample_transcriptions, display_transcription_results

# 2. Получите транскрипции для 10 случайных файлов
results_df = get_sample_transcriptions(num_samples=10)

# 3. Отобразите результаты
display_transcription_results(results_df)

# 4. Сохраните результаты
results_df.to_csv('whisper_transcriptions.csv', index=False)
"""

