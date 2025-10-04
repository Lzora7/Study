"""
Скрипт для скачивания всех аудиофайлов из urls_normalized.tsv
Сохраняет файлы в локальную папку для дальнейшего использования
"""

import os
import requests
from tqdm import tqdm
import time
from urllib.parse import urlparse

def load_urls_from_file(file_path):
    """Загрузка URL из файла urls_normalized.tsv"""
    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('http'):
                urls.append(line)
    return urls

def download_audio_file(url, save_path, timeout=30):
    """
    Скачивание одного аудиофайла
    
    Args:
        url: URL аудиофайла
        save_path: Путь для сохранения
        timeout: Таймаут в секундах
        
    Returns:
        bool: True если успешно, False иначе
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Скачиваем файл
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return False

def get_filename_from_url(url, index):
    """
    Генерирует имя файла из URL или индекса
    
    Args:
        url: URL файла
        index: Индекс файла
        
    Returns:
        str: Имя файла
    """
    # Пытаемся извлечь имя файла из URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Если имя файла пустое или не имеет расширения, используем индекс
    if not filename or '.' not in filename:
        filename = f"audio_{index:03d}.wav"
    
    return filename

def download_all_audio_files(urls_file="urls_normalized.tsv", 
                           output_dir="audio_files", 
                           max_files=None,
                           delay=0.5):
    """
    Скачивает все аудиофайлы из файла
    
    Args:
        urls_file: Путь к файлу с URL
        output_dir: Папка для сохранения файлов
        max_files: Максимальное количество файлов для скачивания (None = все)
        delay: Задержка между запросами в секундах
    """
    
    # Загружаем URL
    print(f"Загружаем URL из файла {urls_file}...")
    urls = load_urls_from_file(urls_file)
    
    if not urls:
        print("Не удалось загрузить URL из файла!")
        return
    
    print(f"Найдено {len(urls)} URL")
    
    # Ограничиваем количество файлов если нужно
    if max_files and max_files < len(urls):
        urls = urls[:max_files]
        print(f"Будем скачивать первые {max_files} файлов")
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    print(f"Создана папка: {output_dir}")
    
    # Статистика
    successful = 0
    failed = 0
    
    print(f"\nНачинаем скачивание {len(urls)} файлов...")
    print("=" * 60)
    
    # Скачиваем файлы
    for i, url in enumerate(tqdm(urls, desc="Скачивание")):
        # Генерируем имя файла
        filename = get_filename_from_url(url, i + 1)
        save_path = os.path.join(output_dir, filename)
        
        # Проверяем, не существует ли уже файл
        if os.path.exists(save_path):
            print(f"Файл {filename} уже существует, пропускаем")
            successful += 1
            continue
        
        # Скачиваем файл
        print(f"\n[{i+1}/{len(urls)}] Скачиваем: {filename}")
        print(f"URL: {url}")
        
        if download_audio_file(url, save_path):
            successful += 1
            print(f"✅ Успешно сохранен: {save_path}")
        else:
            failed += 1
            print(f"❌ Ошибка при скачивании")
        
        # Задержка между запросами
        if delay > 0 and i < len(urls) - 1:
            time.sleep(delay)
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего файлов: {len(urls)}")
    print(f"Успешно скачано: {successful}")
    print(f"Ошибок: {failed}")
    print(f"Процент успеха: {successful/len(urls)*100:.1f}%")
    print(f"Файлы сохранены в папку: {os.path.abspath(output_dir)}")

def main():
    """Основная функция"""
    print("🎵 Скачивание аудиофайлов из urls_normalized.tsv")
    print("=" * 60)
    
    # Параметры скачивания
    urls_file = "urls_normalized.tsv"
    output_dir = "audio_files"
    
    # Проверяем наличие файла с URL
    if not os.path.exists(urls_file):
        print(f"❌ Файл {urls_file} не найден!")
        print("Убедитесь, что файл находится в той же директории.")
        return
    
    # Спрашиваем пользователя о количестве файлов
    print(f"Файл {urls_file} найден.")
    
    try:
        choice = input("\nСкачать все файлы? (y/n): ").lower().strip()
        if choice == 'n':
            max_files = int(input("Сколько файлов скачать? "))
        else:
            max_files = None
    except (ValueError, KeyboardInterrupt):
        print("Скачивание отменено.")
        return
    
    # Начинаем скачивание
    download_all_audio_files(
        urls_file=urls_file,
        output_dir=output_dir,
        max_files=max_files,
        delay=0.5  # Задержка 0.5 секунды между запросами
    )

if __name__ == "__main__":
    main()

