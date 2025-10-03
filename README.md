# hh
# Установка зависимостей
pip install -r requirements.txt

# Запуск (скрипт читает из stdin, пишет в stdout)
echo {"text": "Full news text here...", "title": "News Title"} | python summarizer.py
