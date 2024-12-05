# Настройка GigaChat с LiteLLM

Руководство по установке и настройке прокси LiteLLM для работы с GigaChat API.

## Предварительные требования

1) Устанавливаем питон меньше 3.13(например 3.12.7) + pip
2) Устанавливаем виртуальное окружение (.venv)
3) Активируем его:
$ source crewai-env/Scripts/activate
4) Устанавливаем crewai
$ pip install crewai
5) Создаем новый проект или скачиваем существующий
$ crewai create crew crewai_gigachat
В выборе ставим 10 - другое
6) Устанавливаем litellm
$ pip install litellm
7) Устанавливаем прокси
pip install 'litellm[proxy]'
8) подкладываем конфиги и настраиваем переменные .env
Настройка переменных окружения
Создайте файл .env в корневой директории проекта
Добавьте ваши учетные данные GigaChat в формате Base64
Убедитесь, что файл .env добавлен в .gitignore
9) Запуск: litellm --config config.yaml