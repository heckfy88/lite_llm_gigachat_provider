import litellm
from litellm import CustomLLM, completion, get_llm_provider
import requests
import base64
import uuid
import yaml

from dotenv import load_dotenv
import os

def load_config():
    with open("config_gigachat.yaml", 'r') as file:
        return yaml.safe_load(file)

# Загрузка переменных окружения
load_dotenv()

# Загружаем конфигурацию
config = load_config()

gigachat_config = config.get('gigachat_settings', {})

# Получаем настройки из конфига
AUTH_URL = gigachat_config.get('auth_url')
BASE_URL = gigachat_config.get('base_url')
VERIFY_SSL = gigachat_config.get('verify_ssl', False)
SCOPE = gigachat_config.get('scope', 'GIGACHAT_API_PERS')

# Обновление конфигурации с учетом переменных окружения
CRED64 = os.getenv('GIGACHAT_CREDENTIALS', gigachat_config.get('credentials'))

def get_authorization_key(client_id, client_secret):
    credentials = f"{client_id}:{client_secret}".encode("utf-8")
    return base64.b64encode(credentials).decode("utf-8")

def get_gigachat_token():
    payload = {
        'scope': SCOPE
    }

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {CRED64}'
    }

    try:
        response = requests.post(
            AUTH_URL,
            headers=headers,
            data=payload,
            verify=VERIFY_SSL
        )
        response.raise_for_status()
        return response.json()['access_token']
    except requests.RequestException as e:
        raise Exception(f"Ошибка при получении токена: {e}")

class GigaChatCustomLLM(CustomLLM):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.gigachat_config = self.config.get('gigachat_settings', {})

    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        token = get_gigachat_token()
        messages = kwargs.get("messages", [{"role": "user", "content": "Hello world"}])

        gigachat_payload = {
            'model': 'GigaChat',
            'messages': messages,
            'profanity_check': True,
        }

        response = requests.post(
            f'{BASE_URL}/chat/completions',
            headers={'Authorization': f'Bearer {token}'},
            json=gigachat_payload,
            verify=VERIFY_SSL
        )
        response.raise_for_status()

        gigachat_response = response.json()

        choices = []
        for choice in gigachat_response.get("choices", []):
            choices.append({
                "message": {
                    "role": "assistant",
                    "content": choice["message"]["content"]
                }
            })

        return litellm.ModelResponse(
            created=gigachat_response.get("created", 0),
            model=gigachat_response.get("model", "GigaChat"),
            choices=choices
        )

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return self.completion(*args, **kwargs)

# Создание экземпляра класса GigaChatCustomLLM
gigachat_custom_llm = GigaChatCustomLLM()