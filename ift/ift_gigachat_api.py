import litellm
from litellm import CustomLLM, completion, get_llm_provider
import requests
import base64
import uuid
import yaml

from dotenv import load_dotenv
import os

def load_config():
    with open("ift_config_gigachat.yaml", 'r') as file:
        return yaml.safe_load(file)

# Загрузка переменных окружения
load_dotenv()

# Загружаем конфигурацию
config = load_config()

gigachat_config = config.get('gigachat_settings', {})

# Получаем настройки из конфига
BASE_URL = gigachat_config.get('base_url')
VERIFY_SSL = gigachat_config.get('verify_ssl', False)
SCOPE = gigachat_config.get('scope', 'GIGACHAT_API_PERS')

CRED64 = os.getenv('GIGACHAT_CREDENTIALS', gigachat_config.get('credentials'))

class GigaChatCustomLLM(CustomLLM):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.gigachat_config = self.config.get('gigachat_settings', {})


    def completion(self, *args, **kwargs) -> litellm.ModelResponse:

        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': CRED64
        }

        messages = kwargs.get("messages", [{"role": "user", "content": "Hello world"}])

        gigachat_payload = {
            'model': 'СhatCode',
            'messages': messages,
            'profanity_check': True,
        }

        response = requests.post(
            f'{BASE_URL}/gigacode',
            json=gigachat_payload,
            verify=VERIFY_SSL,
            headers=headers

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