FROM python:3.12-slim

ADD . /app/litellm

WORKDIR /app/litellm

RUN pip3.12 install litellm && pip3.12 install 'litellm[proxy]'

CMD litellm --config ift_config_gigachat.yaml