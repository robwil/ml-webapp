FROM tiangolo/uvicorn-gunicorn-starlette:python3.7

RUN pip install fastai aiohttp python-multipart

COPY ./app /app

