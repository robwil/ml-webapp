FROM tiangolo/uvicorn-gunicorn-starlette:python3.7

RUN pip install onnxruntime aiohttp python-multipart pillow scikit-image

COPY ./app /app

