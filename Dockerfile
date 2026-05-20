FROM python:3.12.7-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    pip==24.0 \
    setuptools==75.1.0 \
    wheel==0.44.0

RUN pip install --no-cache-dir \
    pytest==8.3.3 \
    pytest-cov==5.0.0 \
    hypothesis==6.112.1

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY tests/ tests/

ENV PYTHONPATH=/app/src

CMD ["pytest"]
