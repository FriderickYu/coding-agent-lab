# Small sandbox image that only runs pytest against a mounted, read-only workspace.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install "pytest==8.3.5" \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin runner

USER runner
WORKDIR /workspace

CMD ["pytest", "-q", "-p", "no:cacheprovider"]
