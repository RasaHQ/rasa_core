FROM python:3.6-slim

SHELL ["/bin/bash", "-c"]

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  openssh-client \
  graphviz-dev \
  pkg-config \
  git-core \
  openssl \
  libssl-dev \
  libffi6 \
  libffi-dev \
  libpng-dev \
  curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  mkdir /app

WORKDIR /app

# Copy as early as possible so we can cache ...
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN pip install -e . --no-cache-dir

ENV APP_ROOT=/app
ENV PATH=${APP_ROOT}:${PATH} HOME=${APP_ROOT}
RUN chmod -R u+x ${APP_ROOT} && \
    chgrp -R 0 ${APP_ROOT} && \
    chmod -R g=u ${APP_ROOT}

USER 10001
WORKDIR ${APP_ROOT}

VOLUME ["/app/model", "/app/config", "/app/project"]

EXPOSE 5005

ENTRYPOINT ["./entrypoint.sh"]

CMD ["start", "-d", "./dialogue"]
