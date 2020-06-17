FROM tensorflow/tensorflow

WORKDIR /src
ADD . /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip libsndfile-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "/src/main.py"]