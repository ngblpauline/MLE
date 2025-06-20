FROM apache/airflow:2.6.2-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive

# system-level deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless procps bash libgeos-c1v5 libgeos-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-17-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# drop back to airflow user for Python installs
RUN chown -R airflow: /opt/airflow
USER airflow

WORKDIR /app

COPY requirements.txt ./

# 1) Install NumPy 1.x + Pandas 2.x *as airflow* so the wheels match
RUN pip install --no-cache-dir \
      numpy==1.23.5 \
      "pandas>=2.0.0,<2.1.0"

# 2) Then install the rest of your requirements
RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/app"]
