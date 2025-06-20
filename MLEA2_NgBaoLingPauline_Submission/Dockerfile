# Use the official Apache Airflow image
FROM apache/airflow:2.6.2

# Switch to root to install additional packages
USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install Java, bash, procps, and GEOS (and dev headers if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless \
      procps \
      bash \
      libgeos-c1v5 \
      libgeos-dev && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-17-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app

# Copy requirements.txt (which likely includes shapely, google-cloud-bigquery, etc.)
COPY requirements.txt ./

# Before switching back to airflow user, fix permissions on /opt/airflow
RUN chown -R airflow: /opt/airflow

USER airflow
RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app
