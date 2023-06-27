# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim
EXPOSE 8080
ADD requirements.txt requirements.txt
# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . ./
# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
# ENV APP_HOME /app
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# CMD exec streamlit run Main.py --server.port $PORT

ENTRYPOINT ["streamlit", "run", "Main.py", "--server.port=8080", "--server.address=0.0.0.0"]