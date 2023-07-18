FROM python:3.9-slim

# RUN mkdir /app
# EXPOSE 8080
EXPOSE 5000

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app/ .

COPY templates /app/templates

COPY static /app/static
COPY data /app

CMD [ "python", "./main.py" ]
