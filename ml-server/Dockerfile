FROM python:3.10

COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system 

WORKDIR /app
COPY . /app

EXPOSE 8001

CMD ["python3", "main.py"]
