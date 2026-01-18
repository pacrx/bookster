FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit==1.37.1 httpx==0.27.2 pyyaml==6.0.2
COPY app.py /app/app.py
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
