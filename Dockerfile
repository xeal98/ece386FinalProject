# Compatible with Jetpack 6.2
FROM nvcr.io/nvidia/pytorch:25.03-py3-igpu
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
    transformers==4.49.0\
    accelerate==1.5.2 \
    sounddevice
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
ENTRYPOINT ["python", "speech_recognition.py"]