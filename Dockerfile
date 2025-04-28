# Compatible with Jetpack 6.2
FROM nvcr.io/nvidia/pytorch:25.03-py3-igpu
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    gpiod \
    && rm -rf /var/lib/apt/lists/
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
    transformers==4.49.0\
    accelerate==1.5.2 \
    sounddevice \
    && pip install Jetson.GPIO
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
ENV JETSON_MODEL_NAME=JETSON_ORIN_NANO
ENTRYPOINT ["python", "speech_recognition.py"]

# sudo docker run -it --rm --device=/dev/snd --device=/dev/gpiochip0 --runtime=nvidia --ipc=host -v huggingface:/huggingface/ whisper
