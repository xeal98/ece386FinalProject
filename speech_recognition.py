import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time
import requests
import Jetson.GPIO as GPIO  # Allows for Jetson GPIO passthrough to docker and the usage of it in python
from ollama import Client

# Had to look at the documentation to find the right microphone device.


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)

def intializeTheModel() -> pipeline:
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    print("Using model_id {model_id}")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")
    return pipe

def transcribeMicrophone(pipe: Pipeline) -> str:
    print("Recording...")
    audio = record_audio()
    print("Done")

    print("Transcribing...")
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    print("Done")

    print(speech)
    # Not super necessary for testing
    print(f"Transcription took {(end_time-start_time)/1000000000} seconds")

def intializeAIServer(address: str) -> Client:
    """This script evaluates an LLM prompt for processing text so that it can be used for the wttr.in API"""

    # LLM_MODEL: str = model  # Optional, change this to be the model you want
    client: Client = Client(
        host=address  # Optional, change this to be the URL of your LLM
    )
    return client

def llm_parse_for_wttr(input_str:str, client:Client, LLM_MODEL:str) -> str:
    # Give the LLM a post
    # Strip the reponse for the result
    llmResponse = client.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""You are an expert at extracting location information from text and formatting it for use with the *wttr.in* service. Your goal is to accurately identify locations and format them according to specific rules.
 
                    **Instructions:**
                    
                    Given a sentence about the weather, extract the location mentioned and format it as follows:
                    
                    1. **Lowercase:** Convert all locations to lowercase.
                    2. **Airports:** If the location is an airport, use its 3-letter IATA code.  *Always prioritize using the 3-letter code if "airport" or similar terms are present.*
                    3. **Multi-word Locations (Non-Airport):** If the location contains spaces and is *not* an airport, replace each space with a '+'.
                    4. **Landmarks/Regions:** If the location is a landmark, historical site, region, or any place that is *not* a city or airport, prepend a '~' to the output.  This includes places like "Eiffel Tower", "Machu Picchu", and "Golden Gate Bridge".
                    
                    **Examples:**
                    
                    Input: What is the weather in Denver?
                    Output: denver
                    
                    Input: How warm is it at Ronald Reagan International Airport?
                    Output: dca
                    
                    Input: How cold is it going to be in New Mexico?
                    Output: new+mexico
                    
                    Input: Give me the weather of DCA.
                    Output: dca
                    
                    Input: What is the weather at the Statue of Liberty?
                    Output: ~statue+of+liberty
                    
                    Input: Is it going to snow at the Colosseum?
                    Output: ~colosseum
                    
                    Input: What is the weather at Machu Picchu?
                    Output: ~machu+picchu
                    
                    Input: Weather in Denver airport
                    Output: den
                    
                    Input: Give me the weather at colorado springs airport
                    Output: cos
                    
                    **Now, apply these rules to the following sentence:**
                    
                    **{input_str}**""",
            },
        ],
    )

    return llmResponse.message.content

if __name__ == "__main__":
    model = "gemma3:27b"
    llmAddress = "http://ai.dfec.xyz:11434"
    pipe = intializeTheModel()  # Sets up the model in the GPU to keep it hot
    client = intializeAIServer(llmAddress)

    sd.default.device = (
        "USB Audio",
        None,
    )  # Had to add this line to make sure that it picks up the USB microphone

    # Init as digital input
    my_pin = 29
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(my_pin, GPIO.IN)  # digital input

    print("Starting Demo! Move pin 29 between 0V and 3.3V")

    while True:
        GPIO.wait_for_edge(my_pin, GPIO.RISING, bouncetime=1000)
        strToLargerModel = transcribeMicrophone(pipe)
        llmResponse = llm_parse_for_wttr(strToLargerModel, client, model)
        print(requests.get(f"wttr.in/{llmResponse})"))
        