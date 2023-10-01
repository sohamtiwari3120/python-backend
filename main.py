"""DONE: File currently handles audio capture (thread), detecting silence (thread), invokes chatgpt directly if silent, speaker diarization using azure (thread) handles Hey Rachel, getting chatgpt responses (thread)
TODO: We need: two more threads - one thread to capture video frames, other thread to run the prediction model
VOTING SYSTEM
"""


from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import zmq, msgpack, time, wave
from zmq_utils import *
import time
import azure.cognitiveservices.speech as speechsdk
import threading
from datetime import datetime, timedelta
from typing import List, Dict
import re

# User Initiated
user_invocation_string = "hey rachel"

# CHATGPT LANGCHAIN CONFIG
prompt_template = """You are Rachel, an AI Teaching Assistant. You are given conversation between students who are working on solving a problem. If no explicit question asked of you, then infer the question worked on from the conversation.  Then, give the students a hint to help them solve the question. They have been silent and thinking for a while now, but did not make any progress. Do not state the answer explicitly. Keep the hint subtle. The students should be able to solve the question on their own after getting the hint. Give an example if possible. If they get the answer, congratulate and confirm the same. 
Question:
```{question}```
Conversation:
```{conversation}```
AI:```
"""
chatgpt_last_response_time = datetime.utcnow()
chatgpt_response_interval = timedelta(seconds=30)
PROMPT = PromptTemplate(
  template=prompt_template, input_variables=["question", "conversation"]
)
OPENAI_KEY = "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"
MODEL_NAME = "gpt-3.5-turbo"
LLM_BIG = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=OPENAI_KEY, temperature=0)
CHAIN = LLMChain(llm=LLM_BIG, prompt=PROMPT, verbose=False)
chatgpt_currently_invoked = False

def get_response(conversation, question):
  return CHAIN.run(question=question, conversation=conversation)

chatgpt_resp_pub_socket = create_socket(ip_address='tcp://*:50001')


# SPEAKER DIARIZATION
time_starting = datetime.now()
speaker_history = []
speaker_last_spoken = {}
speaker_history_f = open(f"speaker_history_{time_starting.strftime('%m-%d-%Y_%H:%M:%S')}.txt", "w")
silent_speakers = []

def invoke_chatgpt(override_time_check=False):
    print(f"Invoking chatgpt")
    global chatgpt_currently_invoked, chatgpt_last_response_time, speaker_history
    if chatgpt_currently_invoked:
        print(f"One invocation of ChatGPT already running.")
        return 
    chatgpt_currently_invoked = True
    conversation_string = ""
    for utterance in speaker_history:
        conversation_string += f"[{utterance['utc_time_arrived']}] {utterance['speaker_id']}: {utterance['text']}\n"
    question = f""
    print("\t\t" + f"{conversation_string}")
    print("\t\t" + f"{question}")
    response = get_response(conversation=conversation_string, question=question)
    utc_time_arrived, ticks_time_arrived = generate_current_dotnet_datetime_ticks()
    chatgpt_last_response_time = utc_time_arrived
    speaker_history.append({
        "speaker_id":"ChatGPT", "text":response, "duration":None, "utc_time_arrived":utc_time_arrived, "ticks_time_arrived":ticks_time_arrived
    })
    send_payload(chatgpt_resp_pub_socket, "chatgpt-responses", response)
    chatgpt_currently_invoked = False
    print(f"Chatgpt: {response}")
    print(f"."*100)

def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def generate_current_dotnet_datetime_ticks(base_time = datetime(1, 1, 1)):
    x=datetime.utcnow()
    return x, (x - base_time)/timedelta(microseconds=1) * 1e1

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    print('TRANSCRIBED:')
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text
        speaker_id = evt.result.speaker_id
        duration = evt.result.duration
        utc_time_arrived, ticks_time_arrived = generate_current_dotnet_datetime_ticks()
        print('\tText={}'.format(text))
        print('\tSpeaker ID={}'.format(speaker_id))
        temp = {
            "speaker_id":speaker_id, "text":text, "duration":duration, "utc_time_arrived":utc_time_arrived, "ticks_time_arrived":ticks_time_arrived
        }
        if speaker_id != "Unknown":
            speaker_last_spoken[speaker_id] = utc_time_arrived
        speaker_history.append(temp)
        speaker_history_f.write(f"{speaker_id}|{text}|{duration}|{utc_time_arrived}|{ticks_time_arrived}\n")
        speaker_history_f.flush()
        if user_invocation_string in re.sub(r'[^\w\s]', '', text.lower()):
            invoke_chatgpt()
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

def diarize_from_stream(topic: str, psi_port:int):
    """gives an example how to use a push audio stream to diarize speech from a custom audio
    source"""
    AzureSubscriptionKey = "165ea78f5c7f44bd9d31f07d0f319cc7"
    AzureRegion = "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=AzureSubscriptionKey, region=AzureRegion)
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "./out.log")
    speech_config.speech_recognition_language="en-US"

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    # audio_config = speechsdk.audio.AudioConfig(stream=None)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        #"""callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True
        recognition_done.set()

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # start push stream writer thread
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream, topic, psi_port])
    push_stream_writer_thread.start()

    check_every_second_for_silent_thread = threading.Thread(target=check_every_second_for_silent)
    check_every_second_for_silent_thread.start()

    # start continuous speech recognition
    conversation_transcriber.start_transcribing_async()

    # wait until all input processed

    # stop recognition and clean up

    # Waits for completion.
    # while not transcribing_stop:
    #     time.sleep(.5)

    recognition_done.wait()

    conversation_transcriber.stop_transcribing_async()
    push_stream_writer_thread.join()
    print(f"TERMINATED Push stream thread")
    check_every_second_for_silent_thread.join()
    print(f"TERMINATED Check every second for silence thread")

def check_every_second_for_silent(delta_silence=timedelta(seconds=60)):
    global silent_speakers
    while True:
        silent_speakers = []
        time_now = datetime.utcnow()
        for speaker, last_time_spoken in speaker_last_spoken.items():
            delta = (time_now - last_time_spoken)
            print(delta, delta_silence, delta > delta_silence)
            if delta  > delta_silence:
                silent_speakers.append(speaker)
        print(f"Speakers silent at {time_now}: {silent_speakers}")
        if len(silent_speakers) > 0 and (time_now - chatgpt_last_response_time) > chatgpt_response_interval:
            invoke_chatgpt()
        time.sleep(1)

def push_stream_writer(stream, topic:str, psi_port=40003):
    # The number of bytes to push per buffer
    # n_bytes = 3200
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    
    # start pushing data until all data has been read from the file
    try:
        while True:
            frames, originatingTime = readFrame(sub_socket_to_psi)
            # frames = wav_fh.readframes(n_bytes // 2)
            # print('read {} bytes'.format(len(frames)))
            if not frames:
                break
            stream.write(frames)
            # time.sleep(.1)
    finally:
        sub_socket_to_psi.close()
        stream.close()  # must be done to signal the end of stream


def receive_audio_from_psi(topic:str, psi_port=40003):
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    try:
        with wave.open('myfile.wav', mode='wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            while True:
                frames, originatingTime = readFrame(sub_socket_to_psi)
                # if len(message) !=2 :
                #     print("no audio received")
                #     continue
                # topic, frames = message.split()
                # frames = wav_fh.readframes(n_bytes // 2)
                f.writeframes(frames)
                print('read {} bytes {}'.format(len(frames), originatingTime))
                if not frames:
                    break
                # time.sleep(.1)
    finally:
        sub_socket_to_psi.close()

# Main
if __name__ == "__main__":
    try:
        diarize_from_stream(topic=f"audio-psi-to-python", psi_port=40003)
    except Exception as err:
        print("Encountered exception. {}".format(err))
