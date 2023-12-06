"""DONE: File currently handles audio capture (thread), detecting silence (thread), invokes chatgpt directly if silent, speaker diarization using azure (thread) handles Hey Rachel, getting chatgpt responses (thread)
TODO: We need: two more threads - one thread to capture video frames, other thread to run the prediction model
VOTING SYSTEM
"""


import sys
from typing import Optional
import json
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from interview_assistant import InterviewAssistant
import zmq
import time
import wave
from zmq_utils import *
import time
import azure.cognitiveservices.speech as speechsdk
import threading
from datetime import datetime, timedelta
from typing import List, Dict
import re
import numpy as np
from dotenv import dotenv_values
import argparse
from collections import defaultdict

config = dotenv_values(".env")

sys.path.append("/usr0/home/sohamdit/Jetson/video_scripts/")
# import send_video_dict_with_embed


# UTILS TIME
def generate_current_dotnet_datetime_ticks(base_time=datetime(1, 1, 1)):
    x = datetime.utcnow()
    return x, (x - base_time) / timedelta(microseconds=1) * 1e1


# System Config
# User Initiated
user_invocation_string = "hey rachel"
PERSON_WHO_USER_INVOK = ""
PERSONS_LAST_NUM_RESPONSES_WHEN_INVOK = -1

# CHATGPT LANGCHAIN CONFIG
chatgpt_currently_invoked = False
# asked if help needed
asked_if_help_needed = False
help_needed = False
num_transcripts_when_asked_help_needed = 0
num_transcripts_to_wait_whether_help_needed = 3


chatgpt_last_response_time = datetime.utcnow()
chatgpt_response_interval = timedelta(seconds=20)

OPENAI_KEY = config["OPENAI_KEY"]
MODEL_NAME = "gpt-3.5-turbo-1106"
Q_NO = 0
with open("questions.json", "r") as file:
    question_bank = json.load(file)
interview_agent = InterviewAssistant(
    coding_question=question_bank["question_list"][Q_NO],
    code_solution=question_bank["solution_list"][Q_NO],
    api_key=OPENAI_KEY,
)

# user code
user_code = ""


def get_response(
    current_transcript: str,
    current_code: str,
    question: Optional[str] = None,
    direct_q_flag: bool = False,
):
    global interview_agent
    return interview_agent(
        current_code=current_code,
        current_transcript=current_transcript,
        question=question,
        direct_question_flg=direct_q_flag,
    )


chatgpt_resp_pub_socket = create_socket(ip_address="tcp://*:50001")


# SPEAKER DIARIZATION
time_starting = datetime.now()
speaker_history: List[Dict[str, str]] = []
speaker_last_spoken = defaultdict(lambda: {"last_ts": "", "list_utt_ind": []})
speaker_history_f = open(
    f"speaker_history_{time_starting.strftime('%m-%d-%Y_%H:%M:%S')}.txt", "w"
)
silent_speakers = []


# CHATGPT FLAGS
FLAG_CV_PRED = False
FLAG_USER_INVOK = False
FLAG_USER_INVOK_TIME = None
FLAG_USER_INVOK_TIME_DELTA = 5 # timedelta(seconds=5)
FLAG_SILENCE_DET = False

# Keep Direct Questions
DIRECT_Q = ""

# TTS Invocaton WARNING Not being used
tts_invoked_last = datetime.now() - timedelta(hours=1)
tts_invoked_waiting_for_new_spk_id = False

def add_and_send_rachel_response(response: str, utc_time_arrived, ticks_time_arrived) -> None:
    global speaker_history
    if len(response) > 0:
        # response = json.dumps(response)
        response = response.replace('\n', '\\n')
        response = response.replace('\"', '\\"')

        assert type(response) == str
        speaker_history.append(
                {
                    "speaker_id": "Rachel",
                    "text": response,
                    "duration": None,
                    "utc_time_arrived": utc_time_arrived,
                    "ticks_time_arrived": ticks_time_arrived,
                }
            )
        
        send_payload(chatgpt_resp_pub_socket, "chatgpt-responses", response)
        print(f"Chatgpt: {response}")
        print(f"." * 100)
    else:
        print(f"Empty response")

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def invoke_chatgpt():
    global chatgpt_currently_invoked, chatgpt_last_response_time, speaker_history, FLAG_SILENCE_DET, FLAG_USER_INVOK, FLAG_USER_INVOK_TIME, FLAG_CV_PRED, asked_if_help_needed, help_needed, DIRECT_Q, num_transcripts_when_asked_help_needed, num_transcripts_to_wait_whether_help_needed, user_code, PERSON_WHO_USER_INVOK
    response = ""

    flag_ask_for_help_first = not (FLAG_USER_INVOK or  asked_if_help_needed)

    if flag_ask_for_help_first:
        # first ask user if help is needed
        response = "Do you need help?"
        asked_if_help_needed = True
        num_transcripts_when_asked_help_needed = len(speaker_history)
    else:
        # user has been asked, now check for response, and disable all invok flags
        
        help_needed = False
        search_strings = []
        for txt in speaker_history[num_transcripts_when_asked_help_needed:]:
            search_strings += re.findall( r'\w+|[^\s\w]+', txt['text'].lower())
        if FLAG_USER_INVOK or "yes" in search_strings:
            help_needed = True
            print(f"Setting asked_if_help_needed to false cause user said YES")
            asked_if_help_needed = False # resetting
        elif "no" in search_strings:
            print(f"Setting asked_if_help_needed to false cause user said NO")
            asked_if_help_needed = False # resetting
        elif (len(speaker_history) >= num_transcripts_to_wait_whether_help_needed + num_transcripts_when_asked_help_needed):
            print(f"Setting asked_if_help_needed to false cause user DID NOT RESPOND")
            asked_if_help_needed = False # resetting



        if help_needed:
            print("!"*20 + f"HELP NEEDED")
            chatgpt_currently_invoked = True

            conversation_string = ""
            for utterance in speaker_history:
                conversation_string += f"{utterance['speaker_id']}: {utterance['text']}\n"

            if not FLAG_USER_INVOK:
                response = get_response(
                    current_transcript=conversation_string,
                    current_code=user_code,
                )
            else:
                DIRECT_Q = " ".join([speaker_history[i]['text'] for i in speaker_last_spoken[PERSON_WHO_USER_INVOK]['list_utt_ind'][-2:]])
                try:
                    ind = DIRECT_Q.lower().rindex(user_invocation_string)
                    DIRECT_Q = DIRECT_Q[ind:]
                except:
                    print(f"ERROR: user invocation string not present in current context window.")
                print(f"Direct Question: {DIRECT_Q}")
                response = get_response(
                    current_transcript=conversation_string,
                    current_code=user_code,
                    direct_q_flag=True,
                    question=DIRECT_Q,
                )
        else:
            response = ""

    (utc_time_arrived, ticks_time_arrived) = generate_current_dotnet_datetime_ticks()
    # if not flag_ask_for_help_first:
        # when asking for help it is not really giving away any hint/help
    chatgpt_last_response_time = utc_time_arrived

    chatgpt_currently_invoked = False
    FLAG_CV_PRED = False
    FLAG_USER_INVOK = False
    PERSON_WHO_USER_INVOK = ""
    FLAG_USER_INVOK_TIME = None
    FLAG_SILENCE_DET = False

    return response, utc_time_arrived, ticks_time_arrived



# Speaker Diarization ---------------------------------------------------------------------------------------------------------------
def push_stream_writer(stream, topic: str, psi_port=40003):
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

def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print("Canceled event")


def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print("SessionStopped event")


def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    global tts_invoked_waiting_for_new_spk_id, FLAG_USER_INVOK, PERSONS_LAST_NUM_RESPONSES_WHEN_INVOK
    print("TRANSCRIBED:")
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text
        speaker_id = evt.result.speaker_id
        duration = evt.result.duration
        utc_time_arrived, ticks_time_arrived = generate_current_dotnet_datetime_ticks()
        print("\tText={}".format(text))
        print("\tSpeaker ID={}".format(speaker_id))
        temp = {
            "speaker_id": speaker_id,
            "text": text,
            "duration": duration,
            "utc_time_arrived": utc_time_arrived,
            "ticks_time_arrived": ticks_time_arrived,
        }

        if speaker_id == "Guest-1":
            # this is probably rachel due to her long intro sentence
            print(f"Not adding {speaker_id}'s last text, its probably Rachel.")
            return

        if speaker_id != "Unknown":
            speaker_last_spoken[speaker_id]["last_ts"] = utc_time_arrived
            speaker_last_spoken[speaker_id]["list_utt_ind"].append(len(speaker_history))


        # print(f"line 131 ", tts_invoked_waiting_for_new_spk_id)
        # if tts_invoked_waiting_for_new_spk_id:
        #     if speaker_id not in speaker_last_spoken:
        #         # new speaker id probably rachel
        #         tts_invoked_waiting_for_new_spk_id = False
        #         print(f"Not appending text from spker id: {speaker_id} to history")
        #         return
        speaker_history.append(temp)
        speaker_history_f.write(
            f"{speaker_id}|{text}|{duration}|{utc_time_arrived}|{ticks_time_arrived}\n"
        )
        speaker_history_f.flush()
        if user_invocation_string in re.sub(r"[^\w\s]", "", text.lower()):
            global FLAG_USER_INVOK, FLAG_USER_INVOK_TIME, PERSON_WHO_USER_INVOK
            PERSON_WHO_USER_INVOK=speaker_id
            FLAG_USER_INVOK = True
            PERSONS_LAST_NUM_RESPONSES_WHEN_INVOK = len(speaker_history[speaker_id]['list_utt_ind'])
            FLAG_USER_INVOK_TIME = datetime.utcnow()
            print(f"Rachel invoked", FLAG_USER_INVOK, FLAG_USER_INVOK_TIME, PERSONS_LAST_NUM_RESPONSES_WHEN_INVOK)
            # TODO: Extract direct question here
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print(
            "\tNOMATCH: Speech could not be TRANSCRIBED: {}".format(
                evt.result.no_match_details
            )
        )


def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print("SessionStarted event")


def diarize_from_stream(topic: str, psi_port: int):
    """gives an example how to use a push audio stream to diarize speech from a custom audio
    source"""
    AzureSubscriptionKey = "165ea78f5c7f44bd9d31f07d0f319cc7"
    AzureRegion = "eastus"
    speech_config = speechsdk.SpeechConfig(
        subscription=AzureSubscriptionKey, region=AzureRegion
    )
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "./out.log")
    speech_config.speech_recognition_language = "en-US"

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    # audio_config = speechsdk.audio.AudioConfig(stream=None)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config, audio_config=audio_config
    )
    recognition_done = threading.Event()

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        # """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print("CLOSING on {}".format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True
        recognition_done.set()

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(
        conversation_transcriber_transcribed_cb
    )
    conversation_transcriber.session_started.connect(
        conversation_transcriber_session_started_cb
    )
    conversation_transcriber.session_stopped.connect(
        conversation_transcriber_session_stopped_cb
    )
    conversation_transcriber.canceled.connect(
        conversation_transcriber_recognition_canceled_cb
    )
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # start push stream writer thread
    push_stream_writer_thread = threading.Thread(
        target=push_stream_writer, args=[stream, topic, psi_port]
    )
    push_stream_writer_thread.start()

    check_every_second_for_silent_thread = threading.Thread(
        target=check_every_second_for_silent
    )
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



# Threads for diffeent signals ---------------------------------------------------------------------------------------------------------------
def check_every_second_for_silent(delta_silence=timedelta(seconds=20)):
    """After every second, checking if any user has become silent (and remove users who are not silent)

    Args:
        delta_silence (timedelta, optional): Amount of time passed since a user last spoke for them to be classified as silent. Defaults to timedelta(seconds=20).
    """    
    global silent_speakers
    while True:
        silent_speakers = []
        time_now = datetime.utcnow()
        for speaker, dict_info in speaker_last_spoken.items():
            delta = time_now - dict_info['last_ts']
            if delta > delta_silence:
                silent_speakers.append(speaker)
        print(f"Speakers silent at {time_now}: {silent_speakers}")
        if len(silent_speakers) > 0:
            global FLAG_SILENCE_DET
            FLAG_SILENCE_DET = True
        time.sleep(1)


def receive_cv_preds_from_psi(topic: str, psi_port=40003):
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    try:
        while True:
            frames, originatingTime = readFrame(sub_socket_to_psi)
            emotions = np.frombuffer(frames)
            global FLAG_CV_PRED
            if len(speaker_last_spoken) > 0:
                FLAG_CV_PRED = emotions[0] != 0
            # if FLAG_CV_PRED:
            #     print(f"FLAG_CV_PRED", FLAG_CV_PRED, emotions)
    finally:
        sub_socket_to_psi.close()


def receive_tts_inv_from_psi(topic: str, psi_port=40006):
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    global tts_invoked_last, tts_invoked_waiting_for_new_spk_id, chatgpt_resp_pub_socket
    ctr = 0
    try:
        while True:
            print(f"waiting from fe/python")
            frames, originatingTime = readFrame(sub_socket_to_psi)
            ctr += 1
            tts_invoked = json.loads(frames)["tts_invoked"]
            print(f"received from fe", frames, tts_invoked)
            if tts_invoked:
                tts_invoked_last = convert_ticks_to_timestamp(originatingTime)
                tts_invoked_waiting_for_new_spk_id = True
            else:
                question = question_bank['question_list'][Q_NO].replace('\n', '\\n').replace('\"', '\\"')
                question = f"Question: {question}"
                send_payload(chatgpt_resp_pub_socket, "chatgpt-responses", question)
    finally:
        sub_socket_to_psi.close()


def send_question_to_psi(topic: str, psi_port=40006):
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    # global tts_invoked_last, tts_invoked_waiting_for_new_spk_id
    global user_code
    try:
        while True:
            print(f"waiting for code from fe/python")
            user_code, originatingTime = readFrame(sub_socket_to_psi)
            print(f"received code from fe", user_code)
    finally:
        sub_socket_to_psi.close()

def receive_code_from_psi(topic: str, psi_port=40006):
    sub_socket_to_psi = create_sub_socket(ip_address=f"tcp://localhost:{psi_port}")
    sub_socket_to_psi.setsockopt_string(zmq.SUBSCRIBE, topic)
    # global tts_invoked_last, tts_invoked_waiting_for_new_spk_id
    global user_code
    try:
        while True:
            print(f"waiting for code from fe/python")
            user_code, originatingTime = readFrame(sub_socket_to_psi)
            print(f"received code from fe", user_code)
    finally:
        sub_socket_to_psi.close()

def monitor_flags_and_invoke_chatgpt():
    global chatgpt_currently_invoked, chatgpt_last_response_time, speaker_history, FLAG_SILENCE_DET, FLAG_USER_INVOK, FLAG_USER_INVOK_TIME, FLAG_CV_PRED, asked_if_help_needed, help_needed, PERSONS_LAST_NUM_RESPONSES_WHEN_INVOK

    while True:
        try:
            time_now = datetime.utcnow()
            override_time_check = FLAG_USER_INVOK
            if FLAG_SILENCE_DET or FLAG_USER_INVOK or FLAG_CV_PRED:
                print(f"asked_if_help_needed", asked_if_help_needed)
                if (override_time_check or asked_if_help_needed or (time_now - chatgpt_last_response_time) >= chatgpt_response_interval) and not chatgpt_currently_invoked:
                    if FLAG_USER_INVOK:
                        # waiting for user question to get logged into conversation history
                        time.sleep(FLAG_USER_INVOK_TIME_DELTA)
                        response, utc_time_arrived, time_ticks = invoke_chatgpt()
                    else:
                        # 1. Check if help needed
                        # 2. invoke if needed
                        # if not asked_if_help_needed:
                        #     # first ask user if help is needed
                        #     response = "Do you need help?"
                        #     asked_if_help_needed = True
                        # pass
                        response, utc_time_arrived, time_ticks = invoke_chatgpt()

                    add_and_send_rachel_response(response, utc_time_arrived, time_ticks)
                else:
                    print(f"Wait for atleast {chatgpt_response_interval - (time_now - chatgpt_last_response_time)} before invoking again.")

        except Exception as e:
            raise e
            # print(e)
            # FLAG_SILENCE_DET = False
            # FLAG_USER_INVOK = False
            # FLAG_USER_INVOK_TIME = None
            # PERSON_WHO_USER_INVOK = ""
            # FLAG_CV_PRED = False
            # chatgpt_currently_invoked = False

        time.sleep(1)
        
class Application:
    def __init__(self) -> None:
        pass

    def start(self):
        """Start all the parallel threads
        """        
        try:
            # send selected question
            # time.sleep(20)
            receive_cv_preds_thread = threading.Thread(
                target=receive_cv_preds_from_psi, args=["cv-preds-psi-to-python", 40005]
            )
            receive_cv_preds_thread.start()

            receive_tts_inv_thread = threading.Thread(
                target=receive_tts_inv_from_psi, args=["tts-inv-psi-to-python", 40006]
            )
            receive_tts_inv_thread.start()

            receive_code_thread = threading.Thread(
                target=receive_code_from_psi, args=["fe-code-psi-to-python", 40007]
            )
            receive_code_thread.start()

            check_flags_and_invoke_thread = threading.Thread(
                target=monitor_flags_and_invoke_chatgpt
            )
            check_flags_and_invoke_thread.start()

            diarize_from_stream(topic=f"audio-psi-to-python", psi_port=40003)
            print(f"Started diarization service")

            receive_cv_preds_thread.join()
            receive_tts_inv_thread.join()
            receive_code_thread.join()
            check_flags_and_invoke_thread.join()

            # send_video_dict_with_embed.main()
        except Exception as err:
            print("Encountered exception. {}".format(err))

# Main
if __name__ == "__main__":
    # TODO: setup logging control
    app = Application()
    app.start()
    
