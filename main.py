import zmq, msgpack, time, wave
import os
import time
import azure.cognitiveservices.speech as speechsdk
import threading

speaker_history = []
speaker_history_f = open("speaker_history.txt", "w")

def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    print('TRANSCRIBED:')
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text
        speaker_id = evt.result.speaker_id
        print('\tText={}'.format(text))
        print('\tSpeaker ID={}'.format(speaker_id))
        speaker_history.append((speaker_id, text, "11:55am"))
        speaker_history_f.write(f"{speaker_id}, {text}, 11:55 am\n")
        speaker_history_f.flush()
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

def recognize_from_file():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "./out.log")
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(filename="myfile.wav")
    # audio_config = speechsdk.audio.AudioConfig(stream=None)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        #"""callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    # Waits for completion.
    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()

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

# ZMQ UTILS
def create_sub_socket(ip_address:str=''):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ip_address)
    return socket

def readFrame(socket):
    [topic, payload] = socket.recv_multipart()
    message = msgpack.unpackb(payload, raw=True)
    frame = message[b"message"]
    originatingTime = message[b"originatingTime"]
    return (frame, originatingTime)

def push_stream_writer(stream, topic:str, psi_port=30003):
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

def speech_recognition_with_push_stream(topic: str, psi_port:int):
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    AzureSubscriptionKey = "165ea78f5c7f44bd9d31f07d0f319cc7"
    AzureRegion = "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=AzureSubscriptionKey, region=AzureRegion)

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    # Connect callbacks to the events fired by the speech recognizer
    def session_stopped_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('SESSION STOPPED: {}'.format(evt))
        recognition_done.set()

    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    # start push stream writer thread
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream, topic, psi_port])
    push_stream_writer_thread.start()

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    # wait until all input processed
    recognition_done.wait()

    # stop recognition and clean up
    speech_recognizer.stop_continuous_recognition()
    push_stream_writer_thread.join()

def receive_audio_from_psi(topic:str, psi_port=30003):
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
                # print('read {} bytes'.format(len(frames)))
                if not frames:
                    break
                # time.sleep(.1)
    finally:
        sub_socket_to_psi.close()

# Main
if __name__ == "__main__":
    try:
        diarize_from_stream(topic=f"audio-psi-to-python", psi_port=30003)
        # speech_recognition_with_push_stream(topic=f"audio-psi-to-python", psi_port=30003)
        # recognize_from_file()
    except Exception as err:
        print("Encountered exception. {}".format(err))
