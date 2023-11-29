"""
ZeroMQ utility functions
"""

import zmq, datetime, msgpack, time, json


def generate_current_dotnet_datetime_ticks(base_time=datetime.datetime(1, 1, 1)):
    return (
        (datetime.datetime.utcnow() - base_time)
        / datetime.timedelta(microseconds=1)
        * 1e1
    )


def convert_ticks_to_timestamp(inp_ticks, base_time=datetime.datetime(1, 1, 1)):
    return inp_ticks / 1e1 * datetime.timedelta(microseconds=1) + base_time


def send_payload(pub_sock, topic, message, originatingTime=None):
    payload = {}
    payload["message"] = message
    if originatingTime is None:
        originatingTime = generate_current_dotnet_datetime_ticks()
    payload["originatingTime"] = originatingTime
    pub_sock.send_multipart([topic.encode(), msgpack.dumps(payload)])
    return originatingTime


def create_socket(ip_address="tcp://*:40003"):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(ip_address)
    return socket


# ZMQ UTILS
def create_sub_socket(ip_address: str = ""):
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
