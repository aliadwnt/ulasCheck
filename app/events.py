from flask_socketio import emit
from app import socketio

@socketio.on("connect")
def on_connect():
    print("🟢 SocketIO client connected")

@socketio.on("disconnect")
def on_disconnect():
    print("🔴 SocketIO client disconnected")
