import socket

s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
#instead of gethostname - use local IP
s.connect((socket.gethostname(), 1234))

full_msg = ""
while True:
    msg = s.recv(8)
    if len(msg)  <= 0:
        break
    full_msg += msg.decode("utf-8")

print (full_msg) 