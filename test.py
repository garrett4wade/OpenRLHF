import random
import socket

def get_random_port(min_port=1024, max_port=49151):
    while True:
        port = random.randint(min_port, max_port)
        if is_port_available(port):
            return port

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

# Example usage
random_port = get_random_port()
print("Random port:", random_port)