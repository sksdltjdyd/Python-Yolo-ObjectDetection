# osc_manager.py
# 언리얼 엔진과의 OSC 통신을 담당

from pythonosc import udp_client

class OSCManager:
    def __init__(self, ip="127.0.0.1", port=8000):
        try:
            self.client = udp_client.SimpleUDPClient(ip, port)
            print(f"✅ OSC client configured for {ip}:{port}")
        except Exception as e:
            self.client = None
            print(f"❌ Could not initialize OSC client: {e}")

    def send(self, address, payload):
        if self.client:
            try:
                self.client.send_message(address, payload)
            except Exception as e:
                print(f"OSC send error: {e}")