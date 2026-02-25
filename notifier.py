import requests
import cv2
import os

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/"

    def send_message(self, text):
        url = self.api_url + "sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            r = requests.post(url, data=payload, timeout=10)
            if not r.ok:
                print(f"[Telegram] Error: {r.status_code} - {r.text}")
            else:
                print(f"[Telegram] Message sent successfully.")
        except Exception as e:
            print(f"[Telegram] Exception sending message: {e}")

    def send_photo(self, frame, caption=""):
        # 一時ファイルとして保存して送信
        photo_path = "detected.jpg"
        cv2.imwrite(photo_path, frame)
        
        url = self.api_url + "sendPhoto"
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            payload = {"chat_id": self.chat_id, "caption": caption}
            try:
                r = requests.post(url, data=payload, files=files, timeout=15)
                if not r.ok:
                    print(f"[Telegram] Error (Photo): {r.status_code} - {r.text}")
                else:
                    print(f"[Telegram] Photo sent successfully.")
            except Exception as e:
                print(f"[Telegram] Exception sending photo: {e}")
        
        if os.path.exists(photo_path):
            os.remove(photo_path)
