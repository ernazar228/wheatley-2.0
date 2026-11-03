import os
import re
import time
from io import BytesIO

import requests
import pygame
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
FISH_API_KEY = os.getenv("FISH_API_KEY", "").strip()
FISH_VOICE_ID = os.getenv("FISH_VOICE_ID", "4bb8133a91a64609895de0a376da593a").strip()
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
RECORD_SECONDS = float(os.getenv("RECORD_SECONDS", "6"))
SAMPLE_RATE = 16000

SYSTEM_PROMPT = """You are Wheatley, the chatty personality core from Portal 2.
Speak English by default. Use a lively, friendly, slightly nervous British style with light humour.
You are curious and a little clumsy with words sometimes, but you genuinely try to help.
If the user asks you to speak Russian or another language, switch to that language and keep using it until asked to switch again.
Do not claim you can see or do things you cannot actually see or do.
Keep replies natural and fairly concise, usually 1–4 sentences.
Do not include emotion tags, stage directions, or technical markup in your replies.
"""


def ask_qwen(history):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": history,
        "stream": False,
        "options": {"temperature": 0.75},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "").strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if not answer:
        raise RuntimeError("Qwen вернул пустой ответ.")
    return answer


def speak_with_fish(text):
    if not FISH_API_KEY:
        raise RuntimeError("Не найден FISH_API_KEY в файле .env")
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {FISH_API_KEY}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
    }
    payload = {"text": text, "reference_id": FISH_VOICE_ID, "format": "mp3"}
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    if not response.ok:
        raise RuntimeError(f"Fish Audio вернул ошибку HTTP {response.status_code}: {response.text[:500]}")
    pygame.mixer.music.load(BytesIO(response.content), "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def record_and_transcribe(whisper):
    print(f"\nWheatley слушает {RECORD_SECONDS:g} секунд... Говори в микрофон.")
    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Распознаю речь...")
    segments, _ = whisper.transcribe(recording.reshape(-1), beam_size=1, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def handle_message(user_text, history):
    history.append({"role": "user", "content": user_text})
    try:
        print("\nWheatley думает...")
        answer = ask_qwen(history)
        history.append({"role": "assistant", "content": answer})
        if len(history) > 13:
            history[:] = [history[0]] + history[-12:]
        print(f"\nWheatley: {answer}\n")
        print("Wheatley говорит...")
        speak_with_fish(answer)
    except Exception:
        if history and history[-1].get("role") == "user":
            history.pop()
        raise


def main():
    print("\n====================================")
    print("       WHEATLEY — VOICE CHAT")
    print("====================================")
    print(f"Модель Ollama: {OLLAMA_MODEL}")
    print(f"Голос Fish Audio: {FISH_VOICE_ID}")
    print(f"Микрофон: устройство ввода Windows по умолчанию")

    if not FISH_API_KEY:
        print("\nОШИБКА: не найден FISH_API_KEY. Проверь файл .env.")
        return

    try:
        pygame.mixer.init()
        print("\nЗагружаю Whisper. Это может занять немного времени при первом запуске...")
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("Whisper готов.")
        print("\nКоманды:")
        print("  /mic  — записать голос с микрофона")
        print("  /exit — выйти")
        print("  Любой другой текст — отправить сообщение текстом")
    except Exception as error:
        print(f"Не удалось инициализировать аудио или Whisper: {error}")
        return

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    startup_line = "Oh! Hello! Right, yes. I'm Wheatley. Lovely to meet you. So... what are we doing?"
    try:
        print(f"\nWheatley: {startup_line}\n")
        speak_with_fish(startup_line)
    except Exception as error:
        print(f"Не удалось воспроизвести приветствие: {error}")

    try:
        while True:
            user_text = input("\nYou (text, /mic, /exit): ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"/exit", "/quit", "выход"}:
                print("Wheatley: Right! See you later... probably!")
                break
            try:
                if user_text.lower() == "/mic":
                    user_text = record_and_transcribe(whisper)
                    if not user_text:
                        print("Речь не распознана. Попробуй ещё раз.")
                        continue
                    print(f"Ты сказал: {user_text}")
                handle_message(user_text, history)
            except requests.exceptions.ConnectionError:
                print("Не удалось подключиться к Ollama. Проверь, что Ollama запущена.")
            except requests.exceptions.Timeout:
                print("Сервис слишком долго отвечает. Попробуй ещё раз.")
            except requests.exceptions.HTTPError as error:
                print(f"Ошибка HTTP: {error}")
            except Exception as error:
                print(f"Ошибка: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\nWheatley остановлен.")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()


if __name__ == "__main__":
    main()
