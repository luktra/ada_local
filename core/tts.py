"""
TTS (Text-to-Speech) module using the native Piper Python package.
Provides streaming sentence-based synthesis with interrupt support.
Uses the native Piper Python package for cross-platform speech synthesis.
"""

import io
import re
import queue
import threading
import requests
from pathlib import Path
from piper import PiperVoice
import numpy as np

import sounddevice as sd

# ANSI colors for console output
GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# HTTP session for downloads
http_session = requests.Session()


class SentenceBuffer:
    """Buffers streaming text and extracts complete sentences."""
    
    SENTENCE_ENDINGS = re.compile(r'([.!?])\s+|([.!?])$')
    
    def __init__(self):
        self.buffer = ""
    
    def add(self, text):
        """Add text chunk and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        while True:
            match = self.SENTENCE_ENDINGS.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[end_pos:]
            else:
                break
        
        return sentences
    
    def flush(self):
        """Return any remaining text as a final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


class PiperTTS:
    
    VOICE_MODEL = "en_GB-northern_english_male-medium"
    MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"
    CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json"    

    def __init__(self):
        self.enabled = False
        self.voice = None
        self.model_path = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.interrupt_event = threading.Event()
        self.piper_dir = Path.home() / ".local" / "share" / "piper"
        self.models_dir = self.piper_dir / "voices"
        self.available = True  # We'll check during initialize
    
    def _download_model(self):
        """Download voice model if not present."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / f"{self.VOICE_MODEL}.onnx"
        config_path = self.models_dir / f"{self.VOICE_MODEL}.onnx.json"
        
        if not model_path.exists():
            print(f"{CYAN}[TTS] Downloading voice model ({self.VOICE_MODEL})...{RESET}")
            r = http_session.get(self.MODEL_URL, stream=True)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{GREEN}[TTS] ✓ Voice model downloaded!{RESET}")

        if not config_path.exists():
            print(f"{CYAN}[TTS] Downloading voice configuration...{RESET}")
            r = http_session.get(self.CONFIG_URL)
            r.raise_for_status()
            with open(config_path, "wb") as f:
                f.write(r.content)
            print(f"{GREEN}[TTS] ✓ Voice configuration downloaded!{RESET}")
        
        return str(model_path)
    
    def initialize(self):
        """Set up Piper voice model using the native Python package."""
        try:
            print(f"{CYAN}[TTS] Initializing Piper TTS...{RESET}")

            # Download/find voice model
            self.model_path = self._download_model()

            # Load Piper voice
            print(f"{CYAN}[TTS] Loading voice model...{RESET}")
            self.voice = PiperVoice.load(self.model_path)

            # Start the worker thread
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._speech_worker,
                daemon=True
            )
            self.worker_thread.start()

            print(f"{GREEN}[TTS] ✓ Piper TTS ready ({self.VOICE_MODEL}){RESET}")
            return True

        except Exception as e:
            print(f"{YELLOW}[TTS] Failed to initialize: {e}{RESET}")
            import traceback
            traceback.print_exc()
            self.available = False
            return False
    
    def _speech_worker(self):
        """Background thread that plays queued sentences."""
        while self.running:
            try:
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()
                
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    break
                
                if self.interrupt_event.is_set():
                    self.speech_queue.task_done()
                    continue

                self._speak_text(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def _speak_text(self, text):
        """Synthesize and play text using the native Piper Python package."""
        if not self.voice or not text.strip():
            return

        try:
            audio_buffer = io.BytesIO()

            # Synthesize speech to a WAV file in memory
            import wave
            with wave.open(audio_buffer, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)            

            if self.interrupt_event.is_set():
                return

            # Read the WAV data
            audio_buffer.seek(0)
            import wave

            with wave.open(audio_buffer, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                audio_bytes = wav_file.readframes(wav_file.getnframes())

            if self.interrupt_event.is_set():
                return

            # Convert audio to numpy array
            if sample_width == 2:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            else:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int8)

            # Play audio
            sd.play(
                audio_data,
                samplerate=sample_rate,
                blocking=True
            )

        except Exception as e:
            print(f"{YELLOW}[TTS Error]: {e}{RESET}")
            import traceback
            traceback.print_exc()
    
    def queue_sentence(self, sentence):
        """Add a sentence to the speech queue."""
        if self.enabled and self.voice and sentence.strip():
            self.speech_queue.put(sentence)
    
    def stop(self):
        """Interrupt current speech and clear queue."""
        self.interrupt_event.set()
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        
        # Stop current playback
        try:
            sd.stop()
        except:
            pass
        
            
    def wait_for_completion(self):
        """Wait for all queued speech to finish."""
        if self.enabled:
            self.speech_queue.join()
    
    def toggle(self, enable):
        """Enable/disable TTS."""
        if enable and not self.voice:
            if self.initialize():
                self.enabled = True
                return True
            return False
        self.enabled = enable
        return True
    
    def shutdown(self):
        """Clean up resources."""
        self.running = False
        self.stop()
        self.speech_queue.put(None)


# Global TTS instance
tts = PiperTTS()
