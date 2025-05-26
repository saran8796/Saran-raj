import os
import threading
import time
from pathlib import Path
import requests
import re
import json
import datetime
import psutil
import subprocess
import webbrowser
import warnings
warnings.filterwarnings("ignore")

# Flask imports
from flask import Flask, request, jsonify, render_template

# Voice Assistant imports
import speech_recognition as sr
import pyttsx3

# NLP imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, TokenClassificationPipeline,
    pipeline, AutoModel
)
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NLPCommandProcessor:
    def __init__(self):
        print("Initializing NLP models...")
        self.setup_models()
        self.setup_command_patterns()
        
    def setup_models(self):
        """Initialize Hugging Face models for NLP processing"""
        try:
            # Intent classification model (using a general classification model)
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Named Entity Recognition
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✓ NLP models loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load some NLP models: {e}")
            self.use_fallback = True
    
    def setup_command_patterns(self):
        """Define command categories and their semantic embeddings"""
        self.command_categories = {
            "system_control": [
                "shutdown computer", "turn off system", "power off",
                "restart computer", "reboot system", "restart machine",
                "sleep computer", "hibernate system", "put to sleep",
                "lock computer", "lock screen", "secure system"
            ],
            "application_control": [
                "open application", "launch program", "start software",
                "open notepad", "launch text editor", "start calculator",
                "open browser", "launch web browser", "start internet",
                "open file explorer", "launch file manager"
            ],
            "volume_control": [
                "increase volume", "turn up sound", "volume up",
                "decrease volume", "turn down sound", "volume down",
                "mute audio", "silence sound", "turn off volume"
            ],
            "system_info": [
                "system information", "computer stats", "system status",
                "battery level", "power status", "battery information",
                "current time", "what time is it", "tell me the time",
                "current date", "what date is it", "today's date"
            ],
            "file_operations": [
                "create folder", "make directory", "new folder",
                "list files", "show files", "display files",
                "delete file", "remove file", "delete folder"
            ],
            "web_operations": [
                "search web", "google search", "search internet",
                "search for", "look up", "find information about"
            ],
            "assistant_control": [
                "stop listening", "pause assistant", "sleep mode",
                "goodbye", "exit", "quit", "terminate"
            ],
            "conversation": [
                "hello", "hi", "how are you", "what can you do",
                "help me", "assist me", "tell me a joke"
            ]
        }
        
        # Create embeddings for all command patterns
        self.category_embeddings = {}
        for category, commands in self.command_categories.items():
            embeddings = self.sentence_model.encode(commands)
            self.category_embeddings[category] = embeddings
    
    def classify_intent(self, text):
        """Classify the intent of the user's command"""
        try:
            # Use zero-shot classification
            candidate_labels = list(self.command_categories.keys())
            result = self.intent_classifier(text, candidate_labels)
            
            # Get the most likely intent
            top_intent = result['labels'][0]
            confidence = result['scores'][0]
            
            return top_intent, confidence
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            return self.fallback_intent_classification(text)
    
    def fallback_intent_classification(self, text):
        """Fallback method using semantic similarity"""
        text_embedding = self.sentence_model.encode([text])
        
        best_category = None
        best_score = 0
        
        for category, embeddings in self.category_embeddings.items():
            similarities = cosine_similarity(text_embedding, embeddings)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_category = category
        
        return best_category, best_score
    
    def extract_entities(self, text):
        """Extract named entities and important information from text"""
        try:
            # Use NER pipeline
            entities = self.ner_pipeline(text)
            
            # Extract specific patterns
            extracted_info = {
                'entities': entities,
                'numbers': re.findall(r'\d+', text),
                'applications': self.extract_applications(text),
                'file_names': self.extract_file_names(text),
                'time_expressions': self.extract_time_expressions(text),
                'search_query': self.extract_search_query(text)
            }
            
            return extracted_info
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {'entities': [], 'numbers': [], 'applications': [], 
                   'file_names': [], 'time_expressions': [], 'search_query': ''}
    
    def extract_applications(self, text):
        """Extract application names from text"""
        apps = {
            'notepad': ['notepad', 'text editor', 'editor'],
            'calculator': ['calculator', 'calc', 'math'],
            'browser': ['browser', 'chrome', 'firefox', 'edge', 'internet'],
            'explorer': ['explorer', 'file manager', 'files', 'folders']
        }
        
        found_apps = []
        text_lower = text.lower()
        
        for app, keywords in apps.items():
            if any(keyword in text_lower for keyword in keywords):
                found_apps.append(app)
        
        return found_apps
    
    def extract_file_names(self, text):
        """Extract potential file or folder names"""
        patterns = [
            r'"([^"]*)"',
            r"named\s+(\w+)",
            r"called\s+(\w+)",
            r"create\s+(?:folder\s+)?(\w+)",
        ]
        
        names = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend(matches)
        
        return names
    
    def extract_time_expressions(self, text):
        """Extract time-related expressions"""
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}\s*(?:am|pm)',
            r'(?:in\s+)?(\d+)\s*(?:minutes?|hours?|seconds?)',
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        return times
    
    def extract_search_query(self, text):
        """Extract search query from text"""
        command_words = ['search', 'for', 'look', 'up', 'find', 'about', 'google']
        
        words = text.lower().split()
        query_words = [word for word in words if word not in command_words]
        
        return ' '.join(query_words)
    
    def process_command(self, text):
        """Main method to process natural language command"""
        intent, confidence = self.classify_intent(text)
        entities = self.extract_entities(text)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'original_text': text
        }

class CombinedVoiceAssistant:
    def __init__(self, name="Nova"):
        self.name = name
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
        self.listening = False
        
        # Initialize NLP processor
        self.nlp_processor = NLPCommandProcessor()
        
        # Wake words
        self.wake_words = ["hey nova", "nova", "assistant", "computer"]
        
        # Command handlers mapped to intents
        self.intent_handlers = {
            "system_control": self.handle_system_control,
            "application_control": self.handle_application_control,
            "volume_control": self.handle_volume_control,
            "system_info": self.handle_system_info,
            "file_operations": self.handle_file_operations,
            "web_operations": self.handle_web_operations,
            "assistant_control": self.handle_assistant_control,
            "conversation": self.handle_conversation,
        }
        
        # Flask app setup
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
        # Hugging Face API setup
        self.HF_API_TOKEN = os.getenv("HF_API_KEY")
        self.HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
        self.headers = {"Authorization": f"Bearer {self.HF_API_TOKEN}"}
        
        print(f"{self.name} with advanced NLP and web interface initialized successfully!")

    def setup_tts(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)

    def setup_flask_routes(self):
        """Setup Flask routes for web interface"""
        
        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/query", methods=["POST"])
        def query():
            user_input = request.json.get("text", "")
            
            # Process with local NLP first
            response = self.process_text_command(user_input)
            
            # If no specific response, try Hugging Face API
            if not response or response == "I understand what you said, but I'm not sure how to help with that yet.":
                try:
                    hf_response = requests.post(
                        self.HF_API_URL, 
                        headers=self.headers, 
                        json={"inputs": user_input}
                    )
                    if hf_response.status_code == 200:
                        result = hf_response.json()
                        response = result[0].get("generated_text", response)
                except Exception as e:
                    print(f"HF API Error: {e}")
            
            return jsonify({"response": response})

        @self.app.route("/voice", methods=["POST"])
        def voice_command():
            """Handle voice commands via web interface"""
            try:
                command = self.listen()
                if command:
                    response = self.process_text_command(command)
                    return jsonify({"command": command, "response": response})
                else:
                    return jsonify({"error": "Could not understand voice input"})
            except Exception as e:
                return jsonify({"error": str(e)})

    def speak(self, text):
        """Convert text to speech"""
        print(f"{self.name}: {text}")
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def listen(self):
        """Listen for voice input"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("🎤 Listening...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            command = self.recognizer.recognize_google(audio).lower()
            print(f"🗣️ You said: {command}")
            return command
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None

    def process_text_command(self, command):
        """Process text command and return response (for web interface)"""
        if not command:
            return "I didn't receive any command."
        
        try:
            nlp_result = self.nlp_processor.process_command(command)
            print(f"🧠 Intent: {nlp_result['intent']} (confidence: {nlp_result['confidence']:.2f})")
            
            intent = nlp_result['intent']
            if intent in self.intent_handlers:
                return self.intent_handlers[intent](nlp_result, return_response=True)
            else:
                return "I understand what you said, but I'm not sure how to help with that yet."
                
        except Exception as e:
            print(f"❌ Error processing command: {e}")
            return f"I encountered an error processing that command: {str(e)}"

    def process_command(self, command):
        """Process command using NLP understanding (for voice interface)"""
        if not command:
            return
        
        try:
            nlp_result = self.nlp_processor.process_command(command)
            print(f"🧠 Intent: {nlp_result['intent']} (confidence: {nlp_result['confidence']:.2f})")
            
            intent = nlp_result['intent']
            if intent in self.intent_handlers:
                self.intent_handlers[intent](nlp_result)
            else:
                self.speak("I understand what you said, but I'm not sure how to help with that yet.")
                
        except Exception as e:
            print(f"❌ Error processing command: {e}")
            self.speak("I encountered an error processing that command.")

    # Intent Handlers (modified to support both voice and text responses)
    def handle_conversation(self, nlp_result, return_response=False):
        """Handle general conversation"""
        text = nlp_result['original_text'].lower()
        
        if any(word in text for word in ['hello', 'hi']):
            response = f"Hello! I'm {self.name}, your intelligent assistant. How can I help you today?"
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_system_control(self, nlp_result, return_response=False):
        """Handle system control commands"""
        text = nlp_result['original_text'].lower()
        
        if any(word in text for word in ['shutdown', 'turn off', 'power off']):
            response = "Initiating system shutdown..."
            if not return_response:
                self.shutdown_system()
        elif any(word in text for word in ['restart', 'reboot']):
            response = "Restarting the system..."
            if not return_response:
                self.restart_system()
        elif any(word in text for word in ['sleep', 'hibernate']):
            response = "Putting system to sleep..."
            if not return_response:
                self.sleep_system()
        elif any(word in text for word in ['lock']):
            response = "Locking the system..."
            if not return_response:
                self.lock_system()
        else:
            response = "What would you like me to do with the system?"
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_application_control(self, nlp_result, return_response=False):
        """Handle application control commands"""
        apps = nlp_result['entities']['applications']
        
        if 'notepad' in apps:
            response = "Opening Notepad..."
            if not return_response:
                self.open_application("notepad.exe")
        elif 'calculator' in apps:
            response = "Opening Calculator..."
            if not return_response:
                self.open_application("calc.exe")
        elif 'browser' in apps:
            response = "Opening web browser..."
            if not return_response:
                webbrowser.open("https://www.google.com")
        elif 'explorer' in apps:
            response = "Opening File Explorer..."
            if not return_response:
                self.open_application("explorer.exe")
        else:
            response = "I'll try to open the application you mentioned."
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_volume_control(self, nlp_result, return_response=False):
        """Handle volume control commands"""
        text = nlp_result['original_text'].lower()
        
        if any(word in text for word in ['up', 'increase', 'raise', 'louder']):
            response = "Increasing volume..."
            if not return_response:
                self.volume_up()
        elif any(word in text for word in ['down', 'decrease', 'lower', 'quieter']):
            response = "Decreasing volume..."
            if not return_response:
                self.volume_down()
        elif any(word in text for word in ['mute', 'silence', 'quiet']):
            response = "Toggling mute..."
            if not return_response:
                self.mute_volume()
        else:
            response = "Volume command processed."
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_system_info(self, nlp_result, return_response=False):
        """Handle system information requests"""
        text = nlp_result['original_text'].lower()
        
        if any(word in text for word in ['time', 'clock']):
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
        elif any(word in text for word in ['date', 'today']):
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            response = f"Today is {current_date}"
        elif any(word in text for word in ['battery', 'power']):
            response = self.get_battery_status_text()
        elif any(word in text for word in ['system', 'computer', 'stats']):
            response = self.get_system_info_text()
        else:
            response = "What system information would you like?"
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_file_operations(self, nlp_result, return_response=False):
        """Handle file operations"""
        text = nlp_result['original_text'].lower()
        file_names = nlp_result['entities']['file_names']
        
        if 'create' in text and 'folder' in text:
            if file_names:
                response = f"Creating folder {file_names[0]}..."
                if not return_response:
                    self.create_folder(file_names[0])
            else:
                response = "What should I name the folder?"
        elif 'list' in text or 'show' in text:
            response = self.list_files_text()
        else:
            response = "File operation processed."
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_web_operations(self, nlp_result, return_response=False):
        """Handle web operations"""
        query = nlp_result['entities']['search_query']
        
        if query.strip():
            response = f"Searching for {query}..."
            if not return_response:
                self.web_search(query)
        else:
            response = "What would you like me to search for?"
        
        if return_response:
            return response
        else:
            self.speak(response)

    def handle_assistant_control(self, nlp_result, return_response=False):
        """Handle assistant control commands"""
        text = nlp_result['original_text'].lower()
        
        if any(word in text for word in ['stop', 'pause']):
            response = "Stopping listening mode..."
            if not return_response:
                self.stop_listening()
        elif any(word in text for word in ['goodbye', 'exit', 'quit']):
            response = "Goodbye! Have a great day!"
            if not return_response:
                # Don't actually exit in web mode
                pass
        else:
            response = "Assistant control processed."
        
        if return_response:
            return response
        else:
            self.speak(response)

    # System Control Methods
    def shutdown_system(self):
        self.speak("Shutting down the system in 10 seconds. Say cancel to stop.")
        time.sleep(5)
        try:
            if os.name == 'nt':
                os.system("shutdown /s /t 5")
            else:
                os.system("shutdown -h +1")
        except Exception as e:
            self.speak(f"Failed to shutdown: {str(e)}")

    def restart_system(self):
        self.speak("Restarting the system in 10 seconds.")
        time.sleep(5)
        try:
            if os.name == 'nt':
                os.system("shutdown /r /t 5")
            else:
                os.system("reboot")
        except Exception as e:
            self.speak(f"Failed to restart: {str(e)}")

    def sleep_system(self):
        try:
            if os.name == 'nt':
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            else:
                os.system("systemctl suspend")
        except Exception as e:
            self.speak(f"Failed to sleep: {str(e)}")

    def lock_system(self):
        try:
            if os.name == 'nt':
                os.system("rundll32.exe user32.dll,LockWorkStation")
            else:
                os.system("gnome-screensaver-command -l")
        except Exception as e:
            self.speak(f"Failed to lock: {str(e)}")

    def open_application(self, app_name):
        try:
            if os.name == 'nt':
                subprocess.Popen(app_name)
            else:
                subprocess.Popen([app_name])
        except Exception as e:
            self.speak(f"Failed to open {app_name}: {str(e)}")

    def volume_up(self):
        try:
            if os.name == 'nt':
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, None, None)
                volume = interface.QueryInterface(IAudioEndpointVolume)
                current_volume = volume.GetMasterVolume()
                volume.SetMasterVolume(min(current_volume + 0.1, 0.0), None)
        except Exception as e:
            print("Volume control not available")

    def volume_down(self):
        try:
            if os.name == 'nt':
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, None, None)
                volume = interface.QueryInterface(IAudioEndpointVolume)
                current_volume = volume.GetMasterVolume()
                volume.SetMasterVolume(max(current_volume - 0.1, -65.25), None)
        except Exception as e:
            print("Volume control not available")

    def mute_volume(self):
        try:
            if os.name == 'nt':
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, None, None)
                volume = interface.QueryInterface(IAudioEndpointVolume)
                volume.SetMute(not volume.GetMute(), None)
        except Exception as e:
            print("Volume control not available")

    def get_system_info_text(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return f"CPU usage is {cpu_percent}%. Memory usage is {memory.percent}%. Disk usage is {disk.percent}%."
        except Exception as e:
            return "Could not retrieve system information"

    def get_battery_status_text(self):
        try:
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                plugged = "plugged in" if battery.power_plugged else "not plugged in"
                return f"Battery is at {percent}% and {plugged}"
            else:
                return "Battery information not available"
        except Exception as e:
            return "Could not retrieve battery status"

    def create_folder(self, folder_name):
        try:
            desktop = Path.home() / "Desktop"
            folder_path = desktop / folder_name
            folder_path.mkdir(exist_ok=True)
        except Exception as e:
            self.speak(f"Failed to create folder: {str(e)}")

    def list_files_text(self):
        try:
            files = os.listdir(".")
            if files:
                file_list = ", ".join(files[:5])
                return f"Here are some files: {file_list}"
            else:
                return "No files found in current directory"
        except Exception as e:
            return "Could not list files"

    def web_search(self, query):
        try:
            search_url = f"https://www.google.com/search?q={query}"
            webbrowser.open(search_url)
        except Exception as e:
            print("Could not perform web search")

    def stop_listening(self):
        self.listening = False

    def run_voice_mode(self):
        """Run the voice assistant in standalone mode"""
        self.speak(f"Hello! I'm {self.name}, your intelligent voice assistant. I understand natural language commands.")
        self.listening = True
        
        while True:
            try:
                command = self.listen()
                
                if command:
                    if not self.listening:
                        if any(wake_word in command for wake_word in self.wake_words):
                            self.listening = True
                            self.speak("Yes, I'm here")
                        continue
                    
                    if self.listening:
                        self.process_command(command)
                        
            except KeyboardInterrupt:
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

    def run_web_mode(self, host='0.0.0.0', port=5000, debug=True):
        """Run the web interface"""
        print(f"Starting {self.name} web interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

    def run_combined_mode(self, host='0.0.0.0', port=5000):
        """Run both voice and web interface simultaneously"""
        print(f"Starting {self.name} in combined mode...")
        
        # Start web server in a separate thread
        web_thread = threading.Thread(
            target=self.run_web_mode, 
            kwargs={'host': host, 'port': port, 'debug': False}
        )
        web_thread.daemon = True
        web_thread.start()
        
        # Run voice assistant in main thread
        self.run_voice_mode()

if __name__ == "__main__":
    import sys
    
    # Create the combined assistant
    assistant = CombinedVoiceAssistant("NOVA")
    
    # Check command line arguments for mode selection
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "web":
            assistant.run_web_mode()
        elif mode == "voice":
            assistant.run_voice_mode()
        elif mode == "combined":
            assistant.run_combined_mode()
        else:
            print("Usage: python main.py [web|voice|combined]")
            print("Running in combined mode by default...")
            assistant.run_combined_mode()
    else:
        # Default to combined mode
        assistant.run_combined_mode()