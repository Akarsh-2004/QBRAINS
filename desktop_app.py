#!/usr/bin/env python3
"""
Desktop Application for Quantum Emotion Pipeline
Tkinter-based GUI application
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.quantum_pipeline_integrated import IntegratedQuantumPipeline
from src.realtime_processor import RealTimeProcessor
from src.multi_person_detector import MultiPersonDetector
import cv2
from PIL import Image, ImageTk
import numpy as np


class QuantumEmotionApp:
    """Main desktop application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Emotion Pipeline - Desktop App")
        self.root.geometry("1200x800")
        
        # Initialize pipeline
        self.pipeline = IntegratedQuantumPipeline()
        self.realtime_processor = None
        self.multi_person_detector = None
        self.camera = None
        self.realtime_running = False
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create user interface"""
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Text Processing
        self.text_frame = ttk.Frame(notebook)
        notebook.add(self.text_frame, text="📝 Text Processing")
        self.create_text_tab()
        
        # Tab 2: File Processing
        self.file_frame = ttk.Frame(notebook)
        notebook.add(self.file_frame, text="📁 File Processing")
        self.create_file_tab()
        
        # Tab 3: Real-time Processing
        self.realtime_frame = ttk.Frame(notebook)
        notebook.add(self.realtime_frame, text="🎥 Real-time")
        self.create_realtime_tab()
        
        # Tab 4: Multi-Person
        self.multiperson_frame = ttk.Frame(notebook)
        notebook.add(self.multiperson_frame, text="👥 Multi-Person")
        self.create_multiperson_tab()
        
        # Tab 5: Memory
        self.memory_frame = ttk.Frame(notebook)
        notebook.add(self.memory_frame, text="🧠 Memory")
        self.create_memory_tab()
    
    def create_text_tab(self):
        """Create text processing tab"""
        # Input
        ttk.Label(self.text_frame, text="Enter text:").pack(pady=5)
        self.text_input = scrolledtext.ScrolledText(self.text_frame, height=5, width=80)
        self.text_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Process button
        ttk.Button(self.text_frame, text="Process Text", 
                  command=self.process_text).pack(pady=5)
        
        # Result
        ttk.Label(self.text_frame, text="Result:").pack(pady=5)
        self.text_result = scrolledtext.ScrolledText(self.text_frame, height=10, width=80)
        self.text_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Chat section
        ttk.Separator(self.text_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.text_frame, text="Chat:").pack(pady=5)
        self.chat_input = scrolledtext.ScrolledText(self.text_frame, height=3, width=80)
        self.chat_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        ttk.Button(self.text_frame, text="Send", 
                  command=self.send_chat).pack(pady=5)
        self.chat_result = scrolledtext.ScrolledText(self.text_frame, height=5, width=80)
        self.chat_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
    
    def create_file_tab(self):
        """Create file processing tab"""
        # Audio
        audio_frame = ttk.LabelFrame(self.file_frame, text="Audio Processing")
        audio_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Button(audio_frame, text="Select Audio File", 
                  command=self.select_audio).pack(pady=5)
        self.audio_path_label = ttk.Label(audio_frame, text="No file selected")
        self.audio_path_label.pack(pady=5)
        ttk.Button(audio_frame, text="Process Audio", 
                  command=self.process_audio).pack(pady=5)
        
        # Video
        video_frame = ttk.LabelFrame(self.file_frame, text="Video Processing")
        video_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Button(video_frame, text="Select Video File", 
                  command=self.select_video).pack(pady=5)
        self.video_path_label = ttk.Label(video_frame, text="No file selected")
        self.video_path_label.pack(pady=5)
        ttk.Button(video_frame, text="Process Video", 
                  command=self.process_video).pack(pady=5)
        
        # EEG
        eeg_frame = ttk.LabelFrame(self.file_frame, text="EEG Processing")
        eeg_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Button(eeg_frame, text="Select EEG File (CSV)", 
                  command=self.select_eeg).pack(pady=5)
        self.eeg_path_label = ttk.Label(eeg_frame, text="No file selected")
        self.eeg_path_label.pack(pady=5)
        ttk.Button(eeg_frame, text="Process EEG", 
                  command=self.process_eeg).pack(pady=5)
        
        # Result
        ttk.Label(self.file_frame, text="Result:").pack(pady=5)
        self.file_result = scrolledtext.ScrolledText(self.file_frame, height=15, width=80)
        self.file_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
    
    def create_realtime_tab(self):
        """Create real-time processing tab"""
        # Controls
        control_frame = ttk.Frame(self.realtime_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Start Camera", 
                  command=self.start_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Camera", 
                  command=self.stop_realtime).pack(side=tk.LEFT, padx=5)
        
        # Video display
        self.video_label = ttk.Label(self.realtime_frame, text="Camera feed will appear here")
        self.video_label.pack(pady=10)
        
        # Result
        ttk.Label(self.realtime_frame, text="Real-time Result:").pack(pady=5)
        self.realtime_result = scrolledtext.ScrolledText(self.realtime_frame, height=10, width=80)
        self.realtime_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
    
    def create_multiperson_tab(self):
        """Create multi-person detection tab"""
        # Controls
        control_frame = ttk.Frame(self.multiperson_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Start Multi-Person Detection", 
                  command=self.start_multiperson).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_multiperson).pack(side=tk.LEFT, padx=5)
        
        # Video display
        self.multiperson_label = ttk.Label(self.multiperson_frame, text="Multi-person view will appear here")
        self.multiperson_label.pack(pady=10)
        
        # Summary
        ttk.Label(self.multiperson_frame, text="People Summary:").pack(pady=5)
        self.multiperson_result = scrolledtext.ScrolledText(self.multiperson_frame, height=10, width=80)
        self.multiperson_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
    
    def create_memory_tab(self):
        """Create memory tab"""
        ttk.Button(self.memory_frame, text="Get Memory Summary", 
                  command=self.get_memory).pack(pady=10)
        ttk.Button(self.memory_frame, text="Clear Memory", 
                  command=self.clear_memory).pack(pady=5)
        
        self.memory_result = scrolledtext.ScrolledText(self.memory_frame, height=20, width=80)
        self.memory_result.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    
    def process_text(self):
        """Process text input"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text")
            return
        
        def process():
            try:
                result = self.pipeline.process(text=text)
                self.display_result(self.text_result, result)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=process, daemon=True).start()
    
    def send_chat(self):
        """Send chat message"""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        def chat():
            try:
                response = self.pipeline.chat(message)
                self.chat_result.insert(tk.END, f"You: {message}\n")
                self.chat_result.insert(tk.END, f"AI: {response}\n\n")
                self.chat_input.delete("1.0", tk.END)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=chat, daemon=True).start()
    
    def select_audio(self):
        """Select audio file"""
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.m4a")])
        if path:
            self.audio_path = path
            self.audio_path_label.config(text=f"Selected: {Path(path).name}")
    
    def select_video(self):
        """Select video file"""
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.video_path_label.config(text=f"Selected: {Path(path).name}")
    
    def select_eeg(self):
        """Select EEG file"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.eeg_path = path
            self.eeg_path_label.config(text=f"Selected: {Path(path).name}")
    
    def process_audio(self):
        """Process audio file"""
        if not hasattr(self, 'audio_path'):
            messagebox.showwarning("Warning", "Please select an audio file")
            return
        
        def process():
            try:
                result = self.pipeline.process(audio_path=self.audio_path)
                self.display_result(self.file_result, result)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=process, daemon=True).start()
    
    def process_video(self):
        """Process video file"""
        if not hasattr(self, 'video_path'):
            messagebox.showwarning("Warning", "Please select a video file")
            return
        
        def process():
            try:
                result = self.pipeline.process(video_path=self.video_path)
                self.display_result(self.file_result, result)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=process, daemon=True).start()
    
    def process_eeg(self):
        """Process EEG file"""
        if not hasattr(self, 'eeg_path'):
            messagebox.showwarning("Warning", "Please select an EEG file")
            return
        
        def process():
            try:
                result = self.pipeline.process(eeg_path=self.eeg_path)
                self.display_result(self.file_result, result)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=process, daemon=True).start()
    
    def start_realtime(self):
        """Start real-time processing"""
        if self.realtime_running:
            return
        
        self.realtime_running = True
        self.realtime_processor = RealTimeProcessor(pipeline=self.pipeline)
        self.realtime_processor.start_video_stream(camera_id=0)
        
        def update_video():
            cap = cv2.VideoCapture(0)
            while self.realtime_running:
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk
                    
                    # Get latest result
                    result = self.realtime_processor.get_latest_result(timeout=0.1)
                    if result:
                        self.display_result(self.realtime_result, result)
                else:
                    break
            cap.release()
        
        threading.Thread(target=update_video, daemon=True).start()
    
    def stop_realtime(self):
        """Stop real-time processing"""
        self.realtime_running = False
        if self.realtime_processor:
            self.realtime_processor.stop()
    
    def start_multiperson(self):
        """Start multi-person detection"""
        if self.realtime_running:
            return
        
        self.realtime_running = True
        self.multi_person_detector = MultiPersonDetector()
        
        def update_multiperson():
            cap = cv2.VideoCapture(0)
            while self.realtime_running:
                ret, frame = cap.read()
                if ret:
                    # Detect and track
                    tracks = self.multi_person_detector.detect_and_track(frame)
                    
                    # Draw tracks
                    frame_with_tracks = self.multi_person_detector.draw_tracks(frame)
                    
                    # Display
                    frame_rgb = cv2.cvtColor(frame_with_tracks, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.multiperson_label.config(image=imgtk)
                    self.multiperson_label.image = imgtk
                    
                    # Update summary
                    summaries = self.multi_person_detector.get_all_summaries()
                    summary_text = "\n".join([
                        f"Person {pid}: {s['dominant_emotion']} (confidence: {s['confidence']:.2f})"
                        for pid, s in summaries.items()
                    ])
                    self.multiperson_result.delete("1.0", tk.END)
                    self.multiperson_result.insert("1.0", summary_text)
                else:
                    break
            cap.release()
        
        threading.Thread(target=update_multiperson, daemon=True).start()
    
    def stop_multiperson(self):
        """Stop multi-person detection"""
        self.realtime_running = False
    
    def get_memory(self):
        """Get memory summary"""
        def get():
            try:
                summary = self.pipeline.get_memory_summary()
                self.memory_result.delete("1.0", tk.END)
                self.memory_result.insert("1.0", str(summary))
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=get, daemon=True).start()
    
    def clear_memory(self):
        """Clear memory"""
        if messagebox.askyesno("Confirm", "Clear all memory?"):
            self.pipeline.clear_memory()
            messagebox.showinfo("Success", "Memory cleared")
    
    def display_result(self, widget, result):
        """Display result in text widget"""
        widget.delete("1.0", tk.END)
        
        if 'quantum_superposition' in result:
            qs = result['quantum_superposition']
            widget.insert(tk.END, f"Emotion: {qs.get('collapsed_emotion', 'N/A')}\n")
            widget.insert(tk.END, f"Uncertainty: {qs.get('uncertainty', 0):.3f}\n\n")
        
        if 'final_output' in result:
            widget.insert(tk.END, f"Output: {result['final_output'].get('formatted_text', 'N/A')}\n\n")
        
        widget.insert(tk.END, f"Full Result:\n{str(result)}\n")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = QuantumEmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

