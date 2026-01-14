import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime, timedelta
import json
from personal_emotion_memory import PersonalEmotionMemory, EmotionSession

class EmotionMemoryGUI:
    """GUI for Personal Emotion Memory System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Emotion Memory")
        self.root.geometry("1000x700")
        
        self.memory = PersonalEmotionMemory()
        self.current_user_id = None
        
        self.setup_ui()
        self.load_or_create_user()
    
    def setup_ui(self):
        """Setup the main UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # User info section
        self.create_user_section(main_frame)
        
        # Tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_history_tab()
        self.create_insights_tab()
        self.create_settings_tab()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def create_user_section(self, parent):
        """Create user information section"""
        user_frame = ttk.LabelFrame(parent, text="User Profile", padding="10")
        user_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.user_id_label = ttk.Label(user_frame, text="User ID: Not loaded")
        self.user_id_label.grid(row=0, column=0, sticky=tk.W)
        
        self.session_count_label = ttk.Label(user_frame, text="Sessions: 0")
        self.session_count_label.grid(row=0, column=1, padx=(20, 0))
        
        ttk.Button(user_frame, text="Export Data", command=self.export_data).grid(row=0, column=2, padx=(20, 0))
        ttk.Button(user_frame, text="Delete All Data", command=self.delete_data).grid(row=0, column=3, padx=(5, 0))
    
    def create_dashboard_tab(self):
        """Create dashboard tab with current emotions and patterns"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Current emotions section
        current_frame = ttk.LabelFrame(dashboard_frame, text="Current Emotions", padding="10")
        current_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Emotion bars
        self.emotion_canvas = tk.Canvas(current_frame, height=200, bg='white')
        self.emotion_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add emotion button
        ttk.Button(current_frame, text="Add Current Emotions", 
                  command=self.add_emotion_session).grid(row=1, column=0, pady=10)
        
        # Baseline emotions section
        baseline_frame = ttk.LabelFrame(dashboard_frame, text="Your Emotional Baseline", padding="10")
        baseline_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.baseline_canvas = tk.Canvas(baseline_frame, height=150, bg='white')
        self.baseline_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_history_tab(self):
        """Create history tab with emotion timeline"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="History")
        
        # Time range selector
        range_frame = ttk.Frame(history_frame)
        range_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Label(range_frame, text="Time Range:").grid(row=0, column=0)
        self.time_range = ttk.Combobox(range_frame, values=["1 Day", "7 Days", "30 Days", "90 Days"])
        self.time_range.set("7 Days")
        self.time_range.grid(row=0, column=1, padx=(5, 0))
        self.time_range.bind('<<ComboboxSelected>>', lambda e: self.update_history())
        
        # History plot
        self.history_frame = ttk.Frame(history_frame)
        self.history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Configure grid weights
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(1, weight=1)
    
    def create_insights_tab(self):
        """Create insights tab with personal insights"""
        insights_frame = ttk.Frame(self.notebook)
        self.notebook.add(insights_frame, text="Insights")
        
        # Insights list
        self.insights_text = tk.Text(insights_frame, wrap=tk.WORD, height=20)
        insights_scroll = ttk.Scrollbar(insights_frame, orient="vertical", command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=insights_scroll.set)
        
        self.insights_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        insights_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S), pady=10)
        
        # Refresh button
        ttk.Button(insights_frame, text="Refresh Insights", 
                  command=self.update_insights).grid(row=1, column=0, pady=10)
        
        # Configure grid weights
        insights_frame.columnconfigure(0, weight=1)
        insights_frame.rowconfigure(0, weight=1)
    
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Privacy settings
        privacy_frame = ttk.LabelFrame(settings_frame, text="Privacy Settings", padding="10")
        privacy_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Label(privacy_frame, text="Data Retention:").grid(row=0, column=0, sticky=tk.W)
        self.retention_var = tk.StringVar(value="90 days")
        retention_combo = ttk.Combobox(privacy_frame, textvariable=self.retention_var,
                                     values=["30 days", "90 days", "180 days", "1 year", "Forever"])
        retention_combo.grid(row=0, column=1, padx=(5, 0))
        
        # Insight frequency
        ttk.Label(privacy_frame, text="Insight Frequency:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.insight_freq_var = tk.StringVar(value="daily")
        freq_combo = ttk.Combobox(privacy_frame, textvariable=self.insight_freq_var,
                                 values=["real-time", "daily", "weekly"])
        freq_combo.grid(row=1, column=1, padx=(5, 0), pady=(10, 0))
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", 
                  command=self.save_settings).grid(row=1, column=0, pady=20)
    
    def load_or_create_user(self):
        """Load existing user or create new one"""
        # For demo, always create/load "demo_user"
        user_id = "demo_user"
        profile = self.memory.load_user_profile(user_id)
        
        if not profile:
            user_id = self.memory.create_user_profile("demo_user")
            messagebox.showinfo("Welcome", "New profile created for you!")
        
        self.current_user_id = user_id
        self.update_user_info()
        self.refresh_all_data()
    
    def update_user_info(self):
        """Update user information display"""
        if self.current_user_id:
            profile = self.memory.load_user_profile(self.current_user_id)
            if profile:
                self.user_id_label.config(text=f"User ID: {profile.user_id}")
                self.session_count_label.config(text=f"Sessions: {profile.total_sessions}")
    
    def add_emotion_session(self):
        """Add a new emotion session (demo with manual input)"""
        dialog = EmotionInputDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.memory.store_emotion_session(dialog.result, "manual_input")
            self.refresh_all_data()
            messagebox.showinfo("Success", "Emotion session added!")
    
    def refresh_all_data(self):
        """Refresh all displayed data"""
        self.update_dashboard()
        self.update_history()
        self.update_insights()
        self.update_user_info()
    
    def update_dashboard(self):
        """Update dashboard with current data"""
        if not self.current_user_id:
            return
        
        profile = self.memory.load_user_profile(self.current_user_id)
        if not profile:
            return
        
        # Clear canvases
        self.emotion_canvas.delete("all")
        self.baseline_canvas.delete("all")
        
        # Draw baseline emotions
        self.draw_emotion_bars(self.baseline_canvas, profile.baseline_emotions, "baseline")
        
        # Get recent session for current emotions
        recent_sessions = self.memory.get_emotion_history(days=1)
        if recent_sessions:
            current_emotions = recent_sessions[0].emotions
            self.draw_emotion_bars(self.emotion_canvas, current_emotions, "current")
    
    def draw_emotion_bars(self, canvas, emotions, chart_type):
        """Draw emotion bar chart on canvas"""
        canvas.delete("all")
        
        if not emotions:
            return
        
        # Canvas dimensions
        width = canvas.winfo_width() if canvas.winfo_width() > 1 else 600
        height = canvas.winfo_height() if canvas.winfo_height() > 1 else 200
        
        # Emotion names and colors
        emotion_names = list(emotions.keys())
        emotion_values = list(emotions.values())
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#4CAF50', '#2196F3', '#9C27B0', '#FF1493']
        
        # Calculate bar dimensions
        bar_width = (width - 100) / len(emotion_names)
        max_height = height - 40
        
        # Draw bars
        for i, (emotion, value) in enumerate(emotions.items()):
            x = 50 + i * bar_width
            bar_height = value * max_height
            y = height - 20 - bar_height
            
            # Draw bar
            canvas.create_rectangle(x, y, x + bar_width - 10, height - 20, 
                                   fill=colors[i % len(colors)], outline="black")
            
            # Draw emotion label
            canvas.create_text(x + bar_width/2 - 5, height - 5, text=emotion[:3], 
                             font=("Arial", 8), anchor="s")
            
            # Draw value
            canvas.create_text(x + bar_width/2 - 5, y - 5, text=f"{value:.2f}", 
                             font=("Arial", 8), anchor="s")
        
        # Draw title
        title = "Current Emotions" if chart_type == "current" else "Your Emotional Baseline"
        canvas.create_text(width/2, 10, text=title, font=("Arial", 12, "bold"))
    
    def update_history(self):
        """Update history tab with timeline"""
        if not self.current_user_id:
            return
        
        # Clear previous plots
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        
        # Get time range
        range_text = self.time_range.get()
        days_map = {"1 Day": 1, "7 Days": 7, "30 Days": 30, "90 Days": 90}
        days = days_map.get(range_text, 7)
        
        # Get sessions
        sessions = self.memory.get_emotion_history(days=days)
        
        if not sessions:
            ttk.Label(self.history_frame, text="No data available for selected time range").pack()
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        timestamps = [s.timestamp for s in sessions]
        emotions_data = {emotion: [] for emotion in sessions[0].emotions.keys()}
        
        for session in sessions:
            for emotion, value in session.emotions.items():
                emotions_data[emotion].append(value)
        
        # Plot each emotion
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#4CAF50', '#2196F3', '#9C27B0', '#FF1493']
        for i, (emotion, values) in enumerate(emotions_data.items()):
            ax.plot(timestamps, values, label=emotion, color=colors[i % len(colors)], linewidth=2)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Emotion Intensity")
        ax.set_title(f"Emotion History - Last {days} Days")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.history_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_insights(self):
        """Update insights tab"""
        if not self.current_user_id:
            return
        
        insights = self.memory.get_personal_insights(days=30)
        patterns = self.memory.get_emotion_patterns()
        
        self.insights_text.delete(1.0, tk.END)
        
        if not insights and not patterns:
            self.insights_text.insert(tk.END, "No insights available yet. Keep using the app to generate personal insights!")
            return
        
        # Display insights
        if insights:
            self.insights_text.insert(tk.END, "=== RECENT INSIGHTS ===\n\n")
            for insight in insights:
                self.insights_text.insert(tk.END, f"• {insight['content']}\n")
                self.insights_text.insert(tk.END, f"  Confidence: {insight['confidence']:.2f}\n")
                self.insights_text.insert(tk.END, f"  Date: {insight['created_at'][:10]}\n\n")
        
        # Display patterns
        if patterns:
            self.insights_text.insert(tk.END, "=== EMOTION PATTERNS ===\n\n")
            
            if 'daily_average' in patterns:
                self.insights_text.insert(tk.END, "Daily Averages:\n")
                for emotion, avg in patterns['daily_average'].items():
                    self.insights_text.insert(tk.END, f"  {emotion}: {avg:.3f}\n")
                self.insights_text.insert(tk.END, "\n")
            
            if 'context_patterns' in patterns:
                self.insights_text.insert(tk.END, "Context Patterns:\n")
                for context, emotions in patterns['context_patterns'].items():
                    self.insights_text.insert(tk.END, f"  {context}:\n")
                    for emotion, avg in emotions.items():
                        self.insights_text.insert(tk.END, f"    {emotion}: {avg:.3f}\n")
                self.insights_text.insert(tk.END, "\n")
    
    def export_data(self):
        """Export user data"""
        if not self.current_user_id:
            messagebox.showwarning("Warning", "No user data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.memory.export_user_data(filename)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def delete_data(self):
        """Delete all user data"""
        if not self.current_user_id:
            messagebox.showwarning("Warning", "No user data to delete")
            return
        
        if messagebox.askyesno("Confirm Delete", 
                              "This will permanently delete all your emotion data. Are you sure?"):
            try:
                self.memory.delete_user_data()
                messagebox.showinfo("Success", "All data deleted successfully")
                self.load_or_create_user()  # Create fresh profile
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete data: {str(e)}")
    
    def save_settings(self):
        """Save user settings"""
        if not self.current_user_id:
            return
        
        # Update user preferences
        profile = self.memory.load_user_profile(self.current_user_id)
        if profile:
            profile.preferences['retention'] = self.retention_var.get()
            profile.preferences['insight_frequency'] = self.insight_freq_var.get()
            
            # Save to database (simplified - in real app would update DB)
            messagebox.showinfo("Success", "Settings saved successfully!")

class EmotionInputDialog:
    """Dialog for manual emotion input"""
    
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Emotion Session")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_dialog()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def setup_dialog(self):
        """Setup dialog components"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Enter your current emotions (0.0 - 1.0):", 
                 font=("Arial", 10, "bold")).pack(pady=(0, 10))
        
        self.emotion_vars = {}
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        for emotion in emotions:
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{emotion}:", width=10).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=0.0)
            self.emotion_vars[emotion] = var
            
            scale = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
            
            label = ttk.Label(frame, text="0.00", width=5)
            label.pack(side=tk.LEFT)
            
            # Update label when scale changes
            var.trace('w', lambda *args, l=label, v=var: l.config(text=f"{v.get():.2f}"))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))
        
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)
    
    def save(self):
        """Save emotion values"""
        emotions = {}
        total = sum(var.get() for var in self.emotion_vars.values())
        
        if total == 0:
            messagebox.showwarning("Warning", "Please enter at least one emotion value")
            return
        
        # Normalize if sum > 1
        if total > 1:
            for emotion, var in self.emotion_vars.items():
                emotions[emotion] = var.get() / total
        else:
            for emotion, var in self.emotion_vars.items():
                emotions[emotion] = var.get()
        
        self.result = emotions
        self.dialog.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionMemoryGUI(root)
    root.mainloop()
