import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from utils import clean_text, logger

# Ensure the model files exist before starting
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    logger.error("Model files not found. Launch aborted.")
    print("Error: Model files not found. Please run train_model.py first!")
    exit()

# Load the saved model and vectorizer
logger.info("Loading pre-trained model and vectorizer...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Configure customtkinter appearance
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class SpamDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("💌 Pro Spam Detection System")
        self.geometry("700x650")
        self.resizable(False, False)
        
        # Grid layout configuration
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main Content
        self.grid_columnconfigure(0, weight=1)
        
        # -------------------------------
        # Header Frame
        # -------------------------------
        self.header_frame = ctk.CTkFrame(self, height=70, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="📧 EMAIL SPAM DETECTOR", 
            font=ctk.CTkFont(family="Poppins", size=20, weight="bold")
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=25, pady=20)
        
        # Theme toggle
        self.theme_label = ctk.CTkLabel(self.header_frame, text="Theme:", font=ctk.CTkFont(size=12))
        self.theme_label.grid(row=0, column=1, sticky="e", padx=(10, 5), pady=20)
        
        self.theme_option = ctk.CTkOptionMenu(
            self.header_frame, 
            values=["System", "Dark", "Light"], 
            width=100, 
            command=self.change_theme
        )
        self.theme_option.grid(row=0, column=2, sticky="e", padx=25, pady=20)
        self.theme_option.set("System")
        
        # -------------------------------
        # Main Frame
        # -------------------------------
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=25, pady=25)
        
        self.instruction_label = ctk.CTkLabel(
            self.main_frame, 
            text="Paste the email content below to analyze:",
            font=ctk.CTkFont(size=14)
        )
        self.instruction_label.pack(anchor="w", padx=30, pady=(25, 5))
        
        # Input Textbox
        self.input_text = ctk.CTkTextbox(
            self.main_frame, 
            width=580, 
            height=200, 
            font=ctk.CTkFont(family="Arial", size=13),
            wrap="word"
        )
        self.input_text.pack(padx=30, pady=10)
        
        # Analyze Button
        self.analyze_button = ctk.CTkButton(
            self.main_frame, 
            text="Analyze Message", 
            font=ctk.CTkFont(size=15, weight="bold"),
            height=40,
            command=self.check_spam
        )
        self.analyze_button.pack(padx=30, pady=15, fill="x")
        
        # -------------------------------
        # Results Frame (Card Layout)
        # -------------------------------
        self.result_frame = ctk.CTkFrame(
            self.main_frame, 
            corner_radius=10, 
            fg_color=("#F5F5F5", "#2A2A2A"),
            border_width=2,
            border_color=("#E0E0E0", "#3E3E3E")
        )
        self.result_frame.pack(padx=30, pady=(10, 25), fill="both", expand=True)
        
        self.result_title = ctk.CTkLabel(
            self.result_frame,
            text="Analysis Result",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.result_title.pack(anchor="w", padx=20, pady=(15, 5))
        
        self.result_label = ctk.CTkLabel(
            self.result_frame, 
            text="Awaiting text analysis...", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#757575", "#B0B0B0")
        )
        self.result_label.pack(anchor="w", padx=20, pady=5)
        
        # Confidence Indicator
        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="Confidence Level: N/A",
            font=ctk.CTkFont(size=12)
        )
        self.confidence_label.pack(anchor="w", padx=20, pady=0)
        
        self.confidence_bar = ctk.CTkProgressBar(self.result_frame, height=10)
        self.confidence_bar.pack(fill="x", padx=20, pady=(10, 15))
        self.confidence_bar.set(0)

    def change_theme(self, new_mode):
        ctk.set_appearance_mode(new_mode)
        logger.info(f"Appearance theme switched to: {new_mode}")
        
    def check_spam(self):
        msg = self.input_text.get("1.0", "end").strip()
        if not msg:
            messagebox.showwarning("⚠️ Input Error", "Please enter an email message to check!")
            return
        
        logger.info("Analyzing user input...")
        msg_clean = clean_text(msg)
        msg_vector = vectorizer.transform([msg_clean])
        
        # Get prediction and probability
        pred = model.predict(msg_vector)[0]
        prob = model.predict_proba(msg_vector)[0]
        confidence = np.max(prob) * 100
        
        # Update UI components dynamically based on classification
        if pred == 1:
            # SPAM
            logger.warning(f"Message flagged as SPAM with {confidence:.2f}% confidence.")
            
            # Update labels and progress bar
            self.result_label.configure(
                text=f"🚨 SPAM DETECTED ({confidence:.1f}% Confidence)",
                text_color=("#C62828", "#FF8A80")  # Dark Red / Light Coral
            )
            self.confidence_label.configure(text=f"Confidence Level: {confidence:.1f}%")
            self.confidence_bar.configure(progress_color=("#C62828", "#FF8A80"))
            self.confidence_bar.set(confidence / 100.0)
            
            # Highlight result card border/background
            self.result_frame.configure(
                fg_color=("#FFEBEE", "#3E1E1E"),  # Soft Red shades
                border_color=("#EF9A9A", "#D32F2F")
            )
        else:
            # SAFE (HAM)
            logger.info(f"Message flagged as SAFE with {confidence:.2f}% confidence.")
            
            # Update labels and progress bar
            self.result_label.configure(
                text=f"✅ SAFE (HAM) ({confidence:.1f}% Confidence)",
                text_color=("#2E7D32", "#B9F6CA")  # Dark Green / Bright Green
            )
            self.confidence_label.configure(text=f"Confidence Level: {confidence:.1f}%")
            self.confidence_bar.configure(progress_color=("#2E7D32", "#B9F6CA"))
            self.confidence_bar.set(confidence / 100.0)
            
            # Highlight result card border/background
            self.result_frame.configure(
                fg_color=("#E8F5E9", "#1B3B1B"),  # Soft Green shades
                border_color=("#A5D6A7", "#388E3C")
            )

if __name__ == "__main__":
    logger.info("GUI application starting...")
    app = SpamDetectorApp()
    app.mainloop()
