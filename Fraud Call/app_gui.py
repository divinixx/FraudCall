import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
from utils.model import load_model
from utils.predictor import predict_fraud

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Call Detection System")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")
        
        # Load model
        try:
            model_path = "D:\\Python\\AIML\\Fraud Call\\models\\fraud_model.pkl"
            vectorizer_path = "D:\\Python\\AIML\\Fraud Call\\models\\tfidf_vectorizer.pkl"
            self.model, self.vectorizer = load_model(model_path, vectorizer_path)
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="Fraud Call Detection System", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Enter call transcript below to analyze:",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        instructions.pack(anchor="w", padx=20)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=60,
            height=10,
            font=("Arial", 10)
        )
        self.text_input.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Analyze button
        analyze_button = tk.Button(
            self.root,
            text="Analyze Call",
            command=self.analyze_text,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        analyze_button.pack(pady=10)
        
        # Clear button
        clear_button = tk.Button(
            self.root,
            text="Clear",
            command=self.clear_text,
            bg="#f44336",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        clear_button.pack(pady=5)
        
        # Result frame
        result_frame = tk.Frame(self.root, bg="#f0f0f0")
        result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Result label
        self.result_label = tk.Label(
            result_frame,
            text="Results will appear here",
            font=("Arial", 12),
            bg="#f0f0f0",
            wraplength=550,
            justify="left"
        )
        self.result_label.pack(anchor="w")
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready" if self.model_loaded else "Error: Model not loaded",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#dcdcdc" if self.model_loaded else "#ffcccc"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check if model is loaded
        if not self.model_loaded:
            messagebox.showerror(
                "Error", 
                "Model files not found. Please train the model first by running fraud_call.py"
            )
    
    def analyze_text(self):
        if not self.model_loaded:
            messagebox.showerror(
                "Error", 
                "Model not loaded. Cannot analyze text."
            )
            return
        
        # Get text from input
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        # Update status
        self.status_bar.config(text="Analyzing...", bg="#fff9c4")
        self.root.update()
        
        try:
            # Make prediction
            prediction, probability = predict_fraud(text, self.model, self.vectorizer)
            
            # Format result
            if prediction == 1:
                result_text = f"⚠️ FRAUD ALERT: This call is likely fraudulent\nConfidence: {probability:.2%}"
                self.result_label.config(fg="#d32f2f")
            else:
                result_text = f"✅ LEGITIMATE: This call appears to be legitimate\nConfidence: {(1-probability):.2%}"
                self.result_label.config(fg="#388e3c")
            
            # Update result label
            self.result_label.config(text=result_text)
            
            # Update status
            self.status_bar.config(text="Analysis complete", bg="#dcdcdc")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text=f"Error: {str(e)}", bg="#ffcccc")
    
    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Results will appear here", fg="black")
        self.status_bar.config(text="Ready", bg="#dcdcdc")

def main():
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()