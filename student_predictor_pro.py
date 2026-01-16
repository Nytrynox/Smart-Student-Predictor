import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from ml_engine import MLEngine
import os
import threading
import queue
from PIL import Image
import time

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AnimatedChart(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.figure = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ani = None

    def animate_bar(self, x, y, title, xlabel, ylabel):
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')
        
        # Style
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('none')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.set_title(title, color='white', pad=20)
        self.ax.set_xlabel(xlabel, color='white')
        self.ax.set_ylabel(ylabel, color='white')

        bars = self.ax.bar(x, [0]*len(y), color='#3b82f6')
        
        def update(frame):
            for bar, val in zip(bars, y):
                current_height = bar.get_height()
                if current_height < val:
                    bar.set_height(current_height + (val/20))
            return bars

        self.ani = animation.FuncAnimation(self.figure, update, frames=20, interval=50, blit=False, repeat=False)
        self.canvas.draw()

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Smart Student Analysis Predictor Pro")
        self.geometry("1400x900")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State
        self.ml_engine = MLEngine()
        self.msg_queue = queue.Queue()
        
        # Layout
        self.create_sidebar()
        self.create_main_area()
        
        # Start background tasks
        self.check_queue()
        self.load_default_data()

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Student\nPredictor Pro", 
                                     font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Navigation
        self.nav_buttons = {}
        items = [
            ("Dashboard", self.show_dashboard),
            ("Analysis", self.show_analysis),
            ("Prediction", self.show_prediction),
            ("Training", self.show_training),
            ("Data", self.show_data)
        ]

        for i, (text, cmd) in enumerate(items):
            btn = ctk.CTkButton(self.sidebar, text=text, command=cmd, 
                              fg_color="transparent", text_color=("gray10", "gray90"), 
                              hover_color=("gray70", "gray30"), anchor="w", height=50,
                              font=ctk.CTkFont(size=14))
            btn.grid(row=i+1, column=0, sticky="ew", padx=10)
            self.nav_buttons[text] = btn

        # Status
        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray50")
        self.status_label.grid(row=6, column=0, padx=20, pady=20, sticky="s")

    def create_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray95", "gray10"))
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Pages
        self.pages = {}
        self.create_dashboard_page()
        self.create_analysis_page()
        self.create_prediction_page()
        self.create_training_page()
        self.create_data_page()
        
        self.show_dashboard()

    def create_dashboard_page(self):
        page = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pages['Dashboard'] = page
        
        # Header
        ctk.CTkLabel(page, text="Dashboard Overview", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", pady=(0, 20))
        
        # Stats Grid
        stats_frame = ctk.CTkFrame(page, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 30))
        
        self.stats_cards = {}
        for i, title in enumerate(["Total Students", "Avg Grade", "Pass Rate"]):
            card = ctk.CTkFrame(stats_frame, height=150)
            card.pack(side="left", fill="x", expand=True, padx=10 if i==1 else 0)
            
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14)).pack(pady=(20, 5))
            lbl = ctk.CTkLabel(card, text="--", font=ctk.CTkFont(size=36, weight="bold"), text_color="#3b82f6")
            lbl.pack(pady=(0, 20))
            self.stats_cards[title] = lbl

        # Animated Chart Area
        self.dash_chart = AnimatedChart(page, height=400)
        self.dash_chart.pack(fill="both", expand=True)

    def create_analysis_page(self):
        page = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pages['Analysis'] = page
        
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(header, text="Deep Data Analysis", font=ctk.CTkFont(size=32, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="Upload CSV", command=self.upload_csv).pack(side="right")
        
        # Tabs for different analyses
        tabview = ctk.CTkTabview(page)
        tabview.pack(fill="both", expand=True)
        tabview.add("Distributions")
        tabview.add("Correlations")
        tabview.add("A-Z Nodes")
        
        # Distributions
        self.dist_frame = ctk.CTkScrollableFrame(tabview.tab("Distributions"))
        self.dist_frame.pack(fill="both", expand=True)
        
        # Correlations
        self.corr_frame = ctk.CTkFrame(tabview.tab("Correlations"))
        self.corr_frame.pack(fill="both", expand=True)
        
        # A-Z Nodes (Decision Tree Viz)
        self.nodes_frame = ctk.CTkFrame(tabview.tab("A-Z Nodes"))
        self.nodes_frame.pack(fill="both", expand=True)
        
        controls = ctk.CTkFrame(self.nodes_frame, fg_color="transparent")
        controls.pack(fill="x", pady=10)
        ctk.CTkLabel(controls, text="Decision Tree Logic Visualization", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=20)
        ctk.CTkButton(controls, text="Generate Tree", command=self.plot_tree).pack(side="right", padx=20)
        
        self.tree_chart = AnimatedChart(self.nodes_frame)
        self.tree_chart.pack(fill="both", expand=True, padx=20, pady=20)

    def plot_tree(self):
        tree = self.ml_engine.get_decision_tree()
        if tree:
            self.tree_chart.ax.clear()
            self.tree_chart.ax.set_facecolor('#2b2b2b')
            self.tree_chart.figure.patch.set_facecolor('#2b2b2b')
            
            from sklearn.tree import plot_tree
            plot_tree(tree, 
                     feature_names=self.ml_engine.feature_columns,
                     filled=True, 
                     rounded=True, 
                     max_depth=3, 
                     fontsize=8,
                     ax=self.tree_chart.ax)
            
            self.tree_chart.ax.set_title("Decision Tree Logic (Depth 3)", color='white', pad=20)
            self.tree_chart.canvas.draw()
        else:
            messagebox.showwarning("Warning", "Please train a Random Forest model first.")

    def create_prediction_page(self):
        page = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pages['Prediction'] = page
        
        ctk.CTkLabel(page, text="Performance Prediction", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", pady=(0, 20))
        
        content = ctk.CTkFrame(page, fg_color="transparent")
        content.pack(fill="both", expand=True)
        
        # Inputs
        left = ctk.CTkScrollableFrame(content, width=400)
        left.pack(side="left", fill="both", padx=(0, 20))
        
        self.inputs = {}
        fields = [
            ('Gender (0=F, 1=M)', 'Gender'),
            ('Attendance Rate (%)', 'AttendanceRate'),
            ('Study Hours/Week', 'StudyHoursPerWeek'),
            ('Previous Grade', 'PreviousGrade'),
            ('Extracurricular (0-3)', 'ExtracurricularActivities'),
            ('Parental Support (0-3)', 'ParentalSupport'),
            ('Daily Study Hours', 'StudyHours'),
            ('Attendance Percent', 'AttendancePercent'),
            ('Online Classes (0/1)', 'OnlineClasses')
        ]
        
        for label, key in fields:
            f = ctk.CTkFrame(left, fg_color="transparent")
            f.pack(fill="x", pady=10)
            ctk.CTkLabel(f, text=label).pack(anchor="w")
            entry = ctk.CTkEntry(f, placeholder_text="0")
            entry.pack(fill="x")
            self.inputs[key] = entry
            
        ctk.CTkButton(left, text="Analyze & Predict", command=self.predict, height=50, 
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20, fill="x")

        # Result
        right = ctk.CTkFrame(content)
        right.pack(side="right", fill="both", expand=True)
        
        ctk.CTkLabel(right, text="Prediction Result", font=ctk.CTkFont(size=20)).pack(pady=30)
        self.score_label = ctk.CTkLabel(right, text="--", font=ctk.CTkFont(size=96, weight="bold"), text_color="gray50")
        self.score_label.pack(pady=20)
        
        self.pred_chart = AnimatedChart(right)
        self.pred_chart.pack(fill="both", expand=True, padx=20, pady=20)

    def create_training_page(self):
        page = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pages['Training'] = page
        
        ctk.CTkLabel(page, text="Model Training", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", pady=(0, 20))
        
        controls = ctk.CTkFrame(page)
        controls.pack(fill="x", pady=(0, 20))
        
        self.model_var = ctk.StringVar(value="Random Forest")
        ctk.CTkOptionMenu(controls, variable=self.model_var, 
                        values=["Random Forest", "Gradient Boosting", "Neural Network"]).pack(side="left", padx=20, pady=20)
        
        ctk.CTkButton(controls, text="Start Training", command=self.start_training).pack(side="left", padx=20)
        
        self.log_text = ctk.CTkTextbox(page, font=("Consolas", 12))
        self.log_text.pack(fill="both", expand=True)

    def create_data_page(self):
        page = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.pages['Data'] = page
        
        ctk.CTkLabel(page, text="Data Management", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", pady=(0, 20))
        
        self.data_text = ctk.CTkTextbox(page, font=("Consolas", 12))
        self.data_text.pack(fill="both", expand=True)

    def show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        
        # Update nav
        for btn_name, btn in self.nav_buttons.items():
            if btn_name == name:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

    # --- Logic ---

    def show_dashboard(self): self.show_page('Dashboard')
    def show_analysis(self): self.show_page('Analysis')
    def show_prediction(self): self.show_page('Prediction')
    def show_training(self): self.show_page('Training')
    def show_data(self): self.show_page('Data')

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                if msg_type == 'status': self.status_label.configure(text=data)
                elif msg_type == 'log': self.log_text.insert("end", data + "\n")
                elif msg_type == 'error': messagebox.showerror("Error", data)
                elif msg_type == 'training_complete': self.on_training_complete(data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_queue)

    def load_default_data(self):
        default_data = "data/student_performance_numeric.csv"
        if os.path.exists(default_data):
            threading.Thread(target=self._load_data_thread, args=(default_data,), daemon=True).start()

    def _load_data_thread(self, filepath):
        success, msg = self.ml_engine.load_data(filepath)
        if success:
            self.msg_queue.put(('status', "Data Loaded"))
            self.after(0, self.refresh_dashboard_data)
        else:
            self.msg_queue.put(('error', msg))

    def refresh_dashboard_data(self):
        if not hasattr(self.ml_engine, 'df'): return
        
        df = self.ml_engine.df
        self.stats_cards['Total Students'].configure(text=str(len(df)))
        self.stats_cards['Avg Grade'].configure(text=f"{df['FinalGrade'].mean():.1f}")
        pass_rate = (df['FinalGrade'] >= 60).mean() * 100
        self.stats_cards['Pass Rate'].configure(text=f"{pass_rate:.1f}%")
        
        # Animate chart
        counts, bins = np.histogram(df['FinalGrade'], bins=10)
        self.dash_chart.animate_bar(bins[:-1], counts, "Grade Distribution", "Grade", "Count")
        
        # Update Data Tab
        self.data_text.delete("1.0", "end")
        self.data_text.insert("end", df.to_string())

    def upload_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            threading.Thread(target=self._load_data_thread, args=(filename,), daemon=True).start()

    def predict(self):
        try:
            data = {}
            for key, entry in self.inputs.items():
                val = entry.get()
                data[key] = float(val) if val else 0.0
                
            success, result = self.ml_engine.predict(data)
            if success:
                score = result['prediction']
                self.score_label.configure(text=f"{score:.1f}")
                
                if score >= 80: color = "#10b981"
                elif score >= 60: color = "#f59e0b"
                else: color = "#ef4444"
                self.score_label.configure(text_color=color)
                
                # Animate importance
                imp = result['importance']
                names = list(imp.keys())
                vals = list(imp.values())
                self.pred_chart.animate_bar(names, vals, "Feature Contribution", "Feature", "Impact")
            else:
                messagebox.showerror("Error", result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_training(self):
        model = self.model_var.get()
        self.log_text.insert("end", f"Training {model}...\n")
        threading.Thread(target=self._train_thread, args=(model,), daemon=True).start()

    def _train_thread(self, model):
        success, result = self.ml_engine.train_model(model)
        if success:
            self.msg_queue.put(('training_complete', result))
        else:
            self.msg_queue.put(('error', result))

    def on_training_complete(self, result):
        metrics = result['metrics']
        self.log_text.insert("end", f"Done! MSE: {metrics['MSE']}, R2: {metrics['R2']}\n")
        messagebox.showinfo("Success", "Training Complete")

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()
