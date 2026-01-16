import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from ml_engine import MLEngine
import os
import threading
import queue

# Modern Color Palette
COLORS = {
    'bg_dark': '#1e293b',      # Dark sidebar
    'bg_light': '#f1f5f9',     # Light main content
    'card_bg': '#ffffff',      # White cards
    'primary': '#3b82f6',      # Blue primary
    'primary_hover': '#2563eb',
    'text_dark': '#0f172a',    # Dark text
    'text_light': '#64748b',   # Light text
    'text_white': '#ffffff',   # White text
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Orange
    'danger': '#ef4444',       # Red
    'border': '#e2e8f0'        # Light border
}

class ModernButton(tk.Button):
    def __init__(self, master, text, command, bg_color=COLORS['primary'], fg_color='white', **kwargs):
        super().__init__(master, text=text, command=command, **kwargs)
        self.bg_color = bg_color
        self.hover_color = COLORS['primary_hover'] if bg_color == COLORS['primary'] else bg_color
        
        self.config(
            bg=self.bg_color,
            fg=fg_color,
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=10,
            cursor='hand2',
            activebackground=self.hover_color,
            activeforeground=fg_color
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self['bg'] = self.hover_color

    def on_leave(self, e):
        self['bg'] = self.bg_color

class SidebarButton(tk.Button):
    def __init__(self, master, text, command, icon=None, **kwargs):
        super().__init__(master, text=f"  {text}", command=command, **kwargs)
        self.config(
            bg=COLORS['bg_dark'],
            fg='#94a3b8',
            font=('Segoe UI', 11),
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=12,
            anchor='w',
            cursor='hand2',
            activebackground='#334155',
            activeforeground='white'
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        if self['bg'] != COLORS['primary']:
            self['bg'] = '#334155'
            self['fg'] = 'white'

    def on_leave(self, e):
        if self['bg'] != COLORS['primary']:
            self['bg'] = COLORS['bg_dark']
            self['fg'] = '#94a3b8'
            
    def set_active(self, active=True):
        if active:
            self.config(bg=COLORS['primary'], fg='white')
        else:
            self.config(bg=COLORS['bg_dark'], fg='#94a3b8')

class InputCard(tk.Frame):
    def __init__(self, master, label_text, variable, min_val, max_val, **kwargs):
        super().__init__(master, bg=COLORS['card_bg'], **kwargs)
        
        # Label
        tk.Label(self, text=label_text, font=('Segoe UI', 9, 'bold'), 
                bg=COLORS['card_bg'], fg=COLORS['text_dark']).pack(anchor='w')
        
        # Range hint
        tk.Label(self, text=f"Range: {min_val}-{max_val}", font=('Segoe UI', 8), 
                bg=COLORS['card_bg'], fg=COLORS['text_light']).pack(anchor='w', pady=(0, 5))
        
        # Entry with border frame
        border = tk.Frame(self, bg=COLORS['border'], padx=1, pady=1)
        border.pack(fill='x')
        
        self.entry = tk.Entry(border, textvariable=variable, font=('Segoe UI', 10), 
                            relief='flat', bg='#f8fafc')
        self.entry.pack(fill='x', padx=5, pady=5)

class StudentPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Student Analysis Predictor")
        self.root.geometry("1280x850")
        self.root.configure(bg=COLORS['bg_light'])
        
        # Thread-safe queue
        self.msg_queue = queue.Queue()
        
        # Initialize ML Engine
        self.ml_engine = MLEngine()
        
        # Setup UI
        self.setup_styles()
        self.create_layout()
        
        # Start message checker
        self.check_queue()
        
        # Load default data
        self.load_default_data()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', 
                       background='white', 
                       foreground=COLORS['text_dark'], 
                       rowheight=30, 
                       fieldbackground='white',
                       font=('Segoe UI', 9))
        style.configure('Treeview.Heading', 
                       font=('Segoe UI', 9, 'bold'), 
                       background='#f1f5f9', 
                       foreground=COLORS['text_dark'])
        style.map('Treeview', background=[('selected', COLORS['primary'])])

    def create_layout(self):
        # Main Container
        self.main_container = tk.Frame(self.root, bg=COLORS['bg_light'])
        self.main_container.pack(fill='both', expand=True)
        
        # Sidebar
        self.sidebar = tk.Frame(self.main_container, bg=COLORS['bg_dark'], width=250)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        
        # App Logo/Title
        title_frame = tk.Frame(self.sidebar, bg=COLORS['bg_dark'], pady=30)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="Student\nPredictor", font=('Segoe UI', 20, 'bold'), 
                bg=COLORS['bg_dark'], fg='white', justify='left').pack(padx=20, anchor='w')
        
        # Navigation Buttons
        self.nav_buttons = {}
        nav_items = [
            ('dashboard', 'Dashboard'),
            ('predict', 'Prediction'),
            ('train', 'Model Training'),
            ('data', 'Data Management')
        ]
        
        for key, label in nav_items:
            btn = SidebarButton(self.sidebar, text=label, 
                              command=lambda k=key: self.show_page(k))
            btn.pack(fill='x', pady=2)
            self.nav_buttons[key] = btn
            
        # Status Bar at bottom of sidebar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.sidebar, textvariable=self.status_var, 
                bg=COLORS['bg_dark'], fg='#64748b', 
                font=('Segoe UI', 9), pady=20).pack(side='bottom', fill='x')

        # Content Area
        self.content_area = tk.Frame(self.main_container, bg=COLORS['bg_light'])
        self.content_area.pack(side='left', fill='both', expand=True, padx=30, pady=30)
        
        # Initialize Pages
        self.pages = {}
        self.create_dashboard_page()
        self.create_predict_page()
        self.create_train_page()
        self.create_data_page()
        
        # Show initial page
        self.show_page('dashboard')

    def create_dashboard_page(self):
        page = tk.Frame(self.content_area, bg=COLORS['bg_light'])
        self.pages['dashboard'] = page
        
        # Header
        tk.Label(page, text="Dashboard Overview", font=('Segoe UI', 24, 'bold'), 
                bg=COLORS['bg_light'], fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        # Stats Cards Frame
        stats_frame = tk.Frame(page, bg=COLORS['bg_light'])
        stats_frame.pack(fill='x', pady=(0, 30))
        
        self.stats_labels = {}
        stats_items = [('Total Students', '0'), ('Avg Grade', '0.0'), ('Pass Rate', '0%')]
        
        for i, (label, val) in enumerate(stats_items):
            card = tk.Frame(stats_frame, bg='white', padx=20, pady=20)
            card.pack(side='left', fill='x', expand=True, padx=10 if i == 1 else 0)
            
            tk.Label(card, text=label, font=('Segoe UI', 10), 
                    bg='white', fg=COLORS['text_light']).pack(anchor='w')
            lbl = tk.Label(card, text=val, font=('Segoe UI', 24, 'bold'), 
                         bg='white', fg=COLORS['primary'])
            lbl.pack(anchor='w')
            self.stats_labels[label] = lbl
            
        # Charts Area
        self.dashboard_chart_frame = tk.Frame(page, bg='white', padx=20, pady=20)
        self.dashboard_chart_frame.pack(fill='both', expand=True)
        tk.Label(self.dashboard_chart_frame, text="Performance Analytics", 
                font=('Segoe UI', 14, 'bold'), bg='white', fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 10))

    def create_predict_page(self):
        page = tk.Frame(self.content_area, bg=COLORS['bg_light'])
        self.pages['predict'] = page
        
        tk.Label(page, text="Student Performance Prediction", font=('Segoe UI', 24, 'bold'), 
                bg=COLORS['bg_light'], fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        content = tk.Frame(page, bg=COLORS['bg_light'])
        content.pack(fill='both', expand=True)
        
        # Left: Inputs
        left_panel = tk.Frame(content, bg='white', padx=30, pady=30)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        tk.Label(left_panel, text="Enter Student Details", font=('Segoe UI', 14, 'bold'), 
                bg='white', fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        # Grid for inputs
        input_grid = tk.Frame(left_panel, bg='white')
        input_grid.pack(fill='x')
        
        self.inputs = {}
        fields = [
            ('Gender (0=F, 1=M)', 'Gender', 0, 1),
            ('Attendance Rate (%)', 'AttendanceRate', 0, 100),
            ('Study Hours/Week', 'StudyHoursPerWeek', 0, 168),
            ('Previous Grade', 'PreviousGrade', 0, 100),
            ('Extracurricular (0-3)', 'ExtracurricularActivities', 0, 3),
            ('Parental Support (0-3)', 'ParentalSupport', 0, 3),
            ('Daily Study Hours', 'StudyHours', 0, 24),
            ('Attendance Percent', 'AttendancePercent', 0, 100),
            ('Online Classes (0/1)', 'OnlineClasses', 0, 1)
        ]
        
        for i, (label, key, min_val, max_val) in enumerate(fields):
            var = tk.DoubleVar()
            self.inputs[key] = var
            card = InputCard(input_grid, label, var, min_val, max_val)
            card.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='ew')
            
        input_grid.columnconfigure(0, weight=1)
        input_grid.columnconfigure(1, weight=1)
        
        ModernButton(left_panel, text="Analyze Performance", 
                   command=self.make_prediction).pack(fill='x', pady=30)
        
        # Right: Results
        right_panel = tk.Frame(content, bg='white', padx=30, pady=30)
        right_panel.pack(side='right', fill='both', expand=True, padx=(15, 0))
        
        tk.Label(right_panel, text="Prediction Result", font=('Segoe UI', 14, 'bold'), 
                bg='white', fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        self.score_label = tk.Label(right_panel, text="--", font=('Segoe UI', 64, 'bold'), 
                                  bg='white', fg=COLORS['text_light'])
        self.score_label.pack(pady=20)
        
        tk.Label(right_panel, text="Predicted Final Grade", font=('Segoe UI', 12), 
                bg='white', fg=COLORS['text_light']).pack()
                
        self.chart_frame = tk.Frame(right_panel, bg='white')
        self.chart_frame.pack(fill='both', expand=True, pady=20)

    def create_train_page(self):
        page = tk.Frame(self.content_area, bg=COLORS['bg_light'])
        self.pages['train'] = page
        
        tk.Label(page, text="Model Training", font=('Segoe UI', 24, 'bold'), 
                bg=COLORS['bg_light'], fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        card = tk.Frame(page, bg='white', padx=30, pady=30)
        card.pack(fill='both', expand=True)
        
        # Controls
        controls = tk.Frame(card, bg='white')
        controls.pack(fill='x', pady=(0, 20))
        
        tk.Label(controls, text="Select Algorithm:", font=('Segoe UI', 10, 'bold'), 
                bg='white', fg=COLORS['text_dark']).pack(side='left', padx=(0, 10))
                
        self.model_var = tk.StringVar(value='Random Forest')
        models = ['Random Forest', 'Gradient Boosting', 'Neural Network']
        ttk.OptionMenu(controls, self.model_var, models[0], *models).pack(side='left')
        
        ModernButton(controls, text="Train Model", 
                   command=self.start_training).pack(side='left', padx=20)
                   
        # Log Area
        tk.Label(card, text="Training Logs", font=('Segoe UI', 12, 'bold'), 
                bg='white', fg=COLORS['text_dark']).pack(anchor='w', pady=(20, 10))
                
        self.log_text = tk.Text(card, font=('Consolas', 10), bg='#f8fafc', 
                              relief='flat', padx=15, pady=15)
        self.log_text.pack(fill='both', expand=True)

    def create_data_page(self):
        page = tk.Frame(self.content_area, bg=COLORS['bg_light'])
        self.pages['data'] = page
        
        tk.Label(page, text="Data Management", font=('Segoe UI', 24, 'bold'), 
                bg=COLORS['bg_light'], fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        card = tk.Frame(page, bg='white', padx=30, pady=30)
        card.pack(fill='both', expand=True)
        
        # Toolbar
        toolbar = tk.Frame(card, bg='white')
        toolbar.pack(fill='x', pady=(0, 20))
        
        ModernButton(toolbar, text="Load CSV File", 
                   command=self.load_csv).pack(side='left')
                   
        # Treeview
        columns = self.ml_engine.feature_columns + [self.ml_engine.target_column]
        self.tree = ttk.Treeview(card, columns=columns, show='headings')
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        scrollbar = ttk.Scrollbar(card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def show_page(self, page_name):
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()
            
        # Show selected page
        self.pages[page_name].pack(fill='both', expand=True)
        
        # Update nav buttons
        for key, btn in self.nav_buttons.items():
            btn.set_active(key == page_name)
            
        # Refresh content if needed
        if page_name == 'dashboard':
            self.refresh_dashboard()

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                if msg_type == 'log':
                    self.log_text.insert('end', data + '\n')
                    self.log_text.see('end')
                elif msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'training_complete':
                    self.on_training_complete(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def load_default_data(self):
        default_data = "data/student_performance_numeric.csv"
        if os.path.exists(default_data):
            threading.Thread(target=self._load_data_thread, args=(default_data,), daemon=True).start()

    def _load_data_thread(self, filepath):
        success, msg = self.ml_engine.load_data(filepath)
        if success:
            self.msg_queue.put(('status', "Data loaded successfully"))
            self.root.after(0, self.refresh_data_table)
            # Auto train
            self.msg_queue.put(('log', "Auto-training initial model..."))
            success, result = self.ml_engine.train_model()
            if success:
                self.msg_queue.put(('log', "Initial model trained."))
        else:
            self.msg_queue.put(('error', msg))

    def start_training(self):
        model_type = self.model_var.get()
        self.log_text.insert('end', f"Starting training for {model_type}...\n")
        self.status_var.set("Training in progress...")
        
        threading.Thread(target=self._train_thread, args=(model_type,), daemon=True).start()

    def _train_thread(self, model_type):
        try:
            success, result = self.ml_engine.train_model(model_type)
            if success:
                self.msg_queue.put(('training_complete', result))
            else:
                self.msg_queue.put(('error', result))
        except Exception as e:
            self.msg_queue.put(('error', str(e)))

    def on_training_complete(self, result):
        metrics = result['metrics']
        msg = f"Training Complete!\nMSE: {metrics['MSE']}\nR2 Score: {metrics['R2']}\nMAE: {metrics['MAE']}"
        self.log_text.insert('end', msg + "\n" + "-"*50 + "\n")
        self.status_var.set("Ready")
        messagebox.showinfo("Success", "Model trained successfully!")

    def make_prediction(self):
        try:
            data = {key: var.get() for key, var in self.inputs.items()}
            success, result = self.ml_engine.predict(data)
            
            if success:
                score = result['prediction']
                self.score_label.config(text=f"{score:.1f}")
                
                if score >= 80: color = COLORS['success']
                elif score >= 60: color = COLORS['warning']
                else: color = COLORS['danger']
                self.score_label.config(fg=color)
                
                self.plot_importance(result['importance'])
            else:
                messagebox.showerror("Error", result)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def plot_importance(self, importance_dict):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        if not importance_dict: return

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        names = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        # Sort
        sorted_idx = np.argsort(values)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        ax.barh(pos, np.array(values)[sorted_idx], align='center', color=COLORS['primary'])
        ax.set_yticks(pos)
        ax.set_yticklabels(np.array(names)[sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def refresh_dashboard(self):
        if not hasattr(self.ml_engine, 'df'): return
        
        df = self.ml_engine.df
        
        # Update stats
        self.stats_labels['Total Students'].config(text=str(len(df)))
        self.stats_labels['Avg Grade'].config(text=f"{df['FinalGrade'].mean():.1f}")
        pass_rate = (df['FinalGrade'] >= 60).mean() * 100
        self.stats_labels['Pass Rate'].config(text=f"{pass_rate:.1f}%")
        
        # Update charts
        for widget in self.dashboard_chart_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)): widget.destroy()
            
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        # Grade Dist
        sns.histplot(df['FinalGrade'], kde=True, ax=axs[0], color=COLORS['primary'])
        axs[0].set_title('Grade Distribution')
        
        # Study vs Grade
        sns.scatterplot(data=df, x='StudyHoursPerWeek', y='FinalGrade', ax=axs[1], color=COLORS['success'])
        axs[1].set_title('Study Hours vs Grade')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.dashboard_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            threading.Thread(target=self._load_data_thread, args=(filename,), daemon=True).start()

    def refresh_data_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if hasattr(self.ml_engine, 'df'):
            for _, row in self.ml_engine.df.head(100).iterrows():
                self.tree.insert('', 'end', values=list(row[self.ml_engine.feature_columns + [self.ml_engine.target_column]]))

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPredictorGUI(root)
    root.mainloop()
