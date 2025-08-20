import tkinter as tk
from tkinter import ttk # Themed Tkinter for better look
from state_manager import AppMode, SetupStep

class GUIManager:
    """Tkinter를 사용한 그래픽 사용자 인터페이스(GUI) 제어판"""
    def __init__(self, root, app_callbacks):
        self.root = root
        self.app_callbacks = app_callbacks # main.py의 함수들을 호출하기 위한 콜백

        self.root.title("Control Panel")
        self.root.geometry("350x450") # 창 크기
        self.root.resizable(False, False)

        # 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # --- 위젯 변수 ---
        self.sensitivity_var = tk.IntVar(value=25)
        self.noise_var = tk.IntVar(value=3)
        self.min_depth_var = tk.IntVar(value=50)
        self.max_depth_var = tk.IntVar(value=300)
        
        # 프레임 생성
        self.setup_frame = ttk.Frame(self.root, padding="10")
        self.running_frame = ttk.Frame(self.root, padding="10")
        
        self._create_setup_widgets()
        self._create_running_widgets()

        self.show_setup_mode()

    def _create_setup_widgets(self):
        # 1단계: 영역 설정
        step1_frame = ttk.LabelFrame(self.setup_frame, text="Step 1: Set Interaction Area", padding="10")
        step1_frame.pack(fill="x", pady=5)
        ttk.Label(step1_frame, text="In the 'Tracker' window:\n- Left-click to add points (at least 4)\n- Right-click to remove the last point", justify=tk.LEFT).pack()

        # 2단계: 배경 캡처
        step2_frame = ttk.LabelFrame(self.setup_frame, text="Step 2: Capture Background", padding="10")
        step2_frame.pack(fill="x", pady=5)
        self.capture_btn = ttk.Button(step2_frame, text="Capture Background & Define Wall", command=self.app_callbacks['capture_background'])
        self.capture_btn.pack(pady=5)
        self.status_label = ttk.Label(step2_frame, text="Status: Background not captured", foreground="red")
        self.status_label.pack()
        
        # 3단계: 파라미터 튜닝 (슬라이더)
        step3_frame = ttk.LabelFrame(self.setup_frame, text="Step 3: Fine-tune Parameters", padding="10")
        step3_frame.pack(fill="x", pady=5)
        
        # Sensitivity
        ttk.Label(step3_frame, text="Sensitivity").pack(anchor='w')
        ttk.Scale(step3_frame, from_=1, to=100, orient="horizontal", variable=self.sensitivity_var, command=lambda v: self.app_callbacks['update_param']('sensitivity', self.sensitivity_var.get())).pack(fill='x', pady=2)
        
        # Noise Reduction
        ttk.Label(step3_frame, text="Noise Reduction").pack(anchor='w')
        ttk.Scale(step3_frame, from_=0, to=5, orient="horizontal", variable=self.noise_var, command=lambda v: self.app_callbacks['update_param']('noise_reduction', self.noise_var.get())).pack(fill='x', pady=2)
        
        # --- 실행 버튼 ---
        action_frame = ttk.Frame(self.setup_frame)
        action_frame.pack(fill="x", pady=15)
        ttk.Button(action_frame, text="Save & Run", command=self.app_callbacks['start_running']).pack(side="right")
        ttk.Button(action_frame, text="Reset Points", command=self.app_callbacks['reset_points']).pack(side="left")

    def _create_running_widgets(self):
        ttk.Label(self.running_frame, text="Running Mode", font=("Helvetica", 16)).pack(pady=10)
        ttk.Button(self.running_frame, text="Stop and Return to Setup", command=self.app_callbacks['stop_running']).pack(pady=20, ipady=10)

    def update_status(self, is_calibrated):
        if is_calibrated:
            self.status_label.config(text="Status: Background captured!", foreground="green")
        else:
            self.status_label.config(text="Status: Background not captured", foreground="red")

    def show_setup_mode(self):
        self.running_frame.pack_forget()
        self.setup_frame.pack(fill="both", expand=True)

    def show_running_mode(self):
        self.setup_frame.pack_forget()
        self.running_frame.pack(fill="both", expand=True)