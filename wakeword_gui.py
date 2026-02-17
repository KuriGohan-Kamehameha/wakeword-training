import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import json

class WakewordWizard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wakeword Training Wizard")
        self.geometry("420x700")
        self.resizable(False, False)
        self.workflows = self.load_workflows()
        self.device_map = {d.get("id"): d for d in self.workflows.get("devices", []) if d.get("id")}
        self.sample_presets = [
            {
                "id": "xs",
                "label": "XS (quick)",
                "max_pos": 50,
                "max_neg": 200,
                "min_per_source": 1,
                "approx": "~5-10 min",
                "samples": "50 pos / 200 neg",
            },
            {
                "id": "s",
                "label": "S (fast)",
                "max_pos": 100,
                "max_neg": 400,
                "min_per_source": 2,
                "approx": "~10-20 min",
                "samples": "100 pos / 400 neg",
            },
            {
                "id": "m",
                "label": "M (balanced)",
                "max_pos": 250,
                "max_neg": 1000,
                "min_per_source": 3,
                "approx": "~25-45 min",
                "samples": "250 pos / 1000 neg",
            },
            {
                "id": "l",
                "label": "L (thorough)",
                "max_pos": 500,
                "max_neg": 2000,
                "min_per_source": 4,
                "approx": "~45-90 min",
                "samples": "500 pos / 2000 neg",
            },
            {
                "id": "xl",
                "label": "XL (max)",
                "max_pos": 1000,
                "max_neg": 4000,
                "min_per_source": 5,
                "approx": "~90-150 min",
                "samples": "1000 pos / 4000 neg",
            },
        ]
        self.create_widgets()

    def load_workflows(self):
        default = {
            "formats": ["tflite", "onnx"],
            "default_format": "tflite",
            "profile": "medium",
            "threads": 1,
            "notes": "Generic output. Override format/profile as needed.",
        }
        data = {"default": default, "devices": []}
        path = os.path.join(os.path.dirname(__file__), "device_workflows.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            data["default"].update(loaded.get("default", {}))
            data["devices"] = loaded.get("devices", [])
        if not data["devices"]:
            data["devices"] = [{"id": "custom_manual", "label": "Custom / Manual"}]
        return data

    def create_widgets(self):
        # Device workflow
        tk.Label(self, text="Device Workflow:").pack(pady=5)
        self.device_var = tk.StringVar(value=self.workflows["devices"][0]["id"])
        device_labels = [d.get("label", d.get("id")) for d in self.workflows["devices"]]
        self.device_combo = ttk.Combobox(self, values=device_labels, state="readonly")
        self.device_combo.pack(pady=5)
        self.device_combo.current(0)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        self.device_notes = tk.Label(self, text="", wraplength=360, justify="left")
        self.device_notes.pack(pady=5)

        # Wakeword
        tk.Label(self, text="Enter Wakeword:").pack(pady=5)
        self.wakeword_entry = tk.Entry(self, width=30)
        self.wakeword_entry.pack(pady=5)
        self.wakeword_entry.insert(0, "hey assistant")

        # Model format
        tk.Label(self, text="Select Model Format:").pack(pady=5)
        self.format_var = tk.StringVar(value="tflite")
        self.format_combo = ttk.Combobox(self, textvariable=self.format_var, values=["tflite", "onnx"], state="readonly")
        self.format_combo.pack(pady=5)

        # Training profile
        tk.Label(self, text="Select Training Profile:").pack(pady=5)
        self.profile_var = tk.StringVar(value="medium")
        self.profile_combo = ttk.Combobox(self, textvariable=self.profile_var, values=["tiny", "medium", "large"], state="readonly")
        self.profile_combo.pack(pady=5)

        # Sample size preset
        tk.Label(self, text="Sample Size Preset:").pack(pady=5)
        self.sample_var = tk.StringVar(value=self.sample_presets[2]["label"])
        preset_labels = [p["label"] for p in self.sample_presets]
        self.sample_combo = ttk.Combobox(self, textvariable=self.sample_var, values=preset_labels, state="readonly")
        self.sample_combo.pack(pady=5)
        self.sample_combo.bind("<<ComboboxSelected>>", self.on_sample_change)

        self.sample_details = tk.Label(self, text="", wraplength=360, justify="left")
        self.sample_details.pack(pady=5)

        # Threads
        tk.Label(self, text="CPU Threads:").pack(pady=5)
        self.threads_entry = tk.Entry(self, width=10)
        self.threads_entry.pack(pady=5)
        self.threads_entry.insert(0, "1")

        # Piper host
        tk.Label(self, text="Piper Host:").pack(pady=5)
        self.piper_host_entry = tk.Entry(self, width=30)
        self.piper_host_entry.pack(pady=5)
        self.piper_host_entry.insert(0, "kulfi.local")

        # Piper port
        tk.Label(self, text="Piper Port:").pack(pady=5)
        self.piper_port_entry = tk.Entry(self, width=10)
        self.piper_port_entry.pack(pady=5)
        self.piper_port_entry.insert(0, "10200")

        # openWakeWord host
        tk.Label(self, text="openWakeWord Host:").pack(pady=5)
        self.oww_host_entry = tk.Entry(self, width=30)
        self.oww_host_entry.pack(pady=5)
        self.oww_host_entry.insert(0, "kulfi.local")

        # openWakeWord port
        tk.Label(self, text="openWakeWord Port:").pack(pady=5)
        self.oww_port_entry = tk.Entry(self, width=10)
        self.oww_port_entry.pack(pady=5)
        self.oww_port_entry.insert(0, "10400")

        # Start button
        tk.Button(self, text="Start Training", command=self.start_training).pack(pady=20)
        self.apply_device_defaults()
        self.apply_sample_details()

    def current_device(self):
        label = self.device_combo.get()
        for d in self.workflows.get("devices", []):
            if d.get("label", d.get("id")) == label:
                return d
        return {"id": "custom_manual"}

    def apply_device_defaults(self):
        device = self.current_device()
        defaults = self.workflows.get("default", {})
        device_id = device.get("id", "custom_manual")

        formats = device.get("formats") or defaults.get("formats") or ["tflite", "onnx"]
        default_format = device.get("default_format") or defaults.get("default_format") or formats[0]
        self.format_combo["values"] = formats
        self.format_var.set(default_format)
        self.format_combo.configure(state="readonly" if (device_id == "custom_manual" or len(formats) > 1) else "disabled")

        profile = device.get("profile", defaults.get("profile", "medium"))
        self.profile_var.set(profile)
        self.profile_combo.configure(state="readonly" if device_id == "custom_manual" else "disabled")

        threads = str(device.get("threads", defaults.get("threads", 1)))
        self.threads_entry.delete(0, tk.END)
        self.threads_entry.insert(0, threads)
        self.threads_entry.configure(state="normal" if device_id == "custom_manual" else "readonly")

        self.device_notes.configure(text=device.get("notes", defaults.get("notes", "")))

        piper_host = device.get("piper_host", defaults.get("piper_host", "kulfi.local"))
        self.piper_host_entry.delete(0, tk.END)
        self.piper_host_entry.insert(0, piper_host)
        self.piper_host_entry.configure(state="normal" if device_id == "custom_manual" else "readonly")

        piper_port = str(device.get("piper_port", defaults.get("piper_port", 10200)))
        self.piper_port_entry.delete(0, tk.END)
        self.piper_port_entry.insert(0, piper_port)
        self.piper_port_entry.configure(state="normal" if device_id == "custom_manual" else "readonly")

        oww_host = device.get("oww_host", defaults.get("oww_host", "kulfi.local"))
        self.oww_host_entry.delete(0, tk.END)
        self.oww_host_entry.insert(0, oww_host)
        self.oww_host_entry.configure(state="normal" if device_id == "custom_manual" else "readonly")

        oww_port = str(device.get("oww_port", defaults.get("oww_port", 10400)))
        self.oww_port_entry.delete(0, tk.END)
        self.oww_port_entry.insert(0, oww_port)
        self.oww_port_entry.configure(state="normal" if device_id == "custom_manual" else "readonly")

    def on_device_change(self, _event):
        self.apply_device_defaults()

    def on_sample_change(self, _event):
        self.apply_sample_details()

    def apply_sample_details(self):
        preset = self.current_sample_preset()
        details = (
            f"Approx speed: {preset['approx']}. "
            f"Samples: {preset['samples']}. "
            f"Min per source: {preset['min_per_source']}."
        )
        self.sample_details.configure(text=details)

    def current_sample_preset(self):
        label = self.sample_combo.get()
        for preset in self.sample_presets:
            if preset["label"] == label:
                return preset
        return self.sample_presets[2]

    def start_training(self):
        wakeword = self.wakeword_entry.get().strip()
        model_format = self.format_var.get()
        profile = self.profile_var.get()
        threads = self.threads_entry.get().strip()
        if not wakeword:
            messagebox.showerror("Input Error", "Wakeword cannot be empty.")
            return
        if not threads.isdigit() or int(threads) < 1:
            messagebox.showerror("Input Error", "Threads must be a positive integer.")
            return
        piper_host = self.piper_host_entry.get().strip()
        piper_port = self.piper_port_entry.get().strip()
        oww_host = self.oww_host_entry.get().strip()
        oww_port = self.oww_port_entry.get().strip()
        if piper_port and (not piper_port.isdigit() or int(piper_port) < 1):
            messagebox.showerror("Input Error", "Piper port must be a positive integer.")
            return
        if oww_port and (not oww_port.isdigit() or int(oww_port) < 1):
            messagebox.showerror("Input Error", "openWakeWord port must be a positive integer.")
            return
        orchestrate_path = os.path.join(os.path.dirname(__file__), "orchestrate.sh")
        cmd = ["bash", orchestrate_path]
        env = os.environ.copy()
        env["WAKE_WORD"] = wakeword
        env["MODEL_FORMAT"] = model_format
        env["PROFILE"] = profile
        env["THREADS"] = threads
        if piper_host:
            env["WYOMING_PIPER_HOST"] = piper_host
        if piper_port:
            env["WYOMING_PIPER_PORT"] = piper_port
        if oww_host:
            env["WYOMING_OWW_HOST"] = oww_host
        if oww_port:
            env["WYOMING_OWW_PORT"] = oww_port
        preset = self.current_sample_preset()
        env["MAX_POSITIVE_SAMPLES"] = str(preset["max_pos"])
        env["MAX_NEGATIVE_SAMPLES"] = str(preset["max_neg"])
        env["MIN_PER_SOURCE"] = str(preset["min_per_source"])
        try:
            subprocess.run(cmd, env=env, check=True)
            messagebox.showinfo("Success", f"Training complete! Model saved in ~/wakeword_lab/custom_models.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Training Failed", f"Error: {e}")

if __name__ == "__main__":
    import venv
    import sys
    import subprocess
    import shutil
    venv_dir = os.path.join(os.path.dirname(__file__), "wakeword_gui_venv")
    if not os.path.exists(venv_dir):
        venv.create(venv_dir, with_pip=True)
        subprocess.run([os.path.join(venv_dir, "bin", "python"), "-m", "pip", "install", "tkinter", "ttk", "wheel"], check=False)
    python_exe = os.path.join(venv_dir, "bin", "python")
    if sys.executable != python_exe:
        # Relaunch in venv
        subprocess.run([python_exe, __file__])
        sys.exit(0)
    app = WakewordWizard()
    app.mainloop()
