import subprocess
import os
import time
import threading
import sys
import json

VENV_DIR = os.path.join(os.path.dirname(__file__), ".venv")
try:
    from flask import Flask, render_template_string, request, redirect, url_for, jsonify
except ImportError as exc:
    in_venv = getattr(sys, "prefix", "") != getattr(sys, "base_prefix", "")
    if in_venv:
        subprocess.run([sys.executable, "-m", "pip", "install", "flask"], check=True)
        from flask import Flask, render_template_string, request, redirect, url_for, jsonify
    else:
        raise RuntimeError(
            "Flask is not installed in the active interpreter. "
            "Run via 'bash orchestrate.sh' (which creates .venv) or activate a venv first."
        ) from exc

APP_DIR = os.path.dirname(__file__)
VENV_DIR = os.path.join(APP_DIR, ".venv")
BASE_DIR = os.path.join(APP_DIR, "wakeword_lab")
TRAINER_SH = os.path.join(APP_DIR, "trainer.sh")
WORKFLOWS_PATH = os.path.join(APP_DIR, "device_workflows.json")


def load_workflows():
    default = {
        "formats": ["tflite", "onnx"],
        "default_format": "tflite",
        "profile": "medium",
        "threads": 1,
        "notes": "Generic output. Override format/profile as needed.",
    }
    data = {"default": default, "devices": []}
    if os.path.exists(WORKFLOWS_PATH):
        with open(WORKFLOWS_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data["default"].update(loaded.get("default", {}))
        data["devices"] = loaded.get("devices", [])
    if not data["devices"]:
        data["devices"] = [{"id": "custom_manual", "label": "Custom / Manual"}]
    return data


WORKFLOWS = load_workflows()
DEVICE_MAP = {d.get("id"): d for d in WORKFLOWS.get("devices", []) if d.get("id")}
WORKFLOWS_JSON = json.dumps(WORKFLOWS, separators=(",", ":"), ensure_ascii=True)

SAMPLE_PRESETS = [
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
SAMPLE_PRESETS_JSON = json.dumps(SAMPLE_PRESETS, separators=(",", ":"), ensure_ascii=True)
DEFAULT_PRESET_ID = "m"

app = Flask(__name__)
proc = None
current_run_dir = None
last_exit_code = None
proc_lock = threading.Lock()

INDEX_HTML = f"""
<!doctype html>
<title>Wakeword Training Wizard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<div class="container mt-4">
  <h1>Wakeword Training Wizard</h1>
  <form method="post" action="/start">
        <div class="mb-3">
            <label class="form-label">Device workflow</label>
            <select class="form-select" name="device_id" id="device_id"></select>
            <div class="form-text" id="device_notes"></div>
        </div>
    <div class="mb-3">
      <label class="form-label">Wake phrase</label>
      <input class="form-control" name="wake_phrase" value="hey assistant">
    </div>
    <div class="mb-3">
      <label class="form-label">Model format</label>
            <select class="form-select" name="format" id="format"></select>
    </div>
    <div class="mb-3">
      <label class="form-label">Training profile</label>
            <select class="form-select" name="profile" id="profile">
                <option>tiny</option><option selected>medium</option><option>large</option>
            </select>
    </div>
        <div class="mb-3">
            <label class="form-label">Sample size preset</label>
            <select class="form-select" name="sample_preset" id="sample_preset"></select>
            <div class="form-text" id="sample_details"></div>
        </div>
    <div class="mb-3">
      <label class="form-label">CPU threads</label>
            <input class="form-control" name="threads" id="threads" value="1">
    </div>
        <div class="mb-3">
            <label class="form-label">Piper host</label>
            <input class="form-control" name="piper_host" id="piper_host" value="kulfi.local">
        </div>
        <div class="mb-3">
            <label class="form-label">Piper port</label>
            <input class="form-control" name="piper_port" id="piper_port" value="10200">
        </div>
        <div class="mb-3">
            <label class="form-label">openWakeWord host</label>
            <input class="form-control" name="oww_host" id="oww_host" value="kulfi.local">
        </div>
        <div class="mb-3">
            <label class="form-label">openWakeWord port</label>
            <input class="form-control" name="oww_port" id="oww_port" value="10400">
        </div>
    <button class="btn btn-primary" type="submit">Start training</button>
  </form>
  <hr>
  <div id="status">
    <h4>Status</h4>
    <pre id="log">No run started.</pre>
  </div>
</div>
<script>
const workflows = {WORKFLOWS_JSON};
const deviceSelect = document.getElementById('device_id');
const formatSelect = document.getElementById('format');
const profileSelect = document.getElementById('profile');
const threadsInput = document.getElementById('threads');
const piperHostInput = document.getElementById('piper_host');
const piperPortInput = document.getElementById('piper_port');
const owwHostInput = document.getElementById('oww_host');
const owwPortInput = document.getElementById('oww_port');
const deviceNotes = document.getElementById('device_notes');
const samplePresets = {SAMPLE_PRESETS_JSON};
const samplePresetSelect = document.getElementById('sample_preset');
const sampleDetails = document.getElementById('sample_details');

function getDevice(id) {{
    return (workflows.devices || []).find(d => d.id === id);
}}

function setOptions(select, values, selected) {{
    select.innerHTML = '';
    values.forEach(v => {{
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        if (v === selected) opt.selected = true;
        select.appendChild(opt);
    }});
}}

function applyDeviceDefaults() {{
    const deviceId = deviceSelect.value;
    const device = getDevice(deviceId) || {{}};
    const defaults = workflows.default || {{}};
    const formats = device.formats || defaults.formats || ['tflite', 'onnx'];
    const selectedFormat = device.default_format || defaults.default_format || formats[0];
    setOptions(formatSelect, formats, selectedFormat);
    formatSelect.disabled = formats.length === 1 && deviceId !== 'custom_manual';

    const profile = device.profile || defaults.profile || 'medium';
    profileSelect.value = profile;
    profileSelect.disabled = deviceId !== 'custom_manual';

    const threads = device.threads || defaults.threads || 1;
    threadsInput.value = String(threads);
    threadsInput.readOnly = deviceId !== 'custom_manual';

    const piperHost = device.piper_host || defaults.piper_host || 'kulfi.local';
    piperHostInput.value = piperHost;
    piperHostInput.readOnly = deviceId !== 'custom_manual';

    const piperPort = device.piper_port || defaults.piper_port || 10200;
    piperPortInput.value = String(piperPort);
    piperPortInput.readOnly = deviceId !== 'custom_manual';

    const owwHost = device.oww_host || defaults.oww_host || 'kulfi.local';
    owwHostInput.value = owwHost;
    owwHostInput.readOnly = deviceId !== 'custom_manual';

    const owwPort = device.oww_port || defaults.oww_port || 10400;
    owwPortInput.value = String(owwPort);
    owwPortInput.readOnly = deviceId !== 'custom_manual';

    deviceNotes.textContent = device.notes || defaults.notes || '';
}}

function initDevices() {{
    const devices = workflows.devices || [];
    devices.forEach(d => {{
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.label || d.id;
        deviceSelect.appendChild(opt);
    }});
    deviceSelect.value = devices[0] ? devices[0].id : 'custom_manual';
    applyDeviceDefaults();
}}

deviceSelect.addEventListener('change', applyDeviceDefaults);
initDevices();

function setSamplePresets() {{
    samplePresetSelect.innerHTML = '';
    samplePresets.forEach(p => {{
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.label;
        if (p.id === '{DEFAULT_PRESET_ID}') opt.selected = true;
        samplePresetSelect.appendChild(opt);
    }});
    updateSampleDetails();
}}

function updateSampleDetails() {{
    const preset = samplePresets.find(p => p.id === samplePresetSelect.value);
    if (!preset) {{
        sampleDetails.textContent = '';
        return;
    }}
    sampleDetails.textContent = `Approx speed: ${{preset.approx}}. Samples: ${{preset.samples}}. Min per source: ${{preset.min_per_source}}.`;
}}

samplePresetSelect.addEventListener('change', updateSampleDetails);
setSamplePresets();

function fetchLog(){{
  fetch('/log')
    .then(r=>r.json())
    .then(j=>{{ document.getElementById('log').textContent = j.log; }});
}}
setInterval(fetchLog, 2000);
</script>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/start', methods=['POST'])
def start():
    global proc, current_run_dir, last_exit_code
    with proc_lock:
        if proc and proc.poll() is None:
            return "Training already running", 409
        proc = None
        current_run_dir = None
        last_exit_code = None

    wake_phrase = request.form.get('wake_phrase', 'hey assistant')
    device_id = request.form.get('device_id', 'custom_manual')
    profile = request.form.get('profile', 'medium')
    threads = request.form.get('threads', '1')
    model_format = request.form.get('format', 'tflite')
    piper_host = request.form.get('piper_host', 'kulfi.local')
    piper_port = request.form.get('piper_port', '10200')
    oww_host = request.form.get('oww_host', 'kulfi.local')
    oww_port = request.form.get('oww_port', '10400')
    preset_id = request.form.get('sample_preset', DEFAULT_PRESET_ID)

    device = DEVICE_MAP.get(device_id, {})
    defaults = WORKFLOWS.get('default', {})
    if device_id != 'custom_manual':
        profile = device.get('profile', defaults.get('profile', profile))
        threads = device.get('threads', defaults.get('threads', threads))
        model_format = device.get('default_format', defaults.get('default_format', model_format))
        piper_host = device.get('piper_host', defaults.get('piper_host', piper_host))
        piper_port = device.get('piper_port', defaults.get('piper_port', piper_port))
        oww_host = device.get('oww_host', defaults.get('oww_host', oww_host))
        oww_port = device.get('oww_port', defaults.get('oww_port', oww_port))

    try:
        threads = int(threads)
    except (TypeError, ValueError):
        threads = defaults.get('threads', 1)

    try:
        piper_port = int(piper_port)
    except (TypeError, ValueError):
        piper_port = defaults.get('piper_port', 10200)

    try:
        oww_port = int(oww_port)
    except (TypeError, ValueError):
        oww_port = defaults.get('oww_port', 10400)

    os.makedirs(BASE_DIR, exist_ok=True)

    cmd = [
        'bash', TRAINER_SH,
        '--base-dir', BASE_DIR,
        '--allow-low-disk',
        '--wake-phrase', wake_phrase,
        '--train-profile', profile,
        '--train-threads', str(threads),
        '--model-format', model_format,
        '--wyoming-piper-host', str(piper_host),
        '--wyoming-piper-port', str(piper_port),
        '--wyoming-oww-host', str(oww_host),
        '--wyoming-oww-port', str(oww_port),
    ]

    # Launch trainer in background using the venv's python for environment
    env = os.environ.copy()
    env['VENV_DIR'] = VENV_DIR
    preset = next((p for p in SAMPLE_PRESETS if p["id"] == preset_id), None)
    if preset is None:
        preset = next((p for p in SAMPLE_PRESETS if p["id"] == DEFAULT_PRESET_ID), SAMPLE_PRESETS[0])
    env['MAX_POSITIVE_SAMPLES'] = str(preset["max_pos"])
    env['MAX_NEGATIVE_SAMPLES'] = str(preset["max_neg"])
    env['MIN_PER_SOURCE'] = str(preset["min_per_source"])

    # Start as a background process in a new thread so Flask stays responsive
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    out_log = os.path.join(logs_dir, 'trainer_cli.log')
    with open(out_log, 'ab') as f:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.write(f"\n[{stamp}] [wakeword_web] starting run: wake_phrase={wake_phrase!r} profile={profile} format={model_format} threads={threads}\n".encode("utf-8"))

    def run_trainer():
        nonlocal cmd, env, out_log
        global proc, last_exit_code
        exit_code = -1
        process = None
        with open(out_log, 'ab') as f:
            try:
                process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                with proc_lock:
                    proc = process
                exit_code = process.wait()
            finally:
                with proc_lock:
                    if proc is process:
                        proc = None
                    last_exit_code = exit_code

    t = threading.Thread(target=run_trainer, daemon=True)
    t.start()

    # wait for a run directory to appear (longer timeout to allow installs)
    scan_start = time.time()
    run_dir = None
    timeout = 120
    while time.time() - scan_start < timeout:
        runs_dir = os.path.join(BASE_DIR, 'training_runs')
        if os.path.isdir(runs_dir):
            candidates = sorted(os.listdir(runs_dir))
            latest = None
            latest_mtime = 0
            for c in candidates:
                candidate_dir = os.path.join(runs_dir, c)
                start_file = os.path.join(candidate_dir, '.start_time')
                if os.path.exists(start_file):
                    m = os.path.getmtime(start_file)
                    if m > latest_mtime:
                        latest_mtime = m
                        latest = candidate_dir
            if latest and latest_mtime >= scan_start - 1:
                run_dir = latest
                break
        time.sleep(1)

    current_run_dir = run_dir
    return redirect(url_for('index'))

@app.route('/log')
def log():
    with proc_lock:
        is_running = bool(proc and proc.poll() is None)
        exit_code = last_exit_code

    if not current_run_dir:
        # Fallback to CLI-captured trainer output
        out_log = os.path.join(BASE_DIR, 'logs', 'trainer_cli.log')
        if os.path.exists(out_log):
            try:
                with open(out_log, 'r', encoding='utf-8', errors='ignore') as f:
                    data = f.read()[-20000:]
                    status = "Training running..." if is_running else (
                        f"Last run exit code: {exit_code}" if exit_code is not None else "No run started yet."
                    )
                    return jsonify(log=f"{status}\n\n{data}")
            except Exception as e:
                return jsonify(log=f'Error reading fallback log: {e}')
        if exit_code is not None:
            return jsonify(log=f'Last run exit code: {exit_code}')
        return jsonify(log='No run started yet.')

    log_path = os.path.join(current_run_dir, 'training.log')
    if not os.path.exists(log_path):
        # If training hasn't created the run log yet, show CLI output as fallback
        out_log = os.path.join(BASE_DIR, 'logs', 'trainer_cli.log')
        if os.path.exists(out_log):
            try:
                with open(out_log, 'r', encoding='utf-8', errors='ignore') as f:
                    data = f.read()[-20000:]
                    status = "Training running..." if is_running else (
                        f"Last run exit code: {exit_code}" if exit_code is not None else "Waiting for log..."
                    )
                    return jsonify(log=f"{status}\n\n{data}")
            except Exception as e:
                return jsonify(log=f'Error reading fallback log: {e}')
        return jsonify(log='Log not yet created. Waiting...')

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()[-20000:]
        status = "Training running..." if is_running else (
            f"Last run exit code: {exit_code}" if exit_code is not None else "Training status unknown."
        )
        return jsonify(log=f"{status}\n\n{data}")
    except Exception as e:
        return jsonify(log=f'Error reading log: {e}')

if __name__ == '__main__':
    # Ensure venv exists and install minimal packages
    if not os.path.isdir(VENV_DIR):
        import venv
        venv.create(VENV_DIR, with_pip=True)
        pip = os.path.join(VENV_DIR, 'bin', 'pip')
        subprocess.run([pip, 'install', 'flask'], check=True)
    python = os.path.join(VENV_DIR, 'bin', 'python')
    # If running inside the venv already, just run app
    if os.path.realpath(sys.executable) == os.path.realpath(python):
        app.run(port=5000)
    else:
        # Relaunch inside venv
        subprocess.run([python, __file__])
