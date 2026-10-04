"""Run through stdin; all task-specific configuration is passed in environment."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil

config = json.loads(Path(os.environ["CONFIG_TEMPLATE"]).read_text())
config = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in config.items()}
out = Path(config["output"])
run = out.parent
run.mkdir(parents=True, exist_ok=False)
config_path = run / "launch_config.json"
config_path.write_text(json.dumps(config, indent=2) + "\n")
snapshot = run / "source_snapshot"
snapshot.mkdir()
for name in ["core.py", "concepts.py", "run_single.py"]:
    shutil.copy2(Path(os.environ["CODE_ROOT"]) / name, snapshot / name)
if config.get("concept_schema"):
    shutil.copy2(config["concept_schema"], snapshot / "concept_schema.json")
env = os.environ.copy()
env["RUN_CONFIG"] = str(config_path)
with tempfile.NamedTemporaryFile(mode="w", prefix="job_", suffix=".py", dir=os.environ["NEUTRAL_TMP"], delete=False) as entry:
    entry.write('import os, runpy\nrunpy.run_path(os.environ["ENTRYPOINT"], run_name="__main__")\n')
with (run / "process.log").open("w") as log:
    proc = subprocess.Popen([os.environ["PYTHON_EXECUTABLE"], "-u", entry.name], env=env,
                            cwd=os.environ["CODE_ROOT"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"pid": proc.pid, "neutral_entry": entry.name, "config": str(config_path), "output": str(out)}
(run / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
