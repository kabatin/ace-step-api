import json
import os
import sys

from acestep import MODE


class AceStepHandler:
    device = "mps"

    def initialize_service(
        self,
        project_root,
        config_path,
        device="auto",
        prefer_source=None,
        use_mlx_dit=True,
    ):
        record = os.environ.get("FAKE_ACESTEP_RECORD")
        if record:
            with open(record, "w") as f:
                json.dump({"config_path": config_path, "use_mlx_dit": use_mlx_dit}, f)
        if MODE == "noisy":
            print("Loading DiT... progress bar 100%|##########|", flush=True)
            sys.stdout.write("no newline at the end")
            sys.stdout.flush()
        if MODE == "load_fail":
            return "❌ checkpoint missing", False
        return f"✅ {config_path} on mps", True
