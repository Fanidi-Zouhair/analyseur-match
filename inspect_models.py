import os
import sys
from pathlib import Path

def inspect_model(model_path: str):
    print("=" * 80)
    print(f"Inspecting: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: File not found at {model_path}")
        return

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"Import error: {e}")
        return

    try:
        model = YOLO(model_path)
        print(f"Loaded successfully with YOLO.")
        
        # Task type
        task = getattr(model, "task", "unknown")
        print(f"Task: {task}")
        
        # Classes / Names
        names = getattr(model, "names", {})
        print(f"Class Names ({len(names)} classes):")
        if isinstance(names, dict):
            for k, v in sorted(names.items(), key=lambda x: int(x[0]) if isinstance(x[0], int) or str(x[0]).isdigit() else str(x[0])):
                print(f"  [{k}]: '{v}'")
        elif isinstance(names, (list, tuple)):
            for i, name in enumerate(names):
                print(f"  [{i}]: '{name}'")
        else:
            print(f"  {names}")
            
        # Keypoints info (for pose models)
        kpt_shape = getattr(model.model, "kpt_shape", None)
        print(f"Keypoint Shape: {kpt_shape}")
        
        # Overrides / Model Metadata
        overrides = getattr(model, "overrides", {})
        if overrides:
            relevant_keys = ["task", "imgsz", "data", "classes", "kpt_shape"]
            sub_overrides = {k: overrides[k] for k in relevant_keys if k in overrides}
            print(f"Relevant Overrides: {sub_overrides}")
            
        # Model head check
        if hasattr(model, "model") and hasattr(model.model, "yaml"):
            yaml_data = model.model.yaml
            if isinstance(yaml_data, dict):
                print(f"YAML Task/Config: { {k: yaml_data[k] for k in ['nc', 'kpt_shape', 'tasks'] if k in yaml_data} }")

    except Exception as e:
        print(f"Error inspecting model with YOLO: {e}")
        # Fallback to direct torch inspect
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            print("Direct torch checkpoint keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    m = ckpt["model"]
                    print("ckpt['model'] type:", type(m))
                    if hasattr(m, "names"):
                        print("ckpt names:", m.names)
                    if hasattr(m, "kpt_shape"):
                        print("ckpt kpt_shape:", m.kpt_shape)
                    if hasattr(m, "yaml"):
                        print("ckpt yaml:", m.yaml)
        except Exception as e2:
            print(f"Fallback torch load failed: {e2}")

def main():
    models = ["best.pt", "football_keypoints.pt", "football_pitch.pt", "football_pith.pt"]
    cwd = Path.cwd()
    print(f"Current Directory: {cwd}")
    
    found_any = False
    for m in models:
        path = cwd / m
        if path.exists():
            found_any = True
            inspect_model(str(path))
            
    if not found_any:
        print("No expected .pt models found in current directory! Listing .pt files:")
        for p in cwd.glob("*.pt"):
            inspect_model(str(p))

if __name__ == "__main__":
    main()

