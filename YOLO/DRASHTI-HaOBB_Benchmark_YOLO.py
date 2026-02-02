import yaml
from pathlib import Path
import tempfile
from ultralytics.utils.benchmarks import benchmark

# --- CONFIGURATION ---

model_paths = [
    'model1.pt',  # Give the path of the model file on which the benchmarking is required
    'model2.pt',
    'model3.pt',
    'model4.pt',
    'model5.pt'
]

original_yaml_path = 'DRASHTI-HaOBB.yaml' # Give the path of the DRASTI dataset yaml file.
imgsz = 1024
device = 0
export_formats = '-'  # leave empty to benchmark all supported formats
verbose = True

# --- PREPARE TEST YAML ---
# Load original YAML
with open(original_yaml_path) as f:
    data = yaml.safe_load(f)
# Point 'val' to 'test' (if 'test' exists)
if 'test' in data:
    data['val'] = data['test']
else:
    raise ValueError("'test' key not found in your dataset YAML. Please ensure your YAML has a test split.")
# Save temporary YAML
temp_yaml = Path(tempfile.mkdtemp()) / 'DRASHTI-HaOBB_test.yaml' # This will create a temp yaml file 
with open(temp_yaml, 'w') as f:
    yaml.safe_dump(data, f)

# --- BENCHMARK LOOP ---
for model_path in model_paths:
    print(f"\nBenchmarking {model_path} on TEST set...")
    results = benchmark(
        model=model_path,
        data=str(temp_yaml),
        imgsz=imgsz,
        device=device,
        format=export_formats,
        verbose=verbose,
    )
    print(results)

print("\nBenchmarking complete. Temporary YAML used:", temp_yaml)
