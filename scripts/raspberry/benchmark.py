import sys, time, json, hashlib, resource
from pathlib import Path
import numpy as np
import psutil
from ai_edge_litert.interpreter import Interpreter
threads = int(sys.argv[1])
model = Path('modelos/fp16_320.tflite')
start = time.perf_counter()
interpreter = Interpreter(model_path=str(model), num_threads=threads)
interpreter.allocate_tensors()
load_ms = (time.perf_counter()-start)*1000
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
x = np.random.default_rng(42).random(inp['shape']).astype(inp['dtype'])
interpreter.set_tensor(inp['index'], x)
for _ in range(3):
    interpreter.invoke()
times = []
for i in range(20):
    start = time.perf_counter()
    interpreter.invoke()
    times.append((time.perf_counter()-start)*1000)
    if (i+1)%5 == 0:
        print(f'threads={threads}: {i+1}/20', flush=True)
y = interpreter.get_tensor(out['index'])
result = dict(threads=threads, input_shape=inp['shape'].tolist(), input_dtype=str(inp['dtype']), output_shape=out['shape'].tolist(), output_dtype=str(out['dtype']), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), warmup_runs=3, measured_runs=20, load_ms=load_ms, mean_ms=float(np.mean(times)), median_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)), fps_network_only=1000/float(np.mean(times)), rss_mib=psutil.Process().memory_info().rss/2**20, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, finite_output=bool(np.isfinite(y).all()), samples_ms=times)
Path('resultados/raspberry/fp16_320').mkdir(parents=True,exist_ok=True)
Path(f'resultados/raspberry/fp16_320/benchmark-{threads}threads.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
