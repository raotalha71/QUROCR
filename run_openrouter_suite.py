import os, sys, subprocess

newwork_dir = r'c:\Users\Rao Talha\Desktop\FYPtest\newwork'
runner_script = os.path.join(newwork_dir, 'run_benchmark.py')

# Models to evaluate on OpenRouter
models = [
    'openai/gpt-4o-mini',
    'google/gemini-2.5-flash',
    'anthropic/claude-haiku-4.5',
    'qwen/qwen-2.5-vl-72b-instruct'
]

print("=== Starting OpenRouter Models Benchmark Suite ===")
for m in models:
    print(f"\n>>> Running Benchmark for: {m}")
    cmd = [sys.executable, runner_script, '--provider', 'openrouter', '--model', m]
    try:
        proc = subprocess.run(cmd, cwd=newwork_dir, check=False)
        print(f">>> Finished {m} with exit code {proc.returncode}")
    except Exception as e:
        print(f">>> Error running {m}: {e}")

print("\n=== OpenRouter Suite Completed! ===")
