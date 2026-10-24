import os, sys, csv, glob

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')

print("\n" + "="*70)
print("             QUROCR BENCHMARK - LIVE PROGRESS MONITOR")
print("="*70)

if not os.path.exists(results_dir):
    print("No results directory found yet.")
    sys.exit(0)

csv_files = glob.glob(os.path.join(results_dir, "predictions_*.csv"))

if not csv_files:
    print("No prediction files found yet.")
    sys.exit(0)

for p in sorted(csv_files):
    fname = os.path.basename(p)
    model_name = fname.replace("predictions_", "").replace(".csv", "")
    with open(p, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        count = len(reader)
        pct = (count / 200) * 100
        
        # Calculate running CER
        total_dist = 0
        total_len = 0
        for r in reader:
            pred = r.get("prediction", "")
            ref = r.get("reference", "")
            # calculate sample dist
            m, n = len(pred), len(ref)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1): dp[i][0] = i
            for j in range(n+1): dp[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if pred[i-1] == ref[j-1]: dp[i][j] = dp[i-1][j-1]
                    else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            total_dist += dp[m][n]
            total_len += len(ref)
            
        running_cer = (total_dist / total_len * 100) if total_len > 0 else 0.0
        
        bar_len = 20
        filled = int(bar_len * count // 200)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\nModel: {model_name}")
        print(f"  Progress: [{bar}] {count:3d}/200 ({pct:5.1f}%) | Running CER: {running_cer:5.2f}%")
        if reader:
            last = reader[-1]
            print(f"  Latest Sample : {last.get('filename')} (CER: {last.get('cer')}%)")
            print(f"  Prediction    : {last.get('prediction')}")
            print(f"  Reference     : {last.get('reference')}")

print("\n" + "="*70 + "\n")
