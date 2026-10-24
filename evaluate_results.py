#!/usr/bin/env python3
"""
evaluate_results.py - QUROCR Benchmark Evaluation Suite
Computes Character Error Rate (CER), Word Error Rate (WER), and Sentence Error Rate (SER)
across all benchmarked models in results/ on both Real and Synthetic subsets.
"""

import os, sys, glob, csv, random, math

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
labels_file = os.path.join(base_dir, 'benchmark200', 'labels200.csv')

labels_map = {}
if os.path.exists(labels_file):
    with open(labels_file, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            labels_map[r['filename']] = r.get('source', 'unknown')

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_metrics(rows):
    if not rows:
        return 0.0, 0.0, 0.0
    tot_char_dist, tot_char_len = 0, 0
    tot_word_dist, tot_word_len = 0, 0
    sent_errors = 0
    
    for r in rows:
        pred = r.get('prediction', '').strip()
        ref = r.get('reference', '').strip()
        cd = levenshtein(pred, ref)
        tot_char_dist += cd
        tot_char_len += len(ref)
        
        pw = pred.split()
        rw = ref.split()
        wd = levenshtein(pw, rw)
        tot_word_dist += wd
        tot_word_len += len(rw)
        
        if pred != ref:
            sent_errors += 1
            
    cer = (tot_char_dist / tot_char_len * 100) if tot_char_len else 0.0
    wer = (tot_word_dist / tot_word_len * 100) if tot_word_len else 0.0
    ser = (sent_errors / len(rows) * 100) if rows else 0.0
    return cer, wer, ser

def main():
    csv_files = sorted(glob.glob(os.path.join(results_dir, '*.csv')))
    summary_rows = []
    
    for p in csv_files:
        fname = os.path.basename(p)
        if fname.startswith('labels'):
            continue
        display_name = fname.replace('predictions_', '').replace('.csv', '')
        
        with open(p, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
            
        for r in rows:
            fn = r.get('filename') or r.get('image', '')
            if fn and not fn.startswith('img'):
                fn = 'img' + fn
            r['filename'] = fn
            if fn in labels_map:
                r['source'] = labels_map[fn]
                
        cer, wer, ser = compute_metrics(rows)
        real_rows = [r for r in rows if r.get('source') == 'real']
        synth_rows = [r for r in rows if r.get('source') == 'synthetic']
        
        real_cer, real_wer, _ = compute_metrics(real_rows)
        synth_cer, synth_wer, _ = compute_metrics(synth_rows)
        
        summary_rows.append({
            'name': display_name,
            'count': len(rows),
            'cer': cer,
            'wer': wer,
            'ser': ser,
            'real_cer': real_cer,
            'synth_cer': synth_cer
        })
        
    summary_rows.sort(key=lambda x: x['cer'])
    
    print("\n" + "=" * 118)
    print(f"{'MODEL / OCR SYSTEM':<36} | {'SAMPLES':<7} | {'CER (%)':<9} | {'WER (%)':<9} | {'SER (%)':<9} | {'REAL CER':<9} | {'SYNTH CER':<9}")
    print("=" * 118)
    for s in summary_rows:
        print(f"{s['name']:<36} | {s['count']:<7} | {s['cer']:8.2f}% | {s['wer']:8.2f}% | {s['ser']:8.2f}% | {s['real_cer']:8.2f}% | {s['synth_cer']:8.2f}%")
    print("=" * 118 + "\n")

if __name__ == '__main__':
    main()
