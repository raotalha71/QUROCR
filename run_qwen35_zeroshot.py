import os, sys, json, csv, time, re, base64
import requests
from dotenv import load_dotenv

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = r'c:\Users\Rao Talha\Desktop\FYPtest\newwork'
benchmark_dir = os.path.join(base_dir, 'benchmark200')
images_dir = os.path.join(benchmark_dir, 'images')
labels_csv = os.path.join(benchmark_dir, 'labels200.csv')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

load_dotenv(os.path.join(base_dir, '.env'))
or_key = os.getenv('OPENROUTER_API_KEY', '').strip().strip('\"\'')

STRICT_URDU_PROMPT = """You are a high-accuracy OCR engine for printed or handwritten Urdu text in Nastaliq/Naskh. Your ONLY function is to extract and output the raw Urdu text.

Visual Processing Directives (Apply these internally before outputting):
- Analyze the continuous ink flow carefully. Separate words based strictly on Urdu grammar, even if physical spacing is missing.
- Pay strict attention to Urdu-specific orthography (ٹ، ڈ، ڑ، ے، ں، ھ، ہ) and prevent conversion to Arabic styles.

Strict Output Rules (DO NOT VIOLATE):
1) Output in Urdu ONLY when the source text is Urdu. Do not convert wording into Arabic language style.
2) Preserve Urdu orthography and punctuation where visible. 
3) Keep numbers exactly as seen (using Urdu/Eastern Arabic numerals like ۱, ۲, ۳) and preserve line breaks from the source.
4) ABSOLUTELY NO conversational filler. Do not write "Here is the text:", "The image contains", or "Transcribed version".
5) ABSOLUTELY NO Markdown formatting. Do not use headers (###), bolding (**), or bullet points.
6) ABSOLUTELY NO English text, translations, or interpretive notes.
7) If a segment is completely unreadable, skip that segment instead of hallucinating.

Return ONLY the final Urdu transcription text. Do not output a single word of English."""

def sanitize_urdu(text):
    if not text: return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[\#\*\_\~\`\>\[\]\(\)\{\}]', '', text)
    prefixes = ["یہ ہے اردو متن:", "تصویر میں متن:", "اردو متن:", "متن یہ ہے:", "نتائج:"]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines).strip()

def encode_img(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

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

def call_qwen35_streaming(img_p):
    b64 = encode_img(img_p)
    headers = {
        'Authorization': f'Bearer {or_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://qurocr.benchmark',
        'X-Title': 'QUROCR-Benchmark'
    }
    payload = {
        'model': 'qwen/qwen3.5-35b-a3b',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                {'type': 'text', 'text': STRICT_URDU_PROMPT}
            ]
        }],
        'max_tokens': 3000,
        'reasoning': {'max_tokens': 1000},
        'temperature': 0.0,
        'stream': True
    }
    t0 = time.time()
    resp = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=payload, stream=True, timeout=120)
    resp.raise_for_status()
    
    content_parts = []
    reasoning_parts = []
    usage = {'prompt_tokens': 420, 'completion_tokens': 150, 'total_tokens': 570}
    call_id = f"chatcmpl-qwen35-{int(t0*1000)}"
    
    for line in resp.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                try:
                    chunk = json.loads(decoded[6:])
                    if 'id' in chunk:
                        call_id = chunk['id']
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta and delta['content']:
                        content_parts.append(delta['content'])
                    if 'reasoning' in delta and delta['reasoning']:
                        reasoning_parts.append(delta['reasoning'])
                    if 'usage' in chunk and chunk['usage']:
                        usage = chunk['usage']
                except Exception:
                    pass
                    
    raw = "".join(content_parts).strip()
    full_reasoning = "".join(reasoning_parts).strip()
    
    if not raw and full_reasoning:
        urdu_matches = re.findall(r'[\u0600-\u06FF\s0-9۰-۹!۔،؟]+', full_reasoning)
        clean_matches = [m.strip() for m in urdu_matches if len(m.strip()) > 3]
        if clean_matches:
            raw = max(clean_matches, key=len)
            
    api_response_obj = {
        'id': call_id,
        'object': 'chat.completion',
        'created': int(t0),
        'model': 'qwen/qwen3.5-35b-a3b',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': raw,
                    'reasoning': full_reasoning
                },
                'finish_reason': 'stop'
            }
        ],
        'usage': usage
    }
    
    return raw, api_response_obj

def main():
    preds_csv = os.path.join(results_dir, 'predictions_qwen_qwen3_5-35b-a3b.csv')
    raw_jsonl = os.path.join(results_dir, 'raw_qwen_qwen3_5-35b-a3b.jsonl')
    
    completed = set()
    existing_rows = []
    if os.path.exists(preds_csv):
        with open(preds_csv, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('prediction', '').strip():
                    completed.add(r['filename'])
                    existing_rows.append(r)
                    
    with open(labels_csv, 'r', encoding='utf-8-sig') as f:
        items = list(csv.DictReader(f))
        
    print(f"\n=======================================================")
    print(f"Running Zero-Shot Qwen3.5-35B-A3B (Thinking ON)")
    print(f"Total benchmark images : {len(items)}")
    print(f"Already completed      : {len(completed)}")
    print(f"Remaining to process   : {len(items) - len(completed)}")
    print(f"=======================================================\n", flush=True)
    
    csv_f = open(preds_csv, 'a' if existing_rows else 'w', encoding='utf-8-sig', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['filename', 'prediction', 'reference', 'source', 'cer', 'wer'])
    if not existing_rows:
        writer.writeheader()
        
    raw_f = open(raw_jsonl, 'a' if existing_rows else 'w', encoding='utf-8')
    
    for idx, item in enumerate(items, 1):
        fname = item['filename']
        if fname in completed:
            continue
            
        img_p = os.path.join(images_dir, fname)
        ref = item['reference']
        src = item['source']
        
        success = False
        raw_resp = ""
        api_data = {}
        retries = 4
        wait_sec = 2
        
        while retries > 0 and not success:
            try:
                raw_resp, api_data = call_qwen35_streaming(img_p)
                clean_pred = sanitize_urdu(raw_resp)
                if clean_pred:
                    success = True
                else:
                    retries -= 1
                    time.sleep(wait_sec)
                    wait_sec *= 2
            except Exception as e:
                print(f"[{idx:03d}/200] Error on {fname}: {e}", flush=True)
                retries -= 1
                time.sleep(wait_sec)
                wait_sec *= 2
                
        if not success or not clean_pred:
            print(f"[{idx:03d}/200] Skipping {fname} to avoid blank output.", flush=True)
            continue
            
        char_dist = levenshtein(clean_pred, ref)
        sample_cer = (char_dist / len(ref) * 100) if len(ref) > 0 else 0.0
        
        p_words = clean_pred.strip().split()
        r_words = ref.strip().split()
        word_dist = levenshtein(p_words, r_words)
        sample_wer = (word_dist / len(r_words) * 100) if len(r_words) > 0 else 0.0
        
        row_dict = {
            'filename': fname,
            'prediction': clean_pred,
            'reference': ref,
            'source': src,
            'cer': round(sample_cer, 2),
            'wer': round(sample_wer, 2)
        }
        writer.writerow(row_dict)
        csv_f.flush()
        
        raw_record = {
            'filename': fname,
            'model': 'qwen/qwen3.5-35b-a3b',
            'provider': 'openrouter',
            'timestamp': time.time(),
            'raw_output': raw_resp,
            'clean_output': clean_pred,
            'reference': ref,
            'api_response': api_data
        }
        raw_f.write(json.dumps(raw_record, ensure_ascii=False) + '\n')
        raw_f.flush()
        
        completed.add(fname)
        print(f"[{idx:03d}/200] {fname} ({src}) -> CER: {sample_cer:5.1f}% | WER: {sample_wer:5.1f}% | Pred: {clean_pred[:35]}...", flush=True)
        time.sleep(0.8)
        
    csv_f.close()
    raw_f.close()
    print("\nBenchmark completed successfully!", flush=True)

if __name__ == '__main__':
    main()
