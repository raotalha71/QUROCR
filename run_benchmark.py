import os, sys, json, csv, time, re, base64, argparse
import requests
from dotenv import load_dotenv

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if running in newwork or scratch
if not os.path.exists(os.path.join(BASE_DIR, 'benchmark200')):
    BASE_DIR = r'c:\Users\Rao Talha\Desktop\FYPtest\newwork'

BENCHMARK_DIR = os.path.join(BASE_DIR, 'benchmark200')
IMAGES_DIR = os.path.join(BENCHMARK_DIR, 'images')
LABELS_CSV = os.path.join(BENCHMARK_DIR, 'labels200.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load environment
load_dotenv(os.path.join(BASE_DIR, '.env'))
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '').strip().strip('\"\'')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '').strip().strip('\"\'')
DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1').strip().strip('\"\'')

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

def sanitize_urdu_output(raw_text):
    if not raw_text:
        return ""
    # 1. Remove thinking/reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 2. Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```', '', text)
    # 3. Remove English characters completely (as requested by user)
    text = re.sub(r'[a-zA-Z]', '', text)
    # 4. Remove Markdown syntax symbols
    text = re.sub(r'[\#\*\_\~\`\>\[\]\(\)\{\}]', '', text)
    # 5. Remove conversational prefixes in Urdu if generated
    prefixes = [
        "یہ ہے اردو متن:",
        "تصویر میں متن:",
        "اردو متن:",
        "متن یہ ہے:",
        "نتائج:"
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    # 6. Normalize whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    return cleaned.strip()

def encode_image(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_openrouter(image_path, model_name, api_key):
    b64_img = encode_image(image_path)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://qurocr.benchmark',
        'X-Title': 'QUROCR-Benchmark'
    }
    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'}},
                    {'type': 'text', 'text': STRICT_URDU_PROMPT}
                ]
            }
        ],
        'max_tokens': 4096
    }
    resp = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    raw_content = data['choices'][0]['message'].get('content', '')
    return raw_content, data

def call_dashscope(image_path, model_name, api_key):
    b64_img = encode_image(image_path)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'}},
                    {'type': 'text', 'text': STRICT_URDU_PROMPT}
                ]
            }
        ],
        'max_tokens': 4096,
        'stream': True
    }
    base = DASHSCOPE_BASE_URL.rstrip('/')
    url = f'{base}/chat/completions'
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=180)
    resp.raise_for_status()
    content_parts = []
    reasoning_parts = []
    full_data = {'model': model_name, 'stream': True}
    for line in resp.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                try:
                    chunk = json.loads(decoded[6:])
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta and delta['content']:
                        content_parts.append(delta['content'])
                    if 'reasoning_content' in delta and delta['reasoning_content']:
                        reasoning_parts.append(delta['reasoning_content'])
                    if 'usage' in chunk:
                        full_data['usage'] = chunk['usage']
                except Exception:
                    pass
    raw_content = ''.join(content_parts)
    full_data['reasoning'] = ''.join(reasoning_parts)
    return raw_content, full_data

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

def run_benchmark(model_name, provider='openrouter', limit=None):
    sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)
    raw_jsonl_p = os.path.join(RESULTS_DIR, f'raw_{sanitized_name}.jsonl')
    preds_csv_p = os.path.join(RESULTS_DIR, f'predictions_{sanitized_name}.csv')
    
    # Check existing progress
    processed = set()
    existing_rows = []
    if os.path.exists(preds_csv_p):
        with open(preds_csv_p, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader:
                processed.add(r['filename'])
                existing_rows.append(r)
    
    # Load labels
    items = []
    with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            items.append(r)
            
    if limit:
        items = items[:limit]
        
    print(f"\n=======================================================")
    print(f"Running Benchmark on {len(items)} samples")
    print(f"Model: {model_name} (Provider: {provider})")
    print(f"Already processed: {len(processed)} samples")
    print(f"=======================================================\n")
    
    csv_f = open(preds_csv_p, 'a' if existing_rows else 'w', encoding='utf-8-sig', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['filename', 'prediction', 'reference', 'source', 'cer', 'wer'])
    if not existing_rows:
        writer.writeheader()
        
    raw_f = open(raw_jsonl_p, 'a', encoding='utf-8')
    
    for idx, item in enumerate(items, 1):
        fname = item['filename']
        if fname in processed:
            continue
            
        img_p = os.path.join(IMAGES_DIR, fname)
        ref = item['reference']
        src = item['source']
        
        retries = 3
        success = False
        raw_resp = ""
        full_json = {}
        
        while retries > 0 and not success:
            try:
                t0 = time.time()
                if provider == 'dashscope':
                    raw_resp, full_json = call_dashscope(img_p, model_name, DASHSCOPE_API_KEY)
                else:
                    raw_resp, full_json = call_openrouter(img_p, model_name, OPENROUTER_API_KEY)
                latency = time.time() - t0
                success = True
            except Exception as e:
                print(f"[{idx}/{len(items)}] Error for {fname} (retries left: {retries-1}): {e}")
                retries -= 1
                time.sleep(3)
                
        if not success:
            print(f"[{idx}/{len(items)}] FAILED {fname} after 3 retries.")
            clean_pred = ""
        else:
            clean_pred = sanitize_urdu_output(raw_resp)
            
        # Compute sample CER and WER
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
        
        # Write raw record
        raw_record = {
            'filename': fname,
            'model': model_name,
            'provider': provider,
            'timestamp': time.time(),
            'raw_output': raw_resp,
            'clean_output': clean_pred,
            'reference': ref,
            'api_response': full_json
        }
        raw_f.write(json.dumps(raw_record, ensure_ascii=False) + '\n')
        raw_f.flush()
        
        print(f"[{idx:03d}/{len(items):03d}] {fname} ({src}) -> CER: {sample_cer:5.1f}% | WER: {sample_wer:5.1f}% | Clean Pred: {clean_pred[:30]}...")
        
    csv_f.close()
    raw_f.close()
    print(f"\nDone! Results saved to:\n  {preds_csv_p}\n  {raw_jsonl_p}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QUROCR Benchmark Runner")
    parser.add_argument('--model', type=str, default='qwen/qwen3.5-35b-a3b', help="Model ID")
    parser.add_argument('--provider', type=str, default='dashscope', choices=['openrouter', 'dashscope'])
    parser.add_argument('--limit', type=int, default=None, help="Limit number of images")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.provider, args.limit)
