from flask import Flask, render_template, request
import joblib
import socket
from urllib.parse import urlparse
from datetime import datetime
import sys
import os
import re
import xgboost as xgb
from sklearn.pipeline import FeatureUnion

# ── Thêm thư mục gốc vào sys.path để import ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Import các module ────────────────────────
from data_and_apis.external_api_fetcher import extract_features_from_apis
from ML_components.utils import remove_vietnamese_diacritics, SUSPICIOUS_KEYWORDS, strip_scheme_www, normalize_url
from ML_components.title_feature_extractor import TitleFeatureExtractor
from ML_components.url_feature_extractor import URLFeatureExtractor

# ── Patch cho joblib ─────────────────────────
sys.modules['__main__'].TitleFeatureExtractor = TitleFeatureExtractor
sys.modules['__main__'].URLFeatureExtractor = URLFeatureExtractor

app = Flask(__name__)

# ── Load model và vectorizer ────────────────
print("[+] Đang load model và vectorizer...")
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("[✓] Đã load xong model và vectorizer.")

# ── API keys ────────────────────────────────
API_KEYS = {
    "google":     "",
    "virustotal": "",
    "ipinfo":     "",
}

# ── Kiểm tra model predict được không ───────
try:
    print("[+] Đang kiểm tra model có hoạt động...")
    dummy_url = "http://test.com"
    processed = strip_scheme_www(dummy_url)
    title_mapping = {processed: "test title"}

    vectorizer_base = vectorizer.transformer_list[0][1]
    feature_union = FeatureUnion([
        ('tfidf', vectorizer_base),
        ('url_custom', URLFeatureExtractor()),
        ('title_custom', TitleFeatureExtractor(title_mapping=title_mapping))
    ])
    X_vec = feature_union.transform([processed])
    dmatrix = xgb.DMatrix(X_vec)
    _ = model.predict(dmatrix)[0]
    print("[✓] Model hoạt động tốt trên Flask!")
except Exception as e:
    print("[✗] Model không chạy được! Lỗi:", e)

# ── Kiểm tra API keys ──────────────────────
print("[+] Kiểm tra API keys...")
for name, key in API_KEYS.items():
    print(f"  - {name}: {'Có' if key else 'Thiếu'}")
print("[✓] Kiểm tra API keys hoàn tất.")

# ── Hàm tiện ích ───────────────────────────
def check_domain_exists(url: str) -> bool:
    try:
        hostname = urlparse(url).netloc or url
        socket.gethostbyname(hostname)
        return True
    except:
        return False

def dedup_sub_keywords(hit_words):
    hit_words = sorted(hit_words, key=len, reverse=True)
    unique_hits = []
    for w in hit_words:
        if not any(w in longer for longer in unique_hits):
            unique_hits.append(w)
    return unique_hits


def calculate_risk_score(ai_prediction: int, api_features: dict, url: str = "", title: str = "", domain_ok: bool = True) -> tuple[int, int]:
    url_clean   = remove_vietnamese_diacritics(url.lower())
    title_clean = remove_vietnamese_diacritics(title.lower())

    # Kiểm tra URL là địa chỉ IP
    is_ip_url = bool(re.fullmatch(r"(http[s]?://)?(\d{1,3}\.){3}\d{1,3}(/.*)?", url.strip()))

    # Tính keyword hits
    keywords = set(SUSPICIOUS_KEYWORDS)
    url_raw_hits   = [] if is_ip_url else [w for w in keywords if w in url_clean]
    title_raw_hits = [w for w in keywords if w in title_clean]
    url_hit_words   = dedup_sub_keywords(url_raw_hits)
    title_hit_words = dedup_sub_keywords(title_raw_hits)
    url_hits        = len(url_hit_words)
    title_hits      = len(title_hit_words)
    has_url_kw      = url_hits > 0
    has_title_kw    = title_hits > 0
    both_kw         = has_url_kw and has_title_kw
    any_kw          = has_url_kw or has_title_kw

    # Các giá trị API
    sb = api_features.get("safe_browsing", 0)
    wr = api_features.get("web_risk", 0)
    vt = api_features.get("virustotal_malicious", 0)
    api_alert  = sb > 0 or wr > 0 or vt > 0
    google_cnt = int(sb > 0) + int(wr > 0)

    score     = 0.0
    ai_score  = int(ai_prediction >= 0.7)

    if ai_score == 0:
        if sb and wr and vt >= 1:
            ai_score = 1
            score += 10
        elif vt > 4 and (sb or wr):
            ai_score = 1
            score += 8

    if ai_score == 0 and not api_alert:
        if both_kw:
            score += 4
        elif any_kw:
            score += url_hits + title_hits

    if api_alert and any_kw:
        if google_cnt == 2:
            score += 9 if both_kw else 7
        elif google_cnt == 1:
            score += 7 if both_kw else 5

    if ai_score == 0 and not any_kw:
        if sb: score += 4
        if wr: score += 4
        if vt >= 7:
            score = max(score, 10)
        elif 1 <= vt <= 6:
            score += 2 + (vt - 1)
            score = min(score, 6)

    if ai_score == 1 and not api_alert:
        score += 4
        if both_kw:
            score += url_hits + title_hits
        elif any_kw:
            score += 0.5 * (url_hits + title_hits)

    if ai_score == 1 and api_alert:
        score += 4
        score += google_cnt * 2
        score += vt

    # Tính thêm đặc điểm tên miền
    if is_ip_url:
        score += 3
    else:
        dom = strip_scheme_www(url)
        if re.search(r"(\d)\1{1,}", dom):  # số lặp lại
            score += 1
        if len(set(re.findall(r"\d", dom))) >= 3:
            score += 1

    if not domain_ok:
        score += 2
        
    if ai_score == 0 and score >= 7:
        ai_score = 1
        score += 1

    final_score = min(int(round(score)), 10)

    # In ra màn hình phân tích chi tiết
    print("------ Phân tích URL ------")
    print(f"URL: {url}")
    print(f"AI cảnh báo: {ai_score} | SB: {sb}, WR: {wr}, VT: {vt}")
    print("Từ khóa nghi ngờ trong URL:  ", url_hit_words if not is_ip_url else "(bỏ qua - IP)")
    print("Từ khóa nghi ngờ trong Title:", title_hit_words)
    print(f"Keyword hits: URL={url_hits}, Title={title_hits} | Domain OK: {domain_ok}")
    print(f"→ Tổng điểm đánh giá: {final_score}")
    print("---------------------------\n")

    return final_score, ai_score


# ── Context Processor ───────────────────────
@app.context_processor
def inject_now():
    return {"now": lambda: datetime.now().strftime("%H:%M %d-%m-%Y")}

# ── Route ───────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_url = ""
    risk_score = None
    api_details = {}
    domain_status = True
    ai_score_final = 0
    title = ""

    if request.method == "POST":
        input_url = normalize_url(request.form["url"].strip())
        processed = strip_scheme_www(input_url)
        domain_status = check_domain_exists(input_url)

        print("[+] Gọi API và crawl tiêu đề...")
        api_details = extract_features_from_apis(input_url, API_KEYS)
        title = api_details.get("title", "")
        print(f"[✓] Tiêu đề thu được: \"{title}\"")

        title_mapping = {processed: title}
        vectorizer_base = vectorizer.transformer_list[0][1]
        feature_union = FeatureUnion([
            ('tfidf', vectorizer_base),
            ('url_custom', URLFeatureExtractor()),
            ('title_custom', TitleFeatureExtractor(title_mapping=title_mapping))
        ])

        try:
            print("[+] Đang chạy model.predict...")
            X_vec = feature_union.transform([processed])
            dmatrix = xgb.DMatrix(X_vec)
            ai_raw = model.predict(dmatrix)[0]
            ai_score = int(ai_raw >= 0.5)
            print("[✓] Model đã chạy xong.")
        except Exception as e:
            print("[✗] Model không chạy được! Lỗi:", e)
            ai_raw = 0.0
            ai_score = 0

        risk_score, ai_score_final = calculate_risk_score(
            ai_raw, api_details, url=input_url, title=title, domain_ok=domain_status
        )

        if risk_score >= 7:
            result = "NGUY HIỂM"
        elif risk_score >= 4:
            result = "NGHI NGỜ"
        else:
            result = "AN TOÀN"

        if not domain_status:
            result += " (Tên miền không phân giải IP)"

    return render_template(
        "index.html",
        url=input_url,
        prediction=result,
        score=risk_score,
        api=api_details,
        domain_ok=domain_status,
        ai_score=ai_score_final,
        title=title,
    )

# ── Chạy app ────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
