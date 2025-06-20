import requests, os, csv, time, json
from datetime import datetime

def load_cookie(cookie_path):
    try:
        with open(cookie_path, "r") as f:
            cookies = json.load(f)
        return "; ".join([f"{c['name']}={c['value']}" for c in cookies])
    except Exception as e:
        print(f"[ERROR] Tidak bisa memuat cookie: {e}")
        return ""

def scrape_reviews(url, cookie_path):
    cookies = load_cookie(cookie_path)
    if not cookies:
        return None, "Gagal membaca cookie"

    # Ambil shop_id dan user_id dari URL
    try:
        parts = url.strip().split("/")
        user_id = parts[4]  # contoh: buyer/12345678/rating
        shop_part = parts[5]  # harus mengandung "shop_id="
        if "shop_id=" not in shop_part:
            return None, "shop_id tidak ditemukan dalam URL"
        shop_id = shop_part.split("shop_id=")[-1]
    except IndexError:
        return None, "Format URL tidak valid"

    headers = {
        "cookie": cookies,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "content-type": "application/json"
    }

    offset = 0
    limit = 6
    result = []

    while len(result) < 30:
        api_url = (
            f"https://shopee.co.id/api/v4/seller_operation/get_shop_ratings_new"
            f"?limit={limit}&offset={offset}&replied=false&shopid={shop_id}&userid={user_id}"
        )
        try:
            res = requests.get(api_url, headers=headers, timeout=5)
            if res.status_code != 200:
                return None, f"Status code {res.status_code} saat akses API Shopee"

            items = res.json().get("data", {}).get("items", [])
            if not items:
                break  # Tidak ada lagi data

            for item in items:
                review = {
                    "Username": item.get("author_username", ""),
                    "Produk": item.get("product_items", [{}])[0].get("name", ""),
                    "Review": item.get("comment", ""),
                    "Rating": item.get("rating_star", 0),
                    "ReviewAt": datetime.fromtimestamp(item["ctime"]).strftime("%Y-%m-%d %H:%M")
                }
                result.append(review)

            offset += limit
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            return None, f"Request gagal: {e}"
        except Exception as e:
            return None, f"Gagal parsing data: {e}"

    if not result:
        return None, "Tidak ada data review yang berhasil diambil"

    # Simpan hasil scraping ke CSV
    os.makedirs("scraping-result", exist_ok=True)
    csv_path = "scraping-result/rating-new.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
        return csv_path, f"✅ Berhasil mengambil {len(result)} ulasan"
    except Exception as e:
        return None, f"[ERROR] Gagal menyimpan CSV: {e}"
