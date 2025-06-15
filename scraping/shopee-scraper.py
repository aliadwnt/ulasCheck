import requests
import csv
import os
import time
from datetime import datetime
import json
from urllib.parse import urlparse, parse_qs

def load_cookie(cookies_json) -> str:
    try:
        with open(cookies_json, "r") as f:
            cookies_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membuka atau memproses file cookie: {e}")
        return ""

    cookies_string = "; ".join([f"{c['name']}={c['value']}" for c in cookies_data])
    return cookies_string

def shopee(url, cookies_json):
    cookies = load_cookie(cookies_json)
    if not cookies:
        print("[ERROR] Cookie tidak valid atau kosong")
        return

    # Ambil shop_id dan user_id dari URL
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        shop_id = query.get("shop_id", [None])[0]

        # user_id biasanya ada di path ke-3 setelah /buyer/
        path_parts = parsed.path.strip("/").split("/")
        user_id = path_parts[1] if len(path_parts) > 1 else None

        if not shop_id or not user_id:
            print("[ERROR] Tidak bisa ekstrak shop_id atau user_id dari URL.")
            return
    except Exception as e:
        print(f"[ERROR] Gagal memproses URL: {e}")
        return

    headers = {
        "content-type": "application/json",
        "cookie": cookies,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    offset = 0
    limit = 6
    max_reviews = 10000
    result = []

    while len(result) < max_reviews:
        try:
            api_url = (
                f"https://shopee.co.id/api/v4/seller_operation/get_shop_ratings_new"
                f"?limit={limit}&offset={offset}&replied=false&shopid={shop_id}&userid={user_id}"
            )
            response = requests.get(api_url, headers=headers, timeout=(5, 5))
            if response.status_code != 200:
                print(f"[ERROR] Gagal akses API. Status code: {response.status_code}")
                break

            data_req = response.json()
            data_data = data_req.get("data")
            if not data_data:
                print("[INFO] Data kosong. Mungkin sudah tidak ada review lagi.")
                break

            items = data_data.get("items", [])
            if not items:
                break

            for value in items:
                if len(result) >= max_reviews:
                    break

                data_result = {
                    "Username": value.get("author_username", ""),
                    "Produk": value.get("product_items", [{}])[0].get("name", ""),
                    "Review": value.get("comment", ""),
                    "Rating": value.get("rating_star", 0),
                    "ReviewAt": datetime.fromtimestamp(value["ctime"]).strftime("%Y-%m-%d %H:%M"),
                }
                result.append(data_result)
                print(f"✅ Mengambil review dari: {data_result['Username']}")

            offset += limit
            time.sleep(2)
        except requests.exceptions.Timeout:
            print("[ERROR] Timeout saat request ke Shopee.")
            break
        except Exception as e:
            print(f"[ERROR] Gagal mengambil data: {e}")
            break

    if result:
        output_dir = "scraping-result"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rating-new.csv")

        keys = result[0].keys()
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result)

        print(f"✅ {len(result)} data berhasil disimpan di {output_path}")
    else:
        print("⚠️ Tidak ada data review yang berhasil diambil.")

# Contoh pemanggilan langsung
if __name__ == '__main__':
    url_shop = input("Masukkan link toko Shopee: ")
    cookies_json = r"D:\shopee\scraping\cookies.json"
    shopee(url_shop, cookies_json)
