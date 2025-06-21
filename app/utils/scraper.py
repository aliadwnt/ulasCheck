import requests, csv, os, time, json, io
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from app import db, socketio
from app.models.reviewModel import Review

def load_cookie(cookies_json) -> str:
    try:
        with open(cookies_json, "r") as f:
            cookies_data = json.load(f)
        return "; ".join([f"{c['name']}={c['value']}" for c in cookies_data])
    except Exception as e:
        print(f"[ERROR] Gagal membuka file cookie: {e}")
        return ""

def shopee(url, cookies_json):
    cookies = load_cookie(cookies_json)
    if not cookies:
        return None, "Cookie tidak valid atau kosong"

    # Ambil shop_id dan user_id dari URL Shopee
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        shop_id = query.get("shop_id", [None])[0]
        path_parts = parsed.path.strip("/").split("/")
        user_id = path_parts[1] if len(path_parts) > 1 else None

        if not shop_id or not user_id:
            return None, "URL tidak valid, gagal ambil shop_id atau user_id"
    except Exception as e:
        return None, f"Gagal memproses URL: {e}"

    headers = {
        "content-type": "application/json",
        "cookie": cookies,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    offset = 0
    limit = 6
    max_reviews = 100
    result = []

    while len(result) < max_reviews:
        try:
            api_url = (
                f"https://shopee.co.id/api/v4/seller_operation/get_shop_ratings_new"
                f"?limit={limit}&offset={offset}&replied=false&shopid={shop_id}&userid={user_id}"
            )
            response = requests.get(api_url, headers=headers, timeout=(5, 5))
            if response.status_code != 200:
                return None, f"Gagal akses API Shopee: {response.status_code}"

            data = response.json().get("data")
            items = data.get("items", []) if data else []
            if not items:
                break

            for value in items:
                if len(result) >= max_reviews:
                    break

                result.append({
                    "Username": value.get("author_username", ""),
                    "Produk": value.get("product_items", [{}])[0].get("name", ""),
                    "Review": value.get("comment", ""),
                    "Rating": value.get("rating_star", 0),
                    "ReviewAt": datetime.fromtimestamp(value["ctime"]).strftime("%Y-%m-%d %H:%M"),
                })

                # Emit progres ke client
                socketio.emit("progress", {
                    "current": len(result),
                    "total": max_reviews
                })

            offset += limit
            time.sleep(1)
        except Exception as e:
            return None, f"Gagal mengambil data: {e}"

    if not result:
        return None, "Tidak ada data review yang berhasil diambil"

    # Simpan data ke dalam memori (CSV dalam bentuk BLOB)
    keys = result[0].keys()
    output_stream = io.StringIO()
    writer = csv.DictWriter(output_stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(result)
    file_data = output_stream.getvalue().encode("utf-8")

    # Simpan ke database
    review = Review(shop_id=shop_id, file="rating.csv", file_data=file_data)
    db.session.add(review)
    db.session.commit()

    return review.id, f"{len(result)} review berhasil disimpan ke database"
