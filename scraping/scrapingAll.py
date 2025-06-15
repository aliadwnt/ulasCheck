# import requests
# import csv
# import os
# import time
# from datetime import datetime
# import json


# def load_cookie(cookies_json) -> str:
#     """Load cookie JSON dari file dan ubah ke string format header"""
#     try:
#         with open(cookies_json, "r") as f:
#             cookies_data = json.load(f)
#     except Exception as e:
#         print(f"[ERROR] Gagal membuka atau memproses file cookie: {e}")
#         return ""

#     cookies_string = "; ".join([f"{c['name']}={c['value']}" for c in cookies_data])
#     return cookies_string


# def shopee(url, cookies_json):
#     cookies = load_cookie(cookies_json)
#     if not cookies:
#         print("[ERROR] Cookie tidak valid atau kosong")
#         return

#     # Ambil shop_id dan user_id dari URL
#     try:
#         parts = url.split("/")
#         user_id = parts[4]
#         shop_id = parts[5].split("shop_id=")[-1]
#     except IndexError:
#         print("[ERROR] Format URL tidak valid")
#         return

#     headers = {
#         "content-type": "application/json",
#         "cookie": cookies,
#         "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#     }

#     offset = 0
#     limit = 6
#     result = []

#     while True:
#         try:
#             api_url = (
#                 f"https://shopee.co.id/api/v4/seller_operation/get_shop_ratings_new"
#                 f"?limit={limit}&offset={offset}&replied=false&shopid={shop_id}&userid={user_id}"
#             )
#             response = requests.get(api_url, headers=headers, timeout=10)
#             if response.status_code != 200:
#                 print(f"[ERROR] Gagal akses API. Status code: {response.status_code}")
#                 break

#             data_req = response.json()
#             items = data_req.get("data", {}).get("items", [])
#             if not items:
#                 break

#             for value in items:
#                 data_result = {
#                     "nama pengguna": value.get("author_username", ""),
#                     "produk": value.get("product_items", [{}])[0].get("name", ""),
#                     "review": value.get("comment", ""),
#                     "rating": value.get("rating_star", 0),
#                     "waktu transaksi": datetime.fromtimestamp(value["ctime"]).strftime(
#                         "%Y-%m-%d %H:%M"
#                     ),
#                 }
#                 result.append(data_result)
#                 print(f"✅ Mengambil review dari: {data_result['nama pengguna']}")

#             offset += limit
#             time.sleep(2)  # delay agar tidak diblokir Shopee
#         except requests.exceptions.Timeout:
#             print("[ERROR] Timeout saat request ke Shopee.")
#             break
#         except Exception as e:
#             print(f"[ERROR] Gagal mengambil data: {e}")
#             break

#     if result:
#         output_dir = "scraping-result"
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, "shopee_rating.csv")

#         keys = result[0].keys()
#         with open(output_path, "w", newline="", encoding="utf-8") as output_file:
#             writer = csv.DictWriter(output_file, fieldnames=keys)
#             writer.writeheader()
#             writer.writerows(result)

#         print(f"✅ Data berhasil disimpan di {output_path}")
#     else:
#         print("⚠️ Tidak ada data review yang berhasil diambil.")


# if __name__ == '__main__':
#     url_shop = "https://shopee.co.id/buyer/555974029/rating?shop_id=555954448"
#     cookies_json = r"D:\shopee\scraping\cookies.json"
#     shopee(url_shop, cookies_json)
import requests
import csv
import os
import time
from datetime import datetime
import json

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
        parts = url.split("/")
        user_id = parts[4]
        shop_id = parts[5].split("shop_id=")[-1]
    except IndexError:
        print("[ERROR] Format URL tidak valid")
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
            # items = data_req.get("data", {}).get("items", [])
            data_data = data_req.get("data")
            if not data_data:
                print("[INFO] Data kosong. Mungkin sudah tidak ada review lagi.")
                break

            items = data_data.get("items", [])
            if not items:
                break

            for value in items:
                if len(result) >= max_reviews:
                    break  # Stop jika sudah 30

                data_result = {
                    "Username": value.get("author_username", ""),
                    "Produk": value.get("product_items", [{}])[0].get("name", ""),
                    "Review": value.get("comment", ""),
                    "Rating": value.get("rating_star", 0),
                    "ReviewAt": datetime.fromtimestamp(value["ctime"]).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
                result.append(data_result)
                print(f"✅ Mengambil review dari: {data_result['Username']}")

            offset += limit
            time.sleep(0.1)
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

if __name__ == '__main__':
    url_shop = "https://shopee.co.id/buyer/481541891/rating?shop_id=481522314"
    cookies_json = r"D:\shopee\scraping\cookies.json"
    shopee(url_shop, cookies_json)

# import time
# import csv
# from datetime import datetime
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options


# def start_driver():
#     options = Options()
#     options.add_argument("--start-maximized")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--disable-notifications")

#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     return driver


# def scroll_to_load_all_reviews(driver):
#     last_height = driver.execute_script("return document.body.scrollHeight")
#     while True:
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(2)
#         new_height = driver.execute_script("return document.body.scrollHeight")
#         if new_height == last_height:
#             break
#         last_height = new_height


# def extract_reviews(driver):
#     reviews = []
#     review_elements = driver.find_elements(By.CLASS_NAME, "shopee-product-rating")  # class review container

#     for review in review_elements:
#         try:
#             username = review.find_element(By.CLASS_NAME, "shopee-product-rating__author-name").text
#             comment = review.find_element(By.CLASS_NAME, "shopee-product-rating__content").text
#             rating_stars = len(review.find_elements(By.CLASS_NAME, "shopee-rating-stars__lit"))
#             timestamp = review.find_element(By.CLASS_NAME, "shopee-product-rating__time").text
#             product = review.find_element(By.CLASS_NAME, "shopee-product-rating__product").text

#             reviews.append({
#                 "nama pengguna": username,
#                 "produk": product,
#                 "review": comment,
#                 "rating": rating_stars,
#                 "waktu transaksi": timestamp
#             })
#         except Exception as e:
#             print(f"[SKIP] Gagal ambil data 1 review: {e}")
#             continue

#     return reviews


# def save_to_csv(data, filename="shopee_all_reviews.csv"):
#     keys = data[0].keys()
#     with open(filename, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(data)
#     print(f"✅ {len(data)} review berhasil disimpan ke {filename}")


# def scrape_all_reviews_from_shop(shop_url):
#     driver = start_driver()
#     print("🌐 Membuka halaman toko...")
#     driver.get(shop_url)
#     time.sleep(5)

#     print("📜 Scrolling untuk memuat semua ulasan...")
#     scroll_to_load_all_reviews(driver)

#     print("🔍 Mengambil semua review...")
#     reviews = extract_reviews(driver)

#     if reviews:
#         save_to_csv(reviews)
#     else:
#         print("⚠️ Tidak ada review ditemukan.")

#     driver.quit()


# if __name__ == "__main__":
#     shop_url = "https://shopee.co.id/buyer/555974029/rating?shop_id=555954448"
#     scrape_all_reviews_from_shop(shop_url)


# import time
# import csv
# from datetime import datetime
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC


# def start_driver():
#     options = Options()
#     options.add_argument("--start-maximized")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--disable-notifications")
#     options.add_argument("--disable-infobars")
#     options.add_argument("--disable-extensions")

#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     return driver


# def scroll_to_load_all_reviews(driver):
#     scroll_pause_time = 2  # Time between scrolls
#     last_height = driver.execute_script("return document.body.scrollHeight")

#     while True:
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(scroll_pause_time)
#         new_height = driver.execute_script("return document.body.scrollHeight")
#         if new_height == last_height:
#             break
#         last_height = new_height


# def extract_reviews(driver):
#     reviews = []

#     try:
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, "shopee-product-rating"))
#         )
#     except Exception as e:
#         print(f"⚠️ Tidak bisa menemukan elemen ulasan: {e}")
#         return reviews

#     review_elements = driver.find_elements(By.CLASS_NAME, "shopee-product-rating")

#     for review in review_elements:
#         try:
#             username = review.find_element(By.CLASS_NAME, "shopee-product-rating__author-name").text
#             comment = review.find_element(By.CLASS_NAME, "shopee-product-rating__content").text
#             rating_stars = len(review.find_elements(By.CLASS_NAME, "shopee-rating-stars__lit"))
#             timestamp = review.find_element(By.CLASS_NAME, "shopee-product-rating__time").text
#             product_elem = review.find_elements(By.CLASS_NAME, "shopee-product-rating__product")
#             product = product_elem[0].text if product_elem else "Tidak ada"

#             reviews.append({
#                 "nama_pengguna": username,
#                 "produk": product,
#                 "review": comment,
#                 "rating": rating_stars,
#                 "waktu_transaksi": timestamp
#             })
#         except Exception as e:
#             print(f"[SKIP] Gagal ambil data 1 review: {e}")
#             continue

#     return reviews


# def save_to_csv(data, filename="shopee_all_reviews.csv"):
#     if not data:
#         print("❌ Tidak ada data yang disimpan.")
#         return

#     keys = data[0].keys()
#     with open(filename, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(data)
#     print(f"✅ {len(data)} review berhasil disimpan ke {filename}")


# def scrape_all_reviews_from_shop(shop_url):
#     driver = start_driver()
#     print("🌐 Membuka halaman toko...")
#     driver.get(shop_url)

#     try:
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, "shopee-product-rating"))
#         )
#     except Exception as e:
#         print(f"❌ Halaman ulasan gagal dimuat: {e}")
#         driver.quit()
#         return

#     print("📜 Scrolling untuk memuat semua ulasan...")
#     scroll_to_load_all_reviews(driver)

#     print("🔍 Mengambil semua review...")
#     reviews = extract_reviews(driver)

#     save_to_csv(reviews)
#     driver.quit()


# if __name__ == "__main__":
#     # Ganti dengan link rating toko
#     shop_url = "https://shopee.co.id/buyer/555974029/rating?shop_id=555954448"
#     scrape_all_reviews_from_shop(shop_url)
