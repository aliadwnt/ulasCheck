# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import json
# import time

# # Setup WebDriver (misalnya menggunakan Chrome)
# chromedriver_path = r"D:\App\chromedriver-win64\chromedriver-win64\chromedriver.exe"  # Ganti dengan path ke chromedriver Anda

# # Konfigurasi Chrome Options (harus dilakukan sebelum inisialisasi driver)
# options = Options()
# options.add_argument("--disable-dev-shm-usage")
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-gpu")
# options.add_argument("--start-maximized")
# options.add_argument("--disable-blink-features=AutomationControlled")

# # Inisialisasi WebDriver setelah options dibuat
# driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)

# # Buka halaman Shopee untuk membuat sesi aktif
# driver.get("https://shopee.co.id")  # Ganti dengan URL Shopee yang sesuai

# # Tunggu beberapa detik agar halaman dapat dimuat dengan baik
# time.sleep(3)

# # Impor cookie dari file JSON
# # Impor cookie dari file JSON (path lengkap dan benar)
# with open(r"D:\shopee\scraping\cookie.core.json", "r") as file:
#     cookies = json.load(file)

# # Menambahkan cookie ke driver
# for cookie in cookies:
#     # Pastikan cookie memiliki atribut 'name', 'value', dan domain sebelum ditambahkan
#     if 'name' in cookie and 'value' in cookie:
#         driver.add_cookie(cookie)

# # Setelah menambahkan cookie, buka ulang situs agar WebDriver menganggap sesi sudah aktif
# driver.get("https://shopee.co.id/home")  # Ganti dengan URL tujuan setelah login di Shopee

# # Tunggu hingga elemen di halaman utama Shopee muncul untuk memastikan halaman sudah dimuat dengan baik
# WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.CLASS_NAME, "shopee-header__logo"))
# )

# # Menampilkan URL saat ini untuk memastikan bahwa halaman sudah ter-load
# print("Halaman sekarang:", driver.current_url)

# # Anda bisa menambahkan langkah-langkah selanjutnya di sini, misalnya mengambil data atau berinteraksi dengan halaman

# # Menutup browser setelah selesai
# driver.quit()
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time

chromedriver_path = r"D:\App\chromedriver-win64\chromedriver-win64\chromedriver.exe"

options = Options()
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)

driver.get("https://shopee.co.id")
time.sleep(3)

with open(r"D:\shopee\scraping\cookie.core.json", "r") as file:
    cookies = json.load(file)

for cookie in cookies:
    if 'name' in cookie and 'value' in cookie:
        if 'sameSite' in cookie and cookie['sameSite'] not in ['Strict', 'Lax', 'None']:
            del cookie['sameSite']
        driver.add_cookie(cookie)

driver.get("https://shopee.co.id/home")

WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "shopee-header__logo"))
)

print("Halaman sekarang:", driver.current_url)
driver.quit()
