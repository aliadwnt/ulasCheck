from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Ganti path kalau tidak di-PATH-kan
service = Service(executable_path="chromedriver.exe")

options = Options()
options.add_argument("--headless")  # tanpa buka jendela Chrome
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(service=service, options=options)
driver.get("https://www.google.com")
print("Judul Halaman:", driver.title)
driver.quit()
