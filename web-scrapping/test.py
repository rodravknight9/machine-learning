import requests
from bs4 import BeautifulSoup

url = "https://www.freecodecamp.org/news/"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

articles = soup.find_all("a")

for a in articles:
    ##print(a.text, a["href"])
    print(a.attrs)