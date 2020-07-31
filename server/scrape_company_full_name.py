import requests
from bs4 import BeautifulSoup

res = requests.get("https://www.google.com/search?client=ubuntu&hs=Tml&channel=fs&sxsrf=ALeKk03TmHT7a89Fy4rFvCa7ycqpCl5-_w%3A1596173641726&ei=Sa0jX7X1K8G6mAXPrqdo&q=A+ticker&oq=A+ticker&gs_lcp=CgZwc3ktYWIQAzIHCCMQJxCdAjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjoECAAQR1CfE1ifE2DlGGgAcAF4AIAB8wOIAfMDkgEDNC0xmAEAoAEBqgEHZ3dzLXdpesABAQ&sclient=psy-ab&ved=0ahUKEwj1vaiX4vbqAhVBHaYKHU_XCQ0Q4dUDCAs&uact=5", headers={"User-Agent":"Mozilla/5.0"})
soup = BeautifulSoup(res.text, 'html.parser')

