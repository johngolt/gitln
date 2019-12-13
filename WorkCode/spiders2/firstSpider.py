class UrlManager:
    def __init__(self):
        self.new_urls = set()
        self.old_urls = set()

    def  has_new_url(self):
        return self.new_url_size() != 0

    def  get_new_url(self):
        new_url = self.new_urls.pop()
        self.old_urls.add(new_url)
        return new_url

    def add_new_url(self, url):
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)

    def add_new_urls(self, urls):
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def new_url_size(self):
        return len(self.new_urls)

    def old_url_size(self):
        return len(self.old_urls)


import requests

class HtmlDownloader:

    def download(self, url):
        if url is None:
            return
        user_agent = 'Mozilla/5.0'
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers = headers)
        if response.status_code == 200:
            response.encoding = response.apparent_encoding
            return response.text
        return


import re
import  urllib.parse as urlparse
from bs4 import BeautifulSoup

class HtmlParser:

    def parser(self, page_url, html_cont):
        if page_url is None or html_cont is None:
            return
        soup = BeautifulSoup(html_cont, 'lxml')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data

    def _get_new_urls(self, page_url, soup):
        new_urls = set()
        links = soup.find_all('a', href = re.compile(r'/view/\d+\.htm'))
        for link in links:
            new_url = link['href']
            new_full_url = urlparse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):
        data={}
        data['url']=page_url
        title = soup.find('dd', class_='lemmaWgt-1emmaTitle-title').find('h1')
        data['title']=title.get_text()
        summary = soup.find('div', class_= 'lemma-summary')
        data['summary'] = summary.get_text()
        return data

import codecs
class DataOutput:

    def __init__(self):
        self.datas=[]

    def store_data(self, data):
        if data is None:
            return
        self.datas.append(data)

    def output_html(self):
        fout = codecs.open('baike.html', 'w', encoding='utf8')
        fout.write('<html>')
        fout.write('<head><meta charset="utf8"/></head>')
        fout.write('<body>')
        fout.write('<table>')
        for data in self.datas:
            fout.write('<tr>')
            fout.write('<td>%s</td>'%data['url'])
            fout.write('<td>%s</td>'%data['title'])
            fout.write('<td>%s</td>'%data['summary'])
            fout.write('</tr>')
            self.datas.remove(data)
        fout.write('</table>')
        fout.write('</body>')
        fout.write('</html>')
        fout.close()

