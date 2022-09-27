import scrapy
import scrapy.http.response.html


class RxListSpider(scrapy.Spider):
    name = "rx_list"

    start_urls = ["https://www.rxlist.com/drug-medical-dictionary/article.htm"]

    def parse(self, response: scrapy.http.response.html.HtmlResponse):
        title = response.xpath("//h1/text()").get()
        assert title is not None
        if title == "Drug-Medical-Dictionary":
            # front page
            hrefs = response.xpath(
                "//div[@id='Drugs_AZlist']" "//ul[@id='maintab']" "/li" "/a" "/@href"
            ).getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        elif title.startswith("MedTerms Medical Dictionary A-Z List -"):
            # letter listing
            hrefs = response.xpath(
                "//div[@id='AZ_container']/div[@class='AZ_results']/ul/li/a/@href"
            ).getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        else:
            # entry page
            contents = response.xpath("//div[@class='pgContent']//*/text()").getall()
            content = "\n".join(content for content in contents)
            yield {"title": title[14:], "entry": content}
