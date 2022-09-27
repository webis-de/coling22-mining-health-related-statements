import scrapy
import scrapy.http.response.html


class MerriamWebsterSpider(scrapy.Spider):
    name = "merriam_webster"

    start_urls = ["https://www.merriam-webster.com/medical"]

    def parse(self, response: scrapy.http.response.html.HtmlResponse):
        title = response.xpath("//h1/text()").get()
        assert title is not None
        if title == "Medical Dictionary":
            # front page
            hrefs = response.xpath(
                "//div[@class='browse-words']"
                "/div[@class='alphalinks']"
                "/ul[@class='clearfix']"
                "/li[@class='unselected']"
                "/a"
                "/@href"
            ).getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        elif title == "Browse the Medical Dictionary":
            # letter listing
            hrefs = response.xpath("//div[@class='entries']/ul/li/a/@href").getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
            next_page = response.xpath(
                "//ul[@class='pagination']/li/a[@aria-label='Next']/@href"
            ).get()
            yield response.follow(next_page, callback=self.parse)
        else:
            # entry page
            contents = response.xpath(
                "//div[starts-with(@id, 'medical-entry')]"
                "/div[@class='vg']"
                "//span[@class='dt ']"
                "//*/text()"
            ).getall()
            content = "\n".join(content for content in contents if content != ": ")
            yield {"title": title, "entry": content}
