import scrapy
import scrapy.http.response.html


class HealthAMSpider(scrapy.Spider):
    name = "health_am"

    start_urls = ["http://www.health.am/encyclopedia"]

    def parse(self, response: scrapy.http.response.html.HtmlResponse):
        title = response.css("h1.entry-title::text").get()
        assert isinstance(title, str)
        if "Medical Encyclopedia" not in title:
            # entry page
            content = ""
            contents = response.xpath(
                "//div[@class='entry-content']//br/text()|"
                "//div[@class='entry-content']//li/text()|"
                "//div[@class='entry-content']//p/text()"
            ).getall()
            for sub_content in contents:
                content += sub_content + "\n"
            yield {"title": title, "entry": content}
        else:
            if "-" in title:
                # letter listing
                hrefs = response.xpath(
                    "//div[@class='primary-col']"
                    "//li[@class='red ency-list']"
                    "/a"
                    "/@href"
                ).getall()
                for next_page in hrefs:
                    next_page = response.urljoin(next_page)
                    yield response.follow(next_page, callback=self.parse)
            else:
                # front page
                hrefs = response.xpath(
                    "//div[@class='primary-col']"
                    "/article[@class='post']"
                    "/div[@class='entry-content']"
                    "//a[@class='headline']"
                    "/@href"
                ).getall()
                for next_page in hrefs:
                    if "encyclopedia" not in next_page:
                        continue
                    next_page = response.urljoin(next_page)
                    yield response.follow(next_page, callback=self.parse)
