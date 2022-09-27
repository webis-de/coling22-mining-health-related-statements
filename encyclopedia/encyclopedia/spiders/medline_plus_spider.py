import scrapy
import scrapy.http.response.html


class MedlinePlusSpider(scrapy.Spider):
    name = "medline_plus"

    start_urls = ["https://medlineplus.gov/encyclopedia.html"]

    def parse(self, response: scrapy.http.response.html.HtmlResponse):
        title = response.xpath("//h1/text()").get()
        assert title is not None
        if title == "Medical Encyclopedia":
            # front page
            hrefs = response.xpath(
                "//nav[@role='navigation']"
                "//ul[@class='alpha-links']"
                "/li"
                "/a"
                "/@href"
            ).getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        elif title.startswith("Medical Encyclopedia: "):
            # letter listing
            hrefs = response.xpath("//article//ul[@id='index']/li/a/@href").getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        else:
            # entry page
            contents = response.xpath(
                "//article//div[@id='ency_summary']//*/text()"
            ).getall()
            contents += response.xpath(
                "//article"
                "//div[@class='section-header']"
                "//h2[text()!='References' and not(contains(text(), 'Review Date'))]"
                "/../../.."
                "//div[@class='section-body']"
                "//*"
                "/text()"
            ).getall()
            content = "\n".join(content for content in contents)
            yield {"title": title, "entry": content}
