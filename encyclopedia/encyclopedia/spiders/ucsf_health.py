import scrapy
import scrapy.http.response.html


class UCSFHealthSpider(scrapy.Spider):
    name = "ucsf_health"

    start_urls = [
        "https://www.ucsfhealth.org/conditions",
        "https://www.ucsfhealth.org/treatments",
        "https://www.ucsfhealth.org/education",
        "https://www.ucsfhealth.org/medical-tests",
    ]

    def parse(self, response: scrapy.http.response.html.HtmlResponse):
        title = response.xpath("//h1/text()").get()
        assert title is not None
        if response.xpath("//div[@class='letters-and-filter']").get() is not None:
            # letter listing
            hrefs = response.xpath(
                "//div[@class='master-finder-rollup-links component']"
                "//ul[@class='rollup-list']"
                "/li"
                "/p"
                "/a"
                "/@href"
            ).getall()
            for next_page in hrefs:
                next_page = response.urljoin(next_page)
                yield response.follow(next_page, callback=self.parse)
        else:
            # entry page
            # education and treatment header + condition
            contents = response.xpath(
                "//div[@class='component-content']"
                "//div[starts-with(@class, 'font-larger')]"
                "//*/text()"
            ).getall()
            # education and treatment content
            contents += response.xpath(
                "//div[@class='component-content']"
                "//div[starts-with(@class, 'component rich-text paragraph1')]"
                "//div[@class='component-content']"
                "//*/text()"
            ).getall()
            # tests
            contents += response.xpath(
                "//div[@class='component content container']"
                "/div[@class='component-content']"
                "/div[@class='field-content']"
                "//*/text()"
            ).getall()
            content = "\n".join(content for content in contents if content != ": ")
            if contents:
                yield {"title": title, "entry": content}
