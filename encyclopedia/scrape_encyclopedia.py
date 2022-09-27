from scrapy.cmdline import execute

# health_am
# merriam_webster
# medline_plus
# ucsf_health
# rx_list


def main():
    execute(["scrapy", "crawl", "rx_list"])


if __name__ == "__main__":
    main()
