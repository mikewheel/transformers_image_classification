import scrapy


class ImagesSpider(scrapy.Spider):
    """Responsible for pulling and storing the images, and handling errors"""
    name = "openimages-v6"
    
    # TODO: use a different iterable: custom class that iterates over the CSV in small batches
    #       with class-level caching of ID, size, and MD5 for later
    start_urls = []

    def parse(self, response):
        page = response.url
        image = response  # TODO: pull image data from response body
        
        # TODO: get data on this particular image from the cache
        expected_size, actual_size, expected_md5, actual_md5 = None
        assert expected_size is not None and expected_size == actual_size, "Image sizes do not match!"
        assert expected_md5 is not None and expected_md5 == actual_md5, "Image hashes do not match!"
        
        # TODO: save to the appropriate location on disk: handoff to the PyTorch DataLoader
        # TODO: delete this image from the caches
        pass

