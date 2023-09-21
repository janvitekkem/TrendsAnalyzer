zip -r scraper.zip __main__.py src/

ibmcloud fn action update trendsProcessor scraper.zip --docker ibnjunaid/trends-processor