import os
import re
import sys
import optparse
import urllib.parse
import requests
import bs4


def main():
    p = optparse.OptionParser(
        usage="Usage: %prog [OPTIONS] SEARCH_LIST",
        description="Reads a list of search terms from the file SEARCH_LIST, "
        "one per line, and scrapes images matching the search term from the "
        "web."
    )
    p.add_option('-n', '--n-images', dest='count', default=10,
                 help='save N_IMAGES images for each search term')
    p.add_option('-e', '--search-engine', dest='engine', default='google',
                 help='Use ENGINE to find images (currently only google)')
    p.add_option('-o', '--output-dir', dest='output_dir', default='images',
                 help='The directory in which to store the downloaded images')
    opts, args = p.parse_args()

    if len(args) < 1:
        p.print_help(file=sys.stderr)
        sys.exit(1)

    with open(args[0]) as f:
        if not os.path.exists(opts.output_dir):
            os.mkdir(opts.output_dir)

        for search_term in f:
            destpattern = opts.output_dir + '/' + _pathify(search_term)
            download_images(opts.engine, search_term, destpattern,
                            int(opts.count))

def download_images(engine, search_term, destpattern, count):
    ''' download.download_images(engine, term, destpattern, count)

    Performs a search using the search engine 'engine' (currently only
    'google' is supported), and downloads the resulting images to the
    output directory, with filenames corresponding to 'destpattern'.
    The number of downloaded images is limited to 'count', or the number
    of results on the first page, whichever is smaller.
    '''

    if engine != 'google':
        raise NotImplementedError()

    url = _generate_google_url(search_term)
    response = requests.get(url)
    s = bs4.BeautifulSoup(response.content.decode('utf-8', 'ignore'), 'lxml')

    image_elems = s.find_all(name='img')
    image_urls = [x.get('src') for x in image_elems]

    # first image is not a search result; drop it
    image_urls.pop(0)

    # download images
    limit = min(len(image_urls), count)
    for i, img in enumerate(image_urls[:limit]):
        resp = requests.get(img)
        out_fname = destpattern + ".%03i.jpg" % i
        with open(out_fname, 'wb') as f:
            f.write(resp.content)

def _generate_google_url(search_term):
    ''' _generate_google_url(search_term, ...) -> url

    Generates a google image search URL for the search term provided.
    This can be expanded later to allow for additional fitlering for
    specific image types.
    '''

    params = {
        'q': search_term,
        'tbm': 'isch',
    }
    return 'https://www.google.com/search?' + urllib.parse.urlencode(params)

def _pathify(string):
    ''' _pathify(string) -> string

    Generates a "sanitized" string from the input given (i.e., one
    containing only alphanumeric characters, digits, underscores, and
    dashes) that cab be safely used as a filename pattern for output
    images. Strips trailing whitespace and converts invalid characters
    to underscores.
    '''

    return re.sub(r'[^0-9a-zA-Z_-]', '_', string.strip())


if __name__ == '__main__':
    main()
