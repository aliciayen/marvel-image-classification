import os
import re
import sys
import optparse
import urllib.parse
import requests
import bs4

# Supported image types for advance Google image search.
IMAGE_STYLES = ('clipart', 'lineart')


def main():
    p = optparse.OptionParser(
        usage="Usage: %prog [OPTIONS] SEARCH_LIST",
        description="Reads a list of search terms from the file SEARCH_LIST, "
        "one per line, and scrapes images matching the search term from the "
        "web."
    )
    p.add_option('-n', '--n-images', dest='count', default=10,
                 help='save N_IMAGES images for each search term')
    p.add_option('-o', '--output-dir', dest='output_dir', default='images',
                 help='The directory in which to store the downloaded images')
    p.add_option('-s', '--style', dest='style', default=None,
                 help='Google image search image type (clipart|lineart)')
    p.add_option('-d', '--domain', dest='domain', default=None,
                 help="Restrict search results to DOMAIN")
    opts, args = p.parse_args()

    if len(args) < 1:
        p.print_help(file=sys.stderr)
        sys.exit(1)

    if (opts.style is not None) and (opts.style not in IMAGE_STYLES):
        print("Unsupported image style: '%s'" % opts.style, file=sys.stderr)
        sys.exit(1)

    with open(args[0]) as f:
        if not os.path.exists(opts.output_dir):
            os.mkdir(opts.output_dir)

        for search_term in f:
            destpattern = opts.output_dir + '/' + _pathify(search_term)
            url = generate_search_url(search_term, style=opts.style,
                                      domain=opts.domain)
            download_images(url, destpattern, int(opts.count))

def download_images(url, destpattern, count):
    ''' download.download_images(url, destpattern, count)

    Performs a Google using the URL provided in 'url', and downloads the
    resulting images to the output directory, with filenames
    corresponding to 'destpattern'. The number of downloaded images is
    limited to 'count', or the number of results on the first page,
    whichever is smaller.
    '''

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
        filetype = _check_magic(resp.content)
        out_fname = destpattern + ".%03i.%s" % (i, filetype)
        with open(out_fname, 'wb') as f:
            f.write(resp.content)

def generate_search_url(search_term, style=None, domain=None):
    ''' download.generate_search_url(search_term, ...) -> url

    Generates a google image search URL for the search term provided.
    This can be expanded later to allow for additional fitlering for
    specific image types.
    '''

    if style is not None:
        if style not in IMAGE_STYLES:
            raise ValueError("Unsupported image type")
        tbs = "itp:%s" % style
    else:
        tbs = ""

    if domain is not None:
        search_term += " site:%s" % domain

    params = {
        'q': search_term,
        'tbm': 'isch',
        'tbs': tbs,
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

def _check_magic(img):
    ''' _check_magic(image_data) -> filetype

    Checks the magic number at the start of the given image data to
    determine the type. Returns the string extension corresponding to
    the file type, or 'img' if the image format is unknown.
    '''

    magic_numbers = {
        b"\xFF\xD8\xFF\xDB": 'jpeg',
        b"\xFF\xD8\xFF\xEE": 'jpeg',
        b"\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01": 'jpeg',
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": 'png',
    }

    for magic, filetype in magic_numbers.items():
        if img[0:len(magic)] == magic:
            return filetype
    else:
        return 'img'


if __name__ == '__main__':
    main()
