import os
import re
import sys
import optparse


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
        for search_term in f:
            destpattern = opts.output_dir + '/' + _pathify(search_term)
            download_images(opts.engine, search_term, destpattern,
                            int(opts.count))

def download_images(engine, search_term, destpattern, count):
    raise NotImplementedError()

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
