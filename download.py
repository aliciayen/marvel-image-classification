import os
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
            download_images(opts.engine, search_term, int(opts.count))

def download_images(engine, search_term, count):
    raise NotImplementedError()


if __name__ == '__main__':
    main()
