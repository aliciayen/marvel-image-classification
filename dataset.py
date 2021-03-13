import pandas as pd 
import argparse
import sys
from download import pathify

def main():
    parser = argparse.ArgumentParser(
        usage="[REQUIRED] Input file name  [OPTIONS] Output file name",
        description="Reads in a CSV of character names from the input file and produces a CSV "
        "containing the character name, good/bad label, and image slug.")
    parser.add_argument('input', type=str, help='Input file name of top 100 characters CSV list', nargs='?')
    parser.add_argument('-o', '--output', type=str, default='dataset.csv', help='Ouput file name for generated CSV')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    generate_files(args.input, args.output)


def generate_files(input_file, output_file):
    """
    Reads in a CSV of character names and produces a CSV containing 
    the character name, good/bad label, and image slug using the pathify function.

    Also produces a text file consisting of 
    only character names, compatible with downloader.py
    """

    characters = pd.read_csv(input_file)

    # Generate search list text file
    characters['Character'].to_csv('search_list.txt', header=None, index=None)

    # Generate CSV with image slug 
    characters['Image_Slug'] = characters['Character'].apply(lambda x: pathify(x))
    characters.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
