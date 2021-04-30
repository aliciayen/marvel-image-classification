#!/usr/bin/env python3

import optparse
import sys
import yaml
import csv
import pipeline


class CSVLogger:
    def __init__(self, outfile):
        self.rows_written = 0

        if isinstance(outfile, str):
            outfile = open(outfile, "w")
        self._outfile = outfile
        self._writer = csv.writer(self._outfile)

    def log_run(self, params, results):
        params = self._flatten_params(params)
        results = self._flatten_results(results)

        if self.rows_written == 0:
            header = list(params.keys()) + list(results.keys())
            self._writer.writerow(header)

        data = list(params.values()) + list(results.values())
        self._writer.writerow(data)
        self._outfile.flush()
        self.rows_written += 1

    @staticmethod
    def _flatten_params(params):
        params = params.copy()
        del params['search_options']
        opt, optparams = params['optimizer']
        params['optimizer'] = opt
        params['lr'] = None
        params['momentum'] = None
        params.update(optparams)

        return params

    @staticmethod
    def _flatten_results(results):
        flat = {}
        for restype in results.keys():
            new = {restype + "_" + k: v for (k,v) in results[restype].items()}
            flat.update(new)
        return flat


def main():
    p = optparse.OptionParser(
        usage="Usage: %prog [OPTIONS] CFGSPEC",
        description="Reads a list of search terms from the file SEARCH_LIST, "
        "one per line, and scrapes images matching the search term from the "
        "web."
    )
    p.add_option('-n', '--n-images', dest='count', default=100,
                 help='save N_IMAGES images for each search term')
    p.add_option('-c', '--cache-dir', dest='cache_dir', default='cache',
                 help='The directory in which to store the downloaded images')
    p.add_option('-o', '--output', dest='output', default=sys.stdout,
                 help='The filename of the output CSV (stdout by default)')
    opts, args = p.parse_args()

    if len(args) < 1:
        p.print_help(file=sys.stderr)
        sys.exit(1)

    with open(args[0], "r") as f:
        cfgspec = yaml.safe_load(f)
    writer = CSVLogger(opts.output)

    pipeline.run_group(cfgspec, writer.log_run, imagecache=opts.cache_dir,
                       n_images=int(opts.count))


main()
