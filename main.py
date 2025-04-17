"""
Main script for FFD generation from TESS lightcurve data

Usage:
    main.py [options]
    main.py <config> [options]

Options:
    --out_dir=<dir>                Directory for outputs [default: ./output/]
    --search_dir=<dir>             Folder for searches [default: ./sectors/]

    --star_names=<star>            Names of stars
    --sectors=<sec>                 TESS Sector to pull stars from
    --spectral_type=<type>         Spectral type to search for stars
    --teff_low=<temp>              Low Teff limit to search [default: 2000]
    --teff_high=<temp>             High Teff limit to search [default: 3000]

    --savgov_iterations=<iter>     Iterations for lc.flatten  [default: 9]
    --savgov_sigma_cutoff=<sig>    Sigma cutoff for lc.flatten  [default: 3]
    --star_max=<max>               Maximum number of stars

"""
import os
import sys
import glob
import utils as ut
from pathlib import Path
from docopt import docopt
from configparser import ConfigParser


def main():
    # Reads in arguments from config file or command line.
    # Also allows x = True commands.
    print('\n###############################')
    print('#### Loading parameters... ####')
    print('###############################\n')
    args = docopt(__doc__)
    if args['<config>'] is not None:
        config_file = Path(args['<config>'])
        config = ConfigParser()
        config.read(str(config_file))
        for n, v in config.items('parameters'):
            for k in args.keys():
                if k.split('--')[-1].lower() == n:
                    if v.lower() == 'true':
                        v = True
                    args[k] = v

    # Search data from selected sectors is saved to avoid re-downloads
    search_dir = args['--search_dir']
    if not os.path.isdir(search_dir):
        os.mkdir(search_dir)

    # Checks if output directory exists; create one if not
    if not os.path.isdir(args['--out_dir']):
        os.mkdir(args['--out_dir'])

    print('\n###############################')
    print('#### Building star list... ####')
    print('###############################\n')

    # Creates list of stars to search for from inputs
    if args['--star_names'] is not None:
        star_names = args['--star_names']
        star_names_list = list(map(str.strip, star_names.split(',')))

    # Creates a list of stars from inputted sectors
    elif args['--sectors'] is not None:
        sec_list = list(map(str.strip, args['--sectors'].split(',')))
        for sec in sec_list:
            ut.save_sector(sec, search_dir)

        star_names_list = ut.build_names_from_sectors(sec_list, search_dir)

    # This uses astroquery to search for TESS stars by temperature.
    # Currently broken - it takes too long.
    else:
        try:  # Uses either defaults or user inputed numbers
            teff_low = int(args['--teff_low'])
            teff_high = int(args['--teff_high'])
        except TypeError:
            raise TypeError('T_eff limits must be integers.')

        if args['--spectral_type'] is not None:
            spectral_type = args['--spectral_type']
            teff_low, teff_high = ut.get_spectral_temp(spectral_type)
            print('Finding stars in spectral class '+str(spectral_type))

        if args['--star_max'] is not None:
            star_max = None  #int(args['--star_max'])

        try:  # checks to see if temperature data is already downloaded
            file = open('./tic_with_temp.csv')
        except FileNotFoundError:  # if it isn't it downloads it
            os.system("wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ftptCZath0-QQoUrTSu9NMqj5KXCJwEE' -O tic_with_temp.csv")  # nopep8

        tics_list_path = "./tic_with_temp.csv"
        tics_arr = ut.tics_from_temp(
            tics_list_path, teff_low, teff_high, star_max)

        star_names_list = []
        for i in range(len(tics_arr)):
            star_names_list.append('TIC '+str(tics_arr[i]))
        print(star_names_list)

    print('\n###############################')
    print('#### Creating stellar FFDs ####')
    print('###############################\n')
    for placement, star in enumerate(star_names_list):
        print('Starting on {} ({}/{}).'.format(star,
                                               placement+1,
                                               len(star_names_list)))

        star_path = os.path.join(args['--out_dir'], star.replace(' ', '_'))

        try:  # Keeps program running if folders already exist
            os.mkdir(star_path)
        except FileExistsError:
            print('A folder for {} already exists.'.format(star))

        try:  # Keeps program running if a star name is not searchable
            ut.save_raw_lc(star,
                           star_path,
                           int(args['--savgov_iterations']),
                           float(args['--savgov_sigma_cutoff']))
        except FileNotFoundError:
            print('No search results found for {}.'.format(star))
            print('Continuing on...\n')
            os.rmdir(star_path)
            continue

        # Analyzes light curve from each sector separately
        lc_path_list = glob.glob(os.path.join(star_path, '*.csv'))
        for lc_path in lc_path_list:
            if 'FFD.csv' not in lc_path:  # NEED TO FIX; is there a better way to export the FFDs?
                ut.analyze_lc(lc_path)

        # Combines flare data from all sectors to make FFD
        flares_path_list = glob.glob(os.path.join(star_path, '*.ecsv'))

        if len(flares_path_list) == 0:
            print('No flares found overall')
        else:
            ut.generate_ffd(star, star_path, flares_path_list)

        print('Operations for {} finished.\n'.format(star))


if __name__ == '__main__':
    main()
