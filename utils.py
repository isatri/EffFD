import os
import urllib.request
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from astroquery.vizier import Vizier
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd

### These will eventually be options for the input.
### Here for now until I have time to redo.
FLUX_TYPE = 'pdcsap_flux'  # pdcsap_flux or sap flux, PDC is processed, SAP is 'raw' data.
                           # The one that is better will depend on your project.
EXPOSURE_TIME = 120        # 120 seconds (2 min) or 20 seconds
WINDOW_ITERATIONS = 5
CUT_TOP_LIMIT_FACTOR = 1.8
CUT_BOT_LIMIT_FACTOR = -0.8

OPTION_INITIAL_CUTOFF_SIGMA = 3
OPTION_END_FLARE_SIGMA = 1.0
### Thank you for your patience!

# Change the window for smoothening depending on if you are using 20-sec or 2-min data
if EXPOSURE_TIME==20:
    WINDOW_SIZE = 151
    FINAL_LENGTH_CONDITION = 12  # how many data points a flare needs to be after expansion to be selected
else:
    WINDOW_SIZE = 31
    FINAL_LENGTH_CONDITION = 4

def type_error_catch(var, vartype, inner_vartype=None, err_msg=None):
    if not isinstance(var, vartype):
        if err_msg is None:
            err_msg = '{} is not a {}'.format(var, vartype.__name__)
        raise TypeError(err_msg)

    elif vartype is list:
        if not var:
            raise ValueError('No passed lists should be empty.')

        for inner in var:
            if not isinstance(inner, inner_vartype):
                if err_msg is None:
                    err_msg = '{} is not a {}'.format(inner,
                                                      inner_vartype.__name__)
                raise TypeError(err_msg)


def get_spectral_temp(classification):
    """Returns upper and lower temperature limits in
    Kelvin of required spectral class

    Parameters
    ----------
    classification : str
        Stellar spectral type to get temperatures for

    Returns
    -------
    temperature_limits : tuple of two int
        Upper and lower temperature of given spectral class

    """
    type_error_catch(classification, str)

    if classification in 'L,T,Y'.split(','):
        raise ValueError('Brown dwarfs are not yet supported.')
    elif classification not in 'O,B,A,F,G,K,M'.split(','):
        raise ValueError('Please use spectral type in OBAFGKM.')
    else:
        # low temp limit, high temp limit, in K
        temp_dict = {'M': (2000, 3500),
                     'K': (3500, 5000),
                     'G': (5000, 6000),
                     'F': (6000, 7500),
                     'A': (7500, 10000),
                     'B': (10000, 30000),
                     'O': (30000, 60000)}
        return temp_dict[classification]


def save_sector(sector, search_path):
    """Function that retrieves data for each sector from the TESS website

    Parameters
    ----------
    sector : str
        TESS sector to search for

    search_path : str
        Path to search directory where previous queries are saved

    Returns
    -------
    None
        Saves the sector data as a CSV in the search folder

    """
    type_error_catch(sector, str)
    type_error_catch(search_path, str)

    save_string = search_path + 'sector{}.csv'.format(sector)
    if not os.path.isfile(save_string):
        try:
            print('Downloading Sector {} observation list.'.format(sector))
            url = "https://tess.mit.edu/wp-content/uploads/all_targets_S{}_v1.csv".format(sector.zfill(3))  # nopep8
            urllib.request.urlretrieve(url, save_string)
        except urllib.error.HTTPError:
            raise ValueError('Inputted URL could not be found.')


def get_sector_tics(sector_list, search_path):
    """For given TESS Sectors, gets the unique TESS identifiers (TIC) of all
    observed objects

    Parameters
    ----------
    sector_list : list of str
        List of TESS Sectors

    search_path : str
        Path to search directory where previous queries are saved

    Returns
    -------
    names_list : list of str
        TESS identifiers (TIC) for unique objects in given Sectors

    """
    type_error_catch(sector_list, list, str)
    type_error_catch(search_path, str)

    tic_array = np.array([])

    for sector in sector_list:
        curr_csv = np.genfromtxt(search_path + 'sector{}.csv'.format(sector),
                                 delimiter=',',
                                 skip_header=6)
        tic_array = np.append(tic_array, curr_csv[:, 0])

    tic_array = np.unique(tic_array)
    tic_list = tic_array.astype(int).astype(str).tolist()
    return tic_list


def build_names_from_sectors(sector_list, search_path):
    """Returns TESS names (e.g. 'TIC 1234') of objects observed in the
    listed Sectors

    Parameters
    ----------
    sector_list : list of str
        List TESS sectors to search through

    search_path: str
        Path to search directory where previous queries are saved

    Returns
    -------
    tess_names_list : list of str
        TESS names for all observed objects in given Sectors

    """
    type_error_catch(sector_list, list, str)
    type_error_catch(search_path, str)

    tics = get_sector_tics(sector_list, search_path)
    tess_names_list = ['TIC ' + name for name in tics]
    return tess_names_list


def tics_from_temp(path, teff_low, teff_high, star_max):
    """Returns list of TESS names (e.g. 'TIC 1234') of objects observed
    within user defined temperature range or spectral class

    Parameters
    ----------
    path : str
        path to csv file of TIC and temperatures

    teff_low: int
        Lower limit on temperature

    teff_high: int
        Upper limit on temperature

    star_max: int
        Maximum number of stars to do FFD on

    Returns
    -------
    tess_names_list : list of str
        TESS names for all (or some depending on star_max)
        observed objects in given temp range

    """
    type_error_catch(path, str)
    type_error_catch(teff_low, int)
    type_error_catch(teff_high, int)

    star_table = pd.read_csv(
        path, names=['TIC', 'Teff', 'Teff_err', 'RA', 'Dec'])
    star_table = star_table.sort_values(by='Teff').reset_index()
    Teff_low_ind = star_table['Teff'].sub(teff_low).abs().idxmin()
    Teff_high_ind = star_table['Teff'].sub(teff_high).abs().idxmin()
    TIC_list_full = []
    for i in range(Teff_low_ind, Teff_high_ind+1):
        TIC_list_full.append(int(star_table['TIC'][i]))
    if star_max is not None:
        TIC_list = []
        rand_arr = np.random.choice(
            range(0, len(TIC_list_full)), size=star_max)
        for i in range(len(rand_arr)):
            TIC_list.append(TIC_list_full[rand_arr[i]])
        return TIC_list
    else:
        return TIC_list_full


def add_indices(arr, num=3):
    spacing = num*2 + 1  # indices to move to have room for expansion
    add_range = np.arange(-num, num+1)  # indices range to add to list
    new_arr = np.zeros(len(arr) * spacing, dtype=int)

    counter = num
    for value in arr:
        for change in add_range:
            new_arr[counter+change] = value + change
        counter += spacing

    positive_condition = (new_arr >= 0)
    return np.unique(new_arr[positive_condition])  # return unique positive idx


def remove_small_time_blocks(dates_day,
                             separation_limit_sec=600,
                             save_limit_sec=18000,
                             date_conversion_to_sec=86400.0):
    dates = dates_day * date_conversion_to_sec
    separations = (dates[1:] - dates[:-1])
    separation_ends = np.where(separations > separation_limit_sec)[0]
    separation_ends = np.append(separation_ends, len(dates)-1)  # attach last point

    keep_sections = []
    start = 0
    for end in separation_ends:
        separation = (dates[end] - dates[start])
        if separation > save_limit_sec:
            keep_sections.append((start, end))
        start = end+1

    return keep_sections


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def remove_light_curve_nans(time, flux):
    cond = (flux > 0.0)
    return np.array(time[cond]), np.array(flux[cond])


def rms(y):
    return np.sqrt(np.mean(y**2))


def save_raw_lc(obj, save_path, filter_iter, filter_sig):
    """Uses lightkurve to retrieve the lightcurve from TESS
    object and saves the light curve image plus the raw data

    Parameters
    ----------
    obj : str
        TESS object (mostly stars)

    save_path : str
        Path to the save directory

    filter_iter: int
        Number of iterations that lightkurve smooths data over

    filter_sig : float
        Statistical sigma at which lightkurve cuts off data

    """
    type_error_catch(obj, str)
    type_error_catch(save_path, str)
    type_error_catch(filter_iter, int)
    type_error_catch(filter_sig, float)

    # SPOC == TESS data pipeline
    # Getting only the 120 second exposure light curves for consistency
    # NOTE: this does not necessarily work for binaries anymore?
    # UV Cet is now split between different TICs which messes things up.
    # Need to test on different binaries.
    search_result = lk.search_lightcurve(obj, author='SPOC', exptime=EXPOSURE_TIME)
    
    if not search_result:
        raise FileNotFoundError('No results for {}.'.format(obj))

    for result in search_result:
        sector = result[0].mission[0][-2:]

        # save files named after star+sector, in the star's output directory
        save_string = '{}/{}_{}'.format(save_path,
                                        obj.replace(' ', '_'),
                                        sector)

        if os.path.isfile(save_string+'.csv'):
            print('Sector {} CSV exists for {}'.format(sector, obj))
            continue
        
        result.table["dataURL"] = result.table["dataURI"]  # workaround MAST issue
        try:
            lc = result.download(quality_bitmask='default')
        except KeyError:
            print('Sector {} for {} gave a KeyError.'.format(sector, obj))
            continue

        # Save light curve CSV file
        lc.to_csv(save_string + '.csv', overwrite=True)

        lc = lc.flatten(niters=filter_iter, sigma=filter_sig)
        # The code calls this name if used, so not saving the flattened one
        # lc.to_csv(save_string + '_flattened.csv', overwrite=True)

        # Saves light curve PNG plot
        # Closes figures to avoid memory problems when running for many stars
        fig, ax = plt.subplots(1,1)
        lc.plot(ax=ax)
        fig.savefig(save_string + '.png')
        fig.clf()
        plt.close(fig)


def group_by_missing(seq):
    """Takes input array and groups
    consecutive numbers
    Parameters
    ----------
    seq : list
        Data to be grouped

    Returns
    -------
    grouped : list
        List of list where each list
        is a group of consequative
        numbers
    """
    if not seq:
        return seq
    grouped = [[seq[0]]]
    for x in seq[1:]:
        if x == grouped[-1][-1] + 1:
            grouped[-1].append(x)
        else:
            grouped.append([x])
    return grouped


def plot_lc_diagnostics(time_all, flux_all, time_final, flux_final):
    fig, ax = plt.subplots()
    ax.plot(time_all, flux_all, time_final, flux_final)
    fig.savefig('/Users/itristan/Desktop/diag.png')
    plt.close(fig)


def self_flatten_lc(csv_path):
    lc = ascii.read(csv_path, guess=False, format='csv')
    lc_time, lc_flux = remove_light_curve_nans(lc['time'], lc[FLUX_TYPE])

    remsep = remove_small_time_blocks(lc_time)
    keep_times = np.array([], dtype=int)
    for b, e in remsep:
        cond = np.arange(b, e+1, dtype=int)
        keep_times = np.append(keep_times, cond)

    lc_time_reduced = lc_time[keep_times]
    lc_flux_reduced = lc_flux[keep_times]

    export_lc_time = np.array([])
    export_lc_flux = np.array([])

    fig, ax = plt.subplots(4,1, figsize=(7,7), sharex=True)
    ax[0].plot(lc_time, lc_flux)
    ax[0].plot(lc_time_reduced, lc_flux_reduced)
    ax[1].plot(lc_time_reduced, lc_flux_reduced)

    bigsec = remove_small_time_blocks(lc_time_reduced, 18001, 18002)
    for b, e in bigsec:
        beginning_time = lc_time_reduced[b]
        ending_time = lc_time_reduced[e]
        cond = (lc_time >= beginning_time) & (lc_time <= ending_time)

        final_time = np.copy(lc_time[cond])
        final_flux = np.copy(lc_flux[cond])

        window = WINDOW_SIZE
        window_iter = WINDOW_ITERATIONS
        for i in np.arange(window_iter):
            data_smooth = savgol_filter(final_flux, window, 3)
            new_curve = (final_flux - data_smooth)[:-window]
            flux_rms = rms(new_curve)
            report_std = np.std((final_flux / data_smooth)[:-window])

            removeable_idxs = np.where(new_curve > CUT_TOP_LIMIT_FACTOR*flux_rms)[0]
            removeable_idxs = np.append(removeable_idxs, np.where(new_curve < CUT_BOT_LIMIT_FACTOR*flux_rms)[0])
            #removeable_idxs = add_indices(removeable_idxs, 1)

            final_time = np.delete(final_time, removeable_idxs)
            final_flux = np.delete(final_flux, removeable_idxs)

        spl = interp1d(x=final_time,
                       y=final_flux,
                       kind='linear',
                       fill_value='extrapolate')
        interpflux = spl(lc_time[cond])

        ax[1].plot(final_time, final_flux, color='red')
        ax[2].plot(final_time, final_flux, color='red')
        ax[2].plot(lc_time[cond], interpflux, color='gray')

        flat_time = lc_time[cond]

        # Mean normalized value should be around 1
        # If not, normalize by the mean
        flatten_factor = np.mean((lc_flux[cond]/interpflux)) # match the bottom of the cutoff
        
        if flatten_factor > 0.9999:
            flatten_factor = 1

        flat_flux = (lc_flux[cond]/interpflux) - flatten_factor

        export_lc_time = np.append(export_lc_time, flat_time)
        export_lc_flux = np.append(export_lc_flux, flat_flux)

    ax[0].set(ylabel='Starting LC\n[electrons/s]')
    ax[1].set(ylabel='Flux\n[electrons/s]')
    ax[2].set(ylabel='Quiescent Line\n[electrons/s]')
    ax[3].plot(export_lc_time, export_lc_flux, '-o', ms=2, lw=1, alpha=0.5)
    ax[3].set(xlabel='Time [BTJD Days]',
              ylabel='Flattened LC')
    fig.tight_layout()
    fig.savefig(csv_path.replace('.csv', '_lc_diagnostics.png'))
    plt.close(fig)

    def calc_quiet_std(flux, iterations=2, sigma=2.0):
        std = np.max(np.abs(flux))
        for iteration in range(iterations):
            std_cond = (flux <= sigma*std) & (flux >= -sigma*std)
            std = np.sqrt(np.mean(flux[std_cond]**2))
        return std

    report_std = calc_quiet_std(export_lc_flux)
    # report_std = np.sqrt(np.mean(export_lc_flux**2))
    # std_cond = (export_lc_flux <= 2*report_std) & (export_lc_flux >= -2*report_std) 
    # report_std = np.sqrt(np.mean(export_lc_flux[std_cond]**2))
    # print('Testing std calculator: {} (manu) vs. {} (func)'.format(report_std,
    #                                                                calc_quiet_std(export_lc_flux)))

    return export_lc_time, export_lc_flux, report_std


def plot_flares_and_lc(T, F, flare_list, reported_std=0.0, pathname=''):

    plot_title = pathname.split('/')[-1].replace('_', ' ').replace('.csv','')
    plot_title = plot_title[:-3] + ': Sector' + plot_title[-3:]

    lc_fig, lc_ax = plt.subplots()
    lc_ax.plot(T, F, 'o', ms=2, lw=1, alpha=0.5)
    lc_ax.axhline(reported_std, color='black', ls='--', alpha=0.8)

    fig_rows = 5
    fig_cols = -(-len(flare_list) // fig_rows)
    ffig, fax = plt.subplots(fig_cols, fig_rows, figsize=(fig_rows*2,fig_cols*2))
    fax = fax.ravel()

    for count, flare_idxs in enumerate(flare_list):
        lc_ax.plot(T[flare_idxs], F[flare_idxs], color='red')
        time_flare_graph = (T[flare_idxs] - T[flare_idxs[0]]) * 1440.0
        fax[count].axhline(0, color='gray', ls='-', alpha=1.0)
        fax[count].axhline(reported_std, color='gray', ls='--', alpha=1.0)
        fax[count].axhline(reported_std/3.0, color='gray', ls='--', alpha=1.0)
        fax[count].plot(time_flare_graph, F[flare_idxs],
                        color='red', ls='-', marker='+', ms=5)

    if len(flare_list) < fig_cols*fig_rows:
        for leftover in range(count+1, fig_cols*fig_rows):
            fax[leftover].set_axis_off()

    ffig.text(0.5, 0.98, plot_title,
              va='center', ha='center', fontsize=12)
    ffig.text(0.5, 0.02, 'Flaring Time [min]',
              va='center', ha='center', fontsize=12)
    ffig.text(0.02, 0.5, 'Normalized Flux',
                va='center', ha='center', rotation='vertical', fontsize=12)

    ffig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    ffig.savefig(pathname.replace('.csv','_all_flares.png'))
    plt.close(ffig)

    lc_ax.set(xlabel='Time BJD [day]',
              ylabel='Normalized Flux',
              title=plot_title)
    lc_fig.tight_layout()
    lc_fig.savefig(pathname.replace('.csv','_marked_lc.png'))
    plt.close(lc_fig)


def expand_flare_indices(flux, flare, sigma):
    for place in [0, -1]:
        val = flux[flare[place]]
        while val > sigma:
            new_idx = flare[place] - 1 - (2*place)
            if place==0:
                flare.insert(place, new_idx)
            elif place==-1:
                flare.append(new_idx)

            try:
                val = flux[new_idx]
            except IndexError:
                print('Flare ran out of times to expand.')
                val = 0


def analyze_lc(csv_path):
    """Takes the light curve data and finds flares
    using a 3 sigma method. If flares are found
    in the data the start time, end time, duration
    max flux, and time of the max flux are added
    to a table and saved.
    Parameters
    ----------
    csv_path : str
        Path to csv file for the sector data

    Returns
    -------
    _flares.ecsv file containing flare data
    """

    type_error_catch(csv_path, str)

    time, flux, lcrms = self_flatten_lc(csv_path)

    # Flare finding method
    criteria = OPTION_INITIAL_CUTOFF_SIGMA * lcrms
    criteria_index = np.where(flux > criteria)[0]
    grouped_criteria = group_by_missing(criteria_index.tolist())

    flare_index = []
    for group in grouped_criteria:
        if len(group) >= 3:
            flare_index.append(group)
        ### COMMENT THIS PART TO ONLY GET THE CLEAREST FLARES ###
        elif len(group) >= 2 and len(flux) not in group: 
            expand_flare_indices(flux, group, lcrms*OPTION_END_FLARE_SIGMA)
            if len(group) >= FINAL_LENGTH_CONDITION:
                flare_index.append(group)
        #########################################################

    if len(flare_index) == 0:
        print('No flares found in this sector data')

    else:

        flare_table = Table(names=['start_time', 'end_time', 'duration',
                                   'max_flux', 'max_flux_time', 'fluence'])

        for flare in flare_index:
            if len(flux) not in flare:
                expand_flare_indices(flux, flare, lcrms*OPTION_END_FLARE_SIGMA)
            else:
                flare.pop(-1)
        flare_index = np.unique(np.array(flare_index, dtype=object))

        if type(flare_index[0]) is int:  # Catch when there is only one flare
            flare_index = [flare_index.tolist()]

        plot_flares_and_lc(time, flux, flare_index, criteria, csv_path)

        for counts, flare in enumerate(flare_index):

            flare_flux = flux[flare]
            flare_time = time[flare] * 86400.0 * u.second

            # Flare properties of interest
            t_start = flare_time[0]
            t_end = flare_time[-1]
            duration = t_end - t_start
            flux_max = np.max(flare_flux)
            t_flux_max = flare_time[(flare_flux == flux_max)]
            fluence = np.trapz(y=flare_flux, x=flare_time)

            # This was added because of a problemw ith CR Dra
            # Where one of the estimated light curves wwnt down suddenly
            # at the start. This needs to be sorted out, bc a fake flare was
            # created there.
            if duration > 0.0:
                flare_table.add_row([t_start, t_end, duration,
                                     flux_max, t_flux_max, fluence])

        # Save total light curve monitoring time for FFD statistics
        flare_table['total_lc_time'] = len(time) * EXPOSURE_TIME * u.second
        save_path = csv_path.replace('.csv', '_flares.ecsv')
        flare_table.write(save_path, overwrite=True)

        print(str(len(flare_table['fluence'])) +
              ' flares were found in this sector data!')


def get_middle_ffd_regime(x, y, min_slope=-5.0, max_slope=-1.0):
    """Finds the location of the middle regime of the flare frequency diagram
    (FFD) using the min and max slope of where most middle regimes lie

    Parameters
    ----------
    x : numpy array
        Energy array

    y : numpy array
        Cumulative frequency array

    min_slope : float
        minimum value of the slope for the condition

    max_slope : float
        maximum value of the slope for the condition

    Returns
    -------
    new_x : numpy array
        Energy array within middle regime
    new_y : numpy array
        Cumulative frequency array within middle regime

    """
    type_error_catch(x, np.ndarray)
    type_error_catch(y, np.ndarray)
    type_error_catch(min_slope, float)
    type_error_catch(max_slope, float)

    dx = np.diff(x, 1)
    dy = np.diff(y, 1)
    yfirst = dy / dx
    # finds points that satisfy the slope condition for the middle regime
    cond = np.where((yfirst > min_slope) & (yfirst < max_slope))[0]
    # in the first regime some points will satisfy the above condition
    # this loop is checking that the next two points also satisfy the
    # condition so that we only get data points in the middle regime and
    # not rogue points from first regime

    try:
        for count, idx in enumerate(cond):  # note: cond is an array of idxs
            if idx-cond[count+1] <= 2 and cond[count+3]-cond[count+2] <= 2:
                starting_idx = idx
                break
        ending_idx = cond[-1]
    except IndexError:
        starting_idx = cond[0]
        ending_idx = cond[-1]

    new_x = x[starting_idx:ending_idx]
    new_y = y[starting_idx:ending_idx]
    return new_x, new_y


def func_powerlaw(x, a, b):
    return a + b*x


def calculate_slope_powerlaw(x, y):
    """Find the slope of powerlaw

    Parameters
    ----------
    x : numpy array
        x array

    y : numpy array
        y array

    Returns
    -------
    a : float
        intercept
    b : float
        slope
    b_err : float
        error on slope

    """
    type_error_catch(x, np.ndarray)
    type_error_catch(y, np.ndarray)

    # get the fit to func_powerlaw() using dataset (x, y)
    solution = curve_fit(func_powerlaw, x, y, maxfev=2000)

    a = solution[0][0]  # intercept
    b = solution[0][1]  # slope
    b_err = np.abs(b / np.sqrt(len(x)))  # slope_err

    return a, b, b_err


def get_time_and_energy(paths):
    """Takes a list of data directory paths and finds the
    total time the object was observed and the energy of
    each flare
    Note: Some TESS sectors overlap therefore the same object
    might be in multiple sectors

    Parameters
    ----------
    paths: list of str
        Path to object data.
        If the object was observed in multiple TESS sectors, there will be
        multiple paths, otherwise it will be a single filepath.

    Returns
    -------
    time : float
        Total time TESS observed the object, in units of days

    flare_eng : numpy array
        Flare energies, sorted by total size

    e_unit : astropy unit
        Unit used for flare energies, to be used in plotting

    """
    type_error_catch(paths, list, str)

    time = 0.0 * u.day
    flare_eng = np.array([])
    flare_duration = np.array([])

    for file_path in paths:
        try:
            tbl = ascii.read(file_path, guess=False, format='ecsv')

            time += tbl['total_lc_time'][0] * (1.0 * tbl['total_lc_time'].unit).to(u.day)

            flare_eng = np.append(flare_eng, tbl['fluence'].value)

            flare_duration = np.append(flare_duration, tbl['duration'].value)

        except FileNotFoundError:
            print('Flare filepath ' + file_path + ' not found.')
            continue

    flare_eng.sort()
    print(time)
    return time.value, flare_eng, tbl['fluence'].unit, flare_duration


def get_log_freq(flare_eng, tot_time):
    """Takes the flare energy array and the time it was observed
    over and returns the cumulative frequency for each energy

    Parameters
    ----------
    flare_eng : numpy array
       array of flare energy

    tot_time : float
        total time TESS observed the object

    Returns
    -------
    energy : numpy array
        log10 flare energies

    frequency : numpy array
        log10 of the cumulative frequency

    """
    type_error_catch(flare_eng, np.ndarray)
    type_error_catch(tot_time, float)

    energy = np.log10(flare_eng)

    cumulative_count = np.arange(len(energy)) + 1
    flare_frequency = cumulative_count[::-1] / tot_time

    frequency = np.log10(flare_frequency)

    return energy, frequency


def generate_ffd(obj, save_path, list_of_paths):
    """This function generates and saves the flare freqeucny diagram (FFD)

    Parameters
    ----------
    obj : str
       Name of object (mainly stars)

    save_path: str
        Path to save directory

    list_of_paths: list of str
        Path to object data.
        If the object was observed in multiple TESS sectors, there will be
        multiple paths, otherwise it will be a single filepath.

    """
    type_error_catch(obj, str)
    type_error_catch(save_path, str)
    type_error_catch(list_of_paths, list, str)

    monitoring_time, flare_energy, e_unit, duration = get_time_and_energy(list_of_paths)
    duration = duration/60.0  # minutes

    log_energy, log_frequency = get_log_freq(flare_energy, monitoring_time)

    # Linear regression to get slope
    try:
        m_ene, m_fre = get_middle_ffd_regime(log_energy, log_frequency)
        intercept, slope, slope_err = calculate_slope_powerlaw(m_ene, m_fre)
    except Exception:
        m_ene = []

    # alpha is used in some papers, but we don't need it for now
    # alpha = np.abs(slope - 1)

    savetbl = Table([log_frequency, log_energy],
                    names=['log10_frequency', 'log10_ED_sec'])
    savetbl.write('{}/{}_FFD.csv'.format(save_path, obj.replace(' ', '_')),
                  overwrite=True)

    fig = plt.figure(figsize=(7.5, 3.5))

    ax = fig.add_axes([0.05, 0.05, 0.5, 1])
    ax2 = fig.add_axes([0.7, 0.05, 0.5, 1])
    cax = fig.add_axes([1.2, 0.05, 0.04, 1])
    col_map = plt.cm.get_cmap('magma')

    if len(flare_energy) >= 3:
        divnorm = mpl.colors.TwoSlopeNorm(
            vmin=duration.min(), vcenter=np.mean(duration),
            vmax=duration.max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=divnorm)
        sm.set_array([])
        colour = sm.to_rgba(duration)
        cbar = plt.colorbar(sm, cax=cax, ticks=[int(duration.min())+1,
                                                int(np.mean(duration)),
                                                int(duration.max())])
        cbar.set_label('Duration [minutes]', labelpad=-40)
        cax.yaxis.set_ticks_position('right')
    else:
        colour = 'blue'

    # Saves FFD figure

    ax2.scatter(log_energy,
                10**log_frequency,
                marker='o',
                color=colour, edgecolor='black')
    if len(m_ene) != 0:
        ax2.plot(m_ene,
                 10**func_powerlaw(m_ene, intercept, slope),
                 color='black',
                 label=r'Slope: $%.2f\pm%.2f$' % (slope, slope_err))
        ax2.legend()
    ax2.set(xlabel=r'Log$_{10}$ $ED_{TESS}$ [s]',
            ylabel=r'Cumulative Number of Flares $>E_{TESS}$ Per Day',
            title='EFFD for {}'.format(obj),
            yscale='log')

    # Creates histogram
    N, bins, patches = ax.hist(np.log10(flare_energy), edgecolor='black', linewidth=1)
    for i in range(len(N)):
        patches[i].set_facecolor(col_map(N[i]/N.max()))

    ax.set(ylabel='Frequency', xlabel=r'Log$_{10}$ $ED_{TESS}$ [s]',
           title='Histogram for {}'.format(obj))

    plt.savefig('{}/{}_FFD.png'.format(save_path,
                obj.replace(' ', '_')), bbox_inches='tight')
    plt.close(fig)
