import os
import time
import numpy as np
import glob
import pandas as pd
from natsort import natsorted
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from textwrap import wrap

plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 21, # Changes font size. 19 for normal plots. 21 for large figures
        "figure.figsize": "12.8, 9.6",  # figure size in inches. This is twice as large as normal (6.4, 4.8)
        "lines.markersize": 12, # marker size, in points. 6 standard. 9 for QCM
        "figure.constrained_layout.use": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "xtick.bottom": True,
        "xtick.minor.visible": True,
        "ytick.left": True,
        "ytick.right": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 14, # major tick size in points (3.5 default)
        "xtick.minor.size": 8, # minor tick size in points (2 default)
        "ytick.major.size": 14, # major tick size in points (3.5 default)
        "ytick.minor.size": 8 # minor tick size in points (2 default)
    })


"""
Operando Processing and AnaLysis for XRD data (OPAL-XRD)


--------
GitHub 13.12.2024
--------

Contributors:
    Amalie Skurtveit (amalie.skurtveit@kjemi.uio.no): conceptualization, functions, readability, arrays
    Erlend T. North (e.t.north@smn.uio.no): readability, improved plotting functions, improved functions
    Casper Skautvedt (casper.skautvedt@smn.uio.no): conceptualization, GC, plotting

"""
#___________________________________________________________________________________________________________________________________________________________________________________________#

"""
#   Section 1: USER DEFINED PARAMETERS #
______________________________________________
________________PARAMETERS____________________
    Bat_ID is the identificator of your battery
    XRD_FOLDER is the folder where your data is located
    XRD_FILETYPE is the file extension of your XRD data (e.g. "xye" or "dat" from synchrotron, "xy" for home-lab data)
    WAVELENGTH is the wavelength of the incident beam
    CONVERT_TO_Q is eithr True or False based on whether or not you would like to display the data in Q or 2th, respectively
    DIFFERENTIAL_XRD is either True or False based on whether or not you would like the data displayed as differential XRDs. 1st scan subtracted from each scan.
    NORMALIZE is either True or False based on whether or not you would like to normalize each scan to the highest peak (i.e. max value) in that scan
    ELCHEM_TYPE (new), either GC or CV technique
    TIME_ON_Y (new), 
    
_______________________________________________
"""

##### XRD SPECIFIC USER DEFINED PARAMETERS #####
BAT_ID         = "BATTERY ID"       ### Change ###
XRD_FOLDER      = r"YOUR FOLDER"    ### Change ###
XRD_FILETYPE    = "xy"              ### Change ###  # supported filetypes are "xye" (synchrotron), "dat" (BM01), and "xy" (home-lab)
WAVELENGTH      = 0.7093            ### Change ###  # value: home lab: 0.7093 Å, BM31 Jun23: 0.24468, BM31 Nov22: 0.31916, BM01 Nov22: 0.60456
CONVERT_TO_Q    = True              ### Change ###  # do you want to convert 2th into Q?
DIFFERENTIAL_XRD = False            ### Change ###  # do you want to display the data as differential XRDs? 1st scan subtracted from each scan.
NORMALIZE        = False            ### Change ###  # do you want to normalize hte intensity of the collected data? If you have large fluctuations, it makes sense.
ELCHEM_TYPE      = "GC"             ### Change ###  # "CV", or "GC", default is GC
TIME_ON_Y        = True             ### Change ###  #do you want time on the y-axis for both op and ew plot?


##### GC SPECIFIC USER DEFINED PARAMETERS #####
GC_FILE         = r"GC FOLDER\GC-ILE.mpt"         ### Change ### # mandatory to include the extention, e.g. ".mpt" or ".txt"
ION             = "Na"              							### Change ### # e.g., "Li", "Na", "K"
MIN_POTENTIAL   = 1.3															### Change ###
MAX_POTENTIAL   = 3.8															### Change ###


### Defining colors for the countor plot. More colormaps available through: https://matplotlib.org/stable/users/explain/colors/colormaps.html
colormap = 'twilight_shifted'


##### PLOT SPECIFIC USER DEFINED PARAMETERS #####   
MIN_INTENSITY   = 0          ### Change ###  # lower limit intensity in operando contour figure (related to intensity in XRD, but not necessarily the lowest observed intensity)
MAX_INTENSITY   = 10         ### Change ###  # higher limit intensity in operando contour figure (related to intensity in XRD, but not necessarily the highest observed intensity)
CBARNUM         = 1000       ### Change ###  # number of distinct colors, aka detail. Change numebr as a fucntion of levels between MIN_I and MAX_I (play with it)
COUNT           = 1          ### Change ###  # number of runs, change if you would like to not overwrite the last plot generated


### x-axis specific limits, do not need to change for home-lab data ###
if CONVERT_TO_Q:
    LEFT_OP_X_AXIS = 1.15
    RIGHT_OP_X_AXIS = 3
else:
    LEFT_OP_X_AXIS = 7.25
    RIGHT_OP_X_AXIS = 22


### CODE STARTS HERE ###
DEST_FOLDER = XRD_FOLDER.replace(f'{XRD_FOLDER}', f'{XRD_FOLDER}\\Figures')

if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

def GC_datatype(gc_file):
    """
    Small function to check where the GC file is from

    Returns: 'Biologic', 'BatSmall', or 'Undefined'
    """

    if gc_file.endswith('.mpt'):
        GC_DATATYPE = 'Biologic'
    elif gc_file.endswith('.txt'):
        GC_DATATYPE = 'BatSmall'
    else:
        GC_DATATYPE = 'Undefined'
    return GC_DATATYPE    


def format_GC(gc_file):
    """
    Function to extract potential and time from the GC file and saving only these extracted values to a file
    You don't need to know where the file is from because the function will do a check (valid: Biologic, BatSmall, all else formats will raise an exception)
    """
    
    time = []
    voltage = []
    capacity = []
    
    GC_DATATYPE = GC_datatype(gc_file)
    print(f"The GC data is from {GC_DATATYPE}")

    if GC_DATATYPE == 'Biologic':
        with open(gc_file, "r") as f:
            data = f.readlines()

            headerlines = 0
            for line in data:    
                if "Nb header lines" in line:
                    headerlines = int(line.split(':')[-1])
                    break
            lines = data[headerlines:]
            for line in lines:
                values = line.split("\t")
                time.append(float(values[7]))
                voltage.append(float(values[11]))
                capacity.append(float(values[23]))
        
        return [t/3600 for t in time], voltage
    
    elif GC_DATATYPE == 'BatSmall':
        with open(gc_file, "r") as f:
            data = f.readlines()
            for line in data:
                values = line.split('\t')
                time.append(float(values[0]))
                voltage.append(float(values[1]))
        #GC_array = np.column_stack((time, voltage))
        return time, voltage
    
    elif GC_DATATYPE == 'Undefined':
        raise Exception(f"GC file dataformat is {GC_DATATYPE} and not supported by this script. Please check if you have made a mistake")



def format_CV(gc_file):
    """
    Function to exctact time and current from the CV file
    """

    current = []
    time = []
    voltage = []

    with open(gc_file, "r") as f:
        data = f.readlines()

        headerlines = 0
        for line in data:    
            if "Nb header lines" in line:
                headerlines = int(line.split(':')[-1])
                break
        lines = data[headerlines:]
        for line in lines:
            column = line.split("\t")
            time.append(float(column[6]))
            current.append(float(column[9]))
            voltage.append(float(column[8]))

    
    return [t/3600 for t in time], current, voltage


def BATX_GC(gc_file):
    FILTER = [
    '[s]',
    '[V]'
    ]
    
    
    df = pd.read_csv(gc_file, sep=";", skiprows=3)
    df = df.filter(FILTER)


    time_in_seconds = np.array(df['[s]'].tolist())
    time_in_hours = [t/3600 for t in time_in_seconds]
    
    return df, time_in_hours   


if __name__ == "__main__":
    # Changing directory to where your XRD files are located
    os.chdir(XRD_FOLDER)

    # Importing XRD files by natsorted (i.e. in human-brain way of counting) by filtering files that ends with 'xy' 
    XRD_FILES = natsorted(glob.glob(f'*{XRD_FILETYPE}'))[:51]

    # Finding number of files
    n_files = len(XRD_FILES)
    print("Number of input files: ", n_files)

    
    # Loading all XRD files into a pandas DataFrame (i.e. Excel for Python)
    df = pd.DataFrame()

    column_names = []
    angle = []
    start_time = time.perf_counter()
    for i, file in enumerate(XRD_FILES):
        #print(file)
        column_names.append(i)
        data = pd.read_csv(file, sep=" ")
        data.index = data.iloc[:, 0]
        
        intensity = data.iloc[:, 1]
        df = pd.concat([df, intensity], axis=1)
        df.columns = column_names
 

    if CONVERT_TO_Q:
        angle = df.index.tolist()
        Q_values = []
        for i, an in enumerate(angle):
            Q_conversion = abs(((4 * np.pi) / WAVELENGTH) * np.sin(an / 2 * (np.pi / 180)))
            Q_values.append(Q_conversion)
        
        df.set_index(pd.Series(Q_values), inplace=True)

        print("Conversion to Q-space is now done")
        op_xlabel = "Q ($\\mathdefault{Å^{-1}}$)"
    else:
        op_xlabel = "2θ (°)"
    
    if NORMALIZE:
        print("Normalization")
        df = df/df.max() 
    
    if DIFFERENTIAL_XRD:
        df_differential = pd.DataFrame()
        num_columns = df.shape[1]

        for columns in range(0, num_columns):
       
            diff_col = df.iloc[:, columns] - df.iloc[:, 52] 
            df_differential[columns] = diff_col
    


   

    ### formatting electrochemistry ###
    if ELCHEM_TYPE == "GC":
        GC_TIME, GC_VOLTAGE = format_GC(GC_FILE)
        max_time = GC_TIME[-1]
        GC_x_axis = f'Potential vs {ION}/'f'{ION}' + '$\\mathdefault{^+}$(V)'
        max_time = 21.7
        
    elif ELCHEM_TYPE == "CV":
        CVtime, current, voltage = format_CV(GC_FILE)
        max_time = CVtime[-1]
        
        #max_time = 11.1
        
        # In order to format the CV data correctly and without making another script, this was my solution:
        GC_TIME = CVtime    # Time in CV
        GC_VOLTAGE = current    # Current 
        GC_x_axis = "Current (mA)"
        MAX_POTENTIAL = 1 # limits on the axis is actually current!
        MIN_POTENTIAL = -1.2

    TIME_OF_INTEREST = [0, max_time]

    if TIME_ON_Y:
        SCAN_TIME = [max_time/(n_files-1)]
        print("scan time is: ", SCAN_TIME)
        op_ylabel = "Time (h)"
    else:
        SCAN_TIME = None
        op_ylabel="Scan number"
     
    
    """
    Plotting specific code from here on out
    """
    ### actual plotting ###
        
    locator = ticker.MaxNLocator(nbins=1000) # not sure if this does anything
    

    # Formatting the DataFrame, readability 
    transposed_df = df.transpose()
    angles = transposed_df.columns
    intensities = transposed_df.values
    scans = transposed_df.index

    # Making the plot based on if you would like to have time on y-axis or not
    if TIME_ON_Y:
        fig, (op1, ew) = plt.subplots(1, 2, sharey=True, width_ratios= [8, 1])
        scans = SCAN_TIME[0]*np.array(range(0, len(intensities)))
    else:
        fig, (op1, ew) = plt.subplots(1, 2, width_ratios= [8, 1]) 

    for _, spine in op1.spines.items():
        spine.set_visible(True)

    
    ### Filling the plot with the dataframe
    op1.contourf(angles, scans, intensities, levels=np.linspace(MIN_INTENSITY, MAX_INTENSITY,CBARNUM), cmap=colormap, extend='max')
    op1.set(xlabel=op_xlabel, ylabel=op_ylabel) 
    op1.set_xlim(left= LEFT_OP_X_AXIS, right=RIGHT_OP_X_AXIS)

    if TIME_ON_Y:
        op1.set_ylim(bottom=TIME_OF_INTEREST[0], top=TIME_OF_INTEREST[1])
    else:
        op1.set_ylim(bottom=scans[0], top=scans[-1])

    op1.xaxis.set_major_locator(ticker.MaxNLocator(5))
    op1.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # Making the colorbar and formatting it. You can change the location of the colorbar by changing location= to either "bottom", "top", "right", "left".
    cbar1 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin= MIN_INTENSITY, vmax= MAX_INTENSITY), cmap= colormap), ax=op1, location="bottom")
    cbar1.set_label('Intensity (a.u.)'); cbar1.set_ticks([MIN_INTENSITY,MAX_INTENSITY]); cbar1.set_ticklabels(['Low','High'])
    

    GC_y_axis = 'Time (h)'

    ew.plot(GC_VOLTAGE, GC_TIME, color='k', linewidth=2)
    ew.set_xlabel(GC_x_axis, loc="center", wrap=True)
    ew.set(ylabel = GC_y_axis) 
    ew.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ew.set_xlim([MIN_POTENTIAL, MAX_POTENTIAL])
    ew.set_ylim([TIME_OF_INTEREST[0], TIME_OF_INTEREST[1]])
    ew.yaxis.set_label_position('right') 
    ew.yaxis.tick_right()
    ew.yaxis.set_ticks_position('right')

    plt.minorticks_on()
    ew.tick_params(which= 'both', direction='in', right= True, top= True); ew.tick_params(axis= 'both')

    op1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2)); op1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    ew.xaxis.set_minor_locator(ticker.AutoMinorLocator(2)); ew.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


    fig.savefig(f'{DEST_FOLDER}/{BAT_ID}_operando_{COUNT}.png', dpi= 300)
    plt.show()




            



