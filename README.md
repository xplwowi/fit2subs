# fit2subs

### Introduction

This is a dirty piece of code I created in a couple of days
just to temporary plug the hole... I mean lack of such tools.

It allows direct or indirect import of Garmin Descent MK1 
FIT files (activity logs). 

Hopefully Subsurface will have such import feature in near future.

### Features

 - imports data from multiple fit files at once
 - discards data from non-diving activities even if they exist in sellection
 - creates new Subsurface logs or updates existing 
 - supports sites, computers, cylinders creation 
 - consolidates sites using defined radius around GPS coordinates 
 - can include apnea dives
 - filters out dives shorter than defined time 
 - lists tabularized FIT files summaries (to easy pick interesting ones)
 - writes some unsuported data in Extra Info field (e.g. per-dive GF settings)
 - can print everything to stdout for output redirection

### Installation

 - be sure that python version installed in system is >= 3.6 (the only tested interpreter)
 - download or clone updated python-fitparse lib from here:
    https://github.com/xplwowi/python-fitparse
 - install python-fitparse library from its main folder:
   ```
   python setup.py install
    - or -
   python3 setup.py install
    - or - (*nixes, systemwide)
   sudo python setup.py install
   ```
 - clone or download *this* repository
 - run fit2subs.py with options described below, in **Usage** section
 - you can play with some sample diving / non-diving activities available in folder *fit-files*

    Listed fit-files directory content with two non-diving activities:
   ```
    $ ./fit2subs.py -f fit-files/* -l 
   
    Garmin Descent MK1 data converter for Subsurface Dive Log (FIT to XML)
    
    FIT files summary...
    
    _________File__________|_______Date________|_____Activity______|__Time__|_Depth__
    2018-01-06-11-47-38.fit|2018-01-06 11:47:38|single_gas_diving  |   39:41| 28.36 m
    2018-04-03-12-07-44.fit|2018-04-03 12:07:44|hiking             |   33:50|        
    2018-04-11-10-56-59.fit|2018-04-11 10:56:59|multi_gas_diving   | 1:25:15| 68.88 m
    2018-05-07-12-27-54.fit|2018-05-07 12:27:54|walking            |   33:01|        
    2018-08-10-16-48-08.fit|2018-08-10 16:48:08|apnea_diving       |   38:48|  3.51 m
    2018-08-11-09-56-30.fit|2018-08-11 09:56:30|single_gas_diving  |   58:35| 27.02 m
    2018-08-11-14-11-36.fit|2018-08-11 14:11:36|single_gas_diving  | 1:05:30| 20.63 m
    2018-08-13-13-48-26.fit|2018-08-13 13:48:26|multi_gas_diving   | 1:09:05| 15.23 m
    2018-08-14-09-50-38.fit|2018-08-14 09:50:38|single_gas_diving  | 1:03:30| 36.22 m
    
    Sucessfully listed 9 of 9 FIT files
   ```
And please, don't judge diving profiles from fit-files folder. 
Some errors and violations were forced intentionally to collect data
for FIT protocol reverse engineering. I dived with second
computer and tried switching Garmin to wrong tanks, violating MODs,
setting sick depth and time alarms, ...


### Usage

```
Garmin Descent MK1 data converter for Subsurface Dive Log (FIT to XML)

usage: fit2subs.py [-h] -f FIT_SOURCE [FIT_SOURCE ...] [-o OUTPUT_LOG]
                   [-i INPUT_LOG] [-d [DUMP_DIR]] [-c BIG_CIRCLE SMALL_CIRCLE]
                   [-a] [-t MINUTES] [-l] [-p] [-n]

optional arguments:
  -h, --help            show this help message and exit
  -f FIT_SOURCE [FIT_SOURCE ...], --fitfiles FIT_SOURCE [FIT_SOURCE ...]
                        Source FIT file(s) list. Shell wildcards can be used.
  -o OUTPUT_LOG, --outlog OUTPUT_LOG
                        Destination Subsurface log file. Can be the same as
                        input log.
  -i INPUT_LOG, --inlog INPUT_LOG
                        Source Subsurface log file. If not specified, output
                        log will be created from scratch.
  -d [DUMP_DIR], --dump [DUMP_DIR]
                        Dump FIT files content to text files. Optional
                        destination directory can be specified. If not - dump
                        files will be created in current directory.
  -c BIG_CIRCLE SMALL_CIRCLE, --circles BIG_CIRCLE SMALL_CIRCLE
                        Set custom radii in meters of circles used for
                        existing dive site reuse detection.
  -a, --apnea           Apnea dives will be also processed.
  -t MINUTES, --timelimit MINUTES
                        Discard dives shorter than this value. Fractional
                        values can be used. Does not affect apnea dives.
  -l, --listdives       Print only summary of chosen FIT files.
  -p, --pipeoutput      Dump data to stdout and suppress all processing
                        messages.
  -n, --nonumbering     Dive numbering won't be set when output log is created
                        from scratch. Allows autonumbering in further joining
                        output file with existing log using Subsurface "import
                        log" feature.

examples:
 fit2subs.py -f 2018-04-03-12-07-44.fit -d dump-files

    Dumps content of file '2018-04-03-12-07-44.fit' to 'dump-files' directory

 fit2subs.py -f myfits/*.fit -d

    Dumps content of all FIT files from 'myfits' directory to current directory

 fit2subs.py -f 2018-04-03-12-07-44.fit -i mylog.ssrf -o new.xml -c 600 80 -a -t 1.5

    Appends dive from file '2018-04-03-12-07-44.fit' to 'mylog.ssrf' and saves
    result to 'new.xml'. Apnea dives are also added. Dives shorter than 90s (1.5min)
    will be discarded. New dive site will be created if distance to the nearest site
    found in log is longer than 600m. Site found in log will be reused if distance
    is shorter than 80m. If distance is between 80m and 600m, new site with new GPS
    coordinates will be created but found name of adjacent site will be reused.

 fit2subs.py -f 2018-04-03-12-07-44.fit -o newlog.xml -n

    Creates completely new Subsurface log from FIT files content. Dive numbers won't
    be created. Subsurface 'import dive log' function will create missing numbers
    automatically.

 fit2subs.py -f myfits/*.fit -l

    Lists FIT files summaries in tabular format.

 fit2subs.py -f myfits/*.fit -l -p

    Lists FIT files summaries to stdout in pipe-character separated format, 
    convenient for redirecting output to (eventual) external GUI file picker.
```
