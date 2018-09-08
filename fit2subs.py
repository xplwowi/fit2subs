#!/usr/bin/env python3

# Copyright (c) 2018 Wojciech Wieckowski
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# description     :Subsurface log importer for Garmin FIT files
# author-email    :xplwowi@gmail.com

__version__ = '0.13'

import hashlib
import datetime
import sys
import os
import glob
import argparse
import math
from fitparse import FitFile
from fitparse.utils import FitParseError
try:
    from lxml import etree as ETree  # Prefered as it keeps attrib order as is
except ImportError:
    import xml.etree.ElementTree as ETree  # Version from standard lib sorts attribs alphabetically

# Constants for default settings
NEW_SITE_DISTANCE_MIN = 500  # ('big circle') Min distance [m] to create new site
OLD_COORDS_DISTANCE_MAX = 50  # ('small circle') Max distance [m] to attach dive to existing coords and site UUID


def hashit(in_list):
    """Calculates 4 HEX digits hash string from input list"""
    m = hashlib.sha1()
    for item in in_list:
        m.update(item.encode('utf-8'))
    digest = m.hexdigest()[:8]
    return digest


def errprint(*args, **kwargs):
    """Prints to stderr"""
    print(*args, file=sys.stderr, **kwargs)


def npprint(*args, **kwargs):
    """Prints to stdout conditionally"""
    if not settings.pipe_output:  # Supress printing when pipe output selected
        print(*args, file=sys.stdout, **kwargs)


def pretty_print(elem, level=0):
    """In-place prettyprint etree XML formatter"""

    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            pretty_print(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def haversine_distance(pos_from, pos_to):
    """Calculates circular meters distance between two GPS positions"""
    lat_from, long_from = (float(pos_from.split(' ')[0]), float(pos_from.split(' ')[1]))  # Pair in string to two floats
    lat_to, long_to = (float(pos_to.split(' ')[0]), float(pos_to.split(' ')[1]))  # Pair in string to two floats
    e_radius = 6371000  # Earth radius in meters

    delta_lat = math.radians(lat_to - lat_from)
    delta_long = math.radians(long_to - long_from)

    a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(math.radians(lat_from)) \
        * math.cos(math.radians(lat_to)) * math.sin(delta_long / 2) * math.sin(delta_long / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = int(e_radius * c)

    return distance  # Meters


def timestamps_diff(timestamp_from, timestamp_to):
    """Difference between two timestamp strings
    returns delta in Subsurface 'MMM:SS min' format
    """
    delta = int(timestamp_to) - int(timestamp_from)
    return '%d:%02d min' % (delta / 60, delta % 60)


def date_time_loc(timestamp_utc, offset_secs):
    """Adds offset seconds to timestamp string to set local time
    returns tuple of 'date' and 'time' strings
    """
    utc_reference = 631065600  # timestamp for UTC 00:00 Dec 31 1989

    # Compute the two's complement of int time_offset value
    bits = 32
    if (int(offset_secs) & (1 << (bits - 1))) != 0:   # If sign bit is set
        offset_secs = int(offset_secs) - (1 << bits)  # Compute negative value

    datetime_loc = datetime.datetime.utcfromtimestamp(utc_reference + int(timestamp_utc) + int(offset_secs))

    date_loc = datetime.datetime.strftime(datetime_loc, '%Y-%m-%d')  # Date format used in Subsurface
    time_loc = datetime.datetime.strftime(datetime_loc, '%H:%M:%S')  # Time format used in Subsurface
    return date_loc, time_loc


def sc_to_degs(semicircles):
    """Semicirceles (Garmin GPS coordinates storage standard)
    to decimal degrees conversion [float].
    """
    if semicircles is None:
        return None

    # Seems that Subsurface uses 6 decimal digits after dot
    degs = '%.6f' % round(int(semicircles) * (180 / 2 ** 31), 6)
    return degs


def dd_to_dms(lat, long):
    """Decimal to DMS coordiantes convertion"""

    # Longtitude
    split_degx = math.modf(float(long))
    degrees_x = int(split_degx[1])
    minutes_x = abs(int(math.modf(split_degx[0] * 60)[1]))
    seconds_x = abs(round(math.modf(split_degx[0] * 60)[0] * 60, 3))

    # Latitude
    split_degy = math.modf(float(lat))
    degrees_y = int(split_degy[1])
    minutes_y = abs(int(math.modf(split_degy[0] * 60)[1]))
    seconds_y = abs(round(math.modf(split_degy[0] * 60)[0] * 60, 3))

    if degrees_x < 0:
        e_or_w = "W"
    else:
        e_or_w = "E"

    if degrees_y < 0:
        n_or_s = "S"
    else:
        n_or_s = "N"

    long_dms = '%d\u00b0%02d\'%06.3f"%s' % (abs(degrees_x), minutes_x, seconds_x, e_or_w)
    lat_dms = '%d\u00b0%02d\'%06.3f"%s' % (abs(degrees_y), minutes_y, seconds_y, n_or_s)

    return '{} {}'.format(lat_dms, long_dms)


def fit_dump(fit_file, dump_file):
    """Dumps all data from all records to text file"""

    # Fit file opening
    try:
        fitparser = FitFile(fit_file)
    except Exception as err:  # Catches fit file parsing errors
        print("parse error: {}".format(err))
        return False

    df_handle = open(dump_file, 'w')

    msg_unique_list = []
    msg_unknown_list = []
    msg_known_dict = {}

    messages = fitparser.get_messages()  # Fitparse messages iterator object

    # Go through all the data entries
    for record in messages:
        if record.mesg_num not in msg_unique_list:
            msg_unique_list.append(record.mesg_num)

        if 'unknown' in record.name.lower():
            if record.mesg_num not in msg_unknown_list:
                msg_unknown_list.append(record.mesg_num)
        else:
            if record.mesg_num not in msg_known_dict:
                msg_known_dict[record.mesg_num] = record.name

        df_handle.write('name: {}\n'.format(record.name))
        df_handle.write('mesg_num: {}\n'.format(record.mesg_num))

        for record_data in record:
            df_handle.write('    {}\n'.format(record_data))

        df_handle.write('==========================\n')

    msg_unique_list.sort()
    msg_unknown_list.sort()

    df_handle.write('\n======== SUMMARY =========\n')
    df_handle.write('\nUnique records: {}\n'.format(msg_unique_list))
    df_handle.write('\nUnknown records: {}\n'.format(msg_unknown_list))
    df_handle.write('\nKnown records: {}\n'.format(msg_known_dict))

    df_handle.close()

    return True


def fit_summary(fit_file):
    """Returns FIT file summary in tuple of boolean and dictionary.
    First field contains success status."""

    content = {'file': os.path.basename(fit_file), 'time': '', 'sport': '', 'elapsed': '', 'depth': ''}

    # Fit file opening
    try:
        fitparser = FitFile(fit_file)
    except FitParseError:  # Catches fit file parsing errors
        content['sport'] = 'parse_error'
        return False, content

    messages = fitparser.get_messages()  # Fitparse messages iterator object
    time_offset = 0

    for record in messages:
        if record.name == 'device_settings':
            val_dict = record.get_values()
            time_offset = int(val_dict['time_offset'])

        if record.name == 'session':

            val_dict = record.get_values()
            try:
                sport = str(val_dict['sport'])
                if sport in ('53', '54', '55', '56', '57'):
                    content['sport'] = str(val_dict['sub_sport'])  # Use 'sub_sport' field for diving
                else:  # Or just sport for other activities
                    content['sport'] = str(val_dict['sport'])
            except KeyError:
                pass

            try:
                content['elapsed'] = str(val_dict['total_elapsed_time'])
            except (KeyError, TypeError, ValueError):
                pass

            try:
                # Decision to use dict values above caused stupid calculations below...
                epoch = datetime.datetime(1989, 12, 31)  # Garmin epoch
                utc_delta = val_dict['start_time'] - epoch
                time_stamp = int(utc_delta / datetime.timedelta(seconds=1))
                loc_time = date_time_loc(time_stamp, time_offset)
                content['time'] = '{} {}'.format(loc_time[0], loc_time[1])
            except (KeyError, TypeError, ValueError):
                pass

        if record.name == 'dive_summary':

            val_dict = record.get_values()
            if val_dict['reference_mesg'] == 'session':
                try:
                    content['depth'] = '%.2f' % float(val_dict['max_depth'])
                except (KeyError, TypeError, ValueError):
                    pass

    return True, content


class RecordDecoder(object):
    """Creates dictionary of record fields objects
    and gives access to fields with possible conversion to Subsurface xml metric standard
    """

    def __init__(self):
        self._fields_dic = {}

    @staticmethod
    def _units_conv(item):
        """Values convertion to Subsurface metric xml standard"""

        if item.value is None:
            return None

        # Seems that FIT files use metric units only - imperial converters not needed
        if item.units == 's':  # [s] to min
            rd = round(float(item.value))
            strval = '%d:%02d min' % (rd / 60, rd % 60)
        elif item.units == 'm':  # Meters rounded to 0.01
            strval = '%.2f m' % round(float(item.value), 2)
        elif item.units == 'C':  # Celcius rounded to 0.1
            strval = '%.1f C' % round(float(item.value), 1)
        elif item.units == 'percent':  # percent to %
            strval = '%d%%' % int(item.value)
        elif item.units == 'OTUs':  # Nothing to do - used only in extra info
            strval = '%d OTUs' % int(item.value)
        elif item.units == 'kg/m^3':  # No calculations needed - units equal
            strval = '%.1f g/l' % float(item.value)
        else:  # Other source units to raw item value without units suffix
            strval = str(item.value)

        return strval

    def load_record(self, record):
        """Fills dictionary with item objects"""
        self._fields_dic = {}
        for item in record:
            self._fields_dic[item.name] = item

    def field(self, field, noconv=False, raw=False):
        """Access to specific field
        if noconv=True - strips [units] but leaves field unconverted
        if raw=True - returns raw field, not converted by fitparse library parsers
        """
        try:
            if self._fields_dic[field].value is None:
                return None

            if noconv:
                return str(self._fields_dic[field].value)
            elif raw:
                return str(self._fields_dic[field].raw_value)
            else:
                return self._units_conv(self._fields_dic[field])
        except KeyError:
            return None


class DiveLog(object):
    """Log processing class"""

    def __init__(self):
        # Private vars
        if settings.in_subslog is not None:  # Detects if source Subsurface log is used
            self._log_file = settings.in_subslog
            self._tree = ETree.parse(self._log_file)
        else:  # No source log - let's create empty XML structure
            self._log_file = None
            self._tree = ETree.ElementTree(ETree.fromstring('<divelog program="subsurface" version="3">'
                                                            '<settings/><divesites/><dives/></divelog>'))
        self._root = self._tree.getroot()
        self._current_dive_element = None
        self._current_computer_element = None
        self._sample_template = {'time': None, 'depth': None, 'temp': None, 'ndl': None, 'tts': None, 'in_deco': None,
                                 'stoptime': None, 'stopdepth': None, 'cns': None, 'heartbeat': None}
        self._dc_data_template = {'model': None, 'deviceid': None, 'serial': None, 'firmware': None}
        self._dive_data_template = {'number': None, 'divesiteid': None, 'date': None, 'time': None, 'duration': None}
        self.travel_gas_index = 0
        self._sample_cache = self._sample_template.copy()
        # Public vars
        self.time_offset = None  # Offset used to calculate dive start time
        self.dive_detected = False  # Dive activity found and started (flag)
        self.processed_count = 0
        self.activity_start_timestamp = None  # Reference time for time offset calculations
        self.lowest_temp = None  # Used to store found lowest temp values during records processing
        self.firstrecord = True  # First sample flag - for calculating initial air pressure
        self.dc_data = self._dc_data_template.copy()  # Computer data
        self.dive_data = self._dive_data_template.copy()
        self.sample_data = self._sample_template.copy()

    @property
    def last_dive_no(self):
        """Finds highest dive # in Subsurface log"""
        dive_elements = self._root.findall('./dives/dive')
        try:
            # List comprehension used to find highest dive number already present in log
            count = max([int(number) for number in [dive.attrib['number'] for dive in dive_elements]])
            return count
        except (ValueError, KeyError):
            return 0

    def reset_all_vars(self):
        """Resets some vars to default values.
        To be called before importing new file (just in case)
        """
        self._current_dive_element = None
        self._current_computer_element = None
        self.time_offset = None
        self.dive_detected = False
        self.activity_start_timestamp = None
        self.lowest_temp = None
        self.firstrecord = True
        self.dc_data = self._dc_data_template.copy()
        self.dive_data = self._dive_data_template.copy()
        self.sample_data = self._sample_template.copy()
        self._sample_cache = self._sample_template.copy()

    def reset_sample_vars(self):
        """Resets vars before storing single sample record.
        To be called before importing new record (just in case)
        """
        self.sample_data = self._sample_template.copy()

    def save_log(self, file):
        """Saves modified Subsurface log"""
        pretty_print(self._root)  # Re-indents entire XML tree
        self._tree.write(file)

    def pipe_log(self):
        """Prints log to stdout"""
        pretty_print(self._root)
        print(ETree.tostring(self._root, encoding='unicode'))

    # -----------------------------
    # XML log manipulation methods
    # -----------------------------
    def update_dc(self):
        """Creates new computer entry or updates FW version"""
        dc_element = self._root.find('./settings/divecomputerid[@deviceid="{}"]'.format(self.dc_data['deviceid']))

        if dc_element is None:  # Computer absent
            settings_element = self._root.find('./settings')
            new_dc = ETree.SubElement(settings_element, 'divecomputerid')
            for attr in self.dc_data:
                if self.dc_data[attr] is not None:
                    new_dc.set(attr, self.dc_data[attr])

    def update_site(self, lat, long):
        """Creates or updates dive sites"""

        # Constants used to decide if to create new site or attach to existing one

        new_site_distance_min = settings.big_circle  # Min distance [m] to create new site
        old_coords_distance_max = settings.small_circle  # Max distance [m] to attach dive to existing coords and site UUID

        # When calculated distance value is between NEW_SITE_DISTANCE_MIN and OLD_COORDS_DISTANCE_MAX
        # new site element with existing description will be created

        sites_element = self._root.find('./divesites')
        if lat is not None and long is not None:
            current_coord = lat + ' ' + long  # Both coordinates in Subsurface 'gps' attrib format

            found_near_site_element = None
            found_near_site_distance = new_site_distance_min + 1  # Set initial value higher than minimum

            for site_element in sites_element.findall('site'):  # Go thorough all stored sites
                stored_coord = site_element.get('gps')
                if stored_coord is not None:
                    distance = haversine_distance(current_coord, stored_coord)  # Calculate circular distance [m]
                    if distance <= new_site_distance_min and distance <= found_near_site_distance:  # Found site near enough
                        found_near_site_element = site_element
                        found_near_site_distance = distance

        else:  # No coordinates found - need to supplement some missing values
            found_near_site_element = None
            current_coord = None
            found_near_site_distance = 0

        if found_near_site_element is not None:  # If near site found
            if found_near_site_distance <= old_coords_distance_max:  # Found site is close enough to use old coords
                self.dive_data['divesiteid'] = found_near_site_element.get('uuid')
            else:  # Create new site element with old name
                new_site = ETree.SubElement(sites_element, 'site')
                new_site.set('gps', current_coord)
                sitename = found_near_site_element.get('name')
                uuid = hashit([self.activity_start_timestamp, sitename])
                new_site.set('name', sitename)  # Use old name
                new_site.set('uuid', uuid)
                self.dive_data['divesiteid'] = uuid
        else:  # Near site not found - let's create completely new
            new_site = ETree.SubElement(sites_element, 'site')
            if current_coord is not None:
                new_site.set('gps', current_coord)
            sitename = '= first time here {} {} ='.format(self.dive_data['date'],
                                                          self.dive_data['time'])
            uuid = hashit([self.activity_start_timestamp, sitename])
            new_site.set('name', sitename)  # Create fake name
            new_site.set('uuid', uuid)
            self.dive_data['divesiteid'] = uuid

    def start_dive(self):
        """Creates 'dive' and 'divecomputer' elements"""
        diveid = hashit([self.activity_start_timestamp])  # Unique ID from timestamp
        dives_element = self._root.find('./dives')

        # Check if dive isn't already present in log
        if dives_element.find('./dive/divecomputer[@diveid="{}"]'.format(diveid)) is not None:
            npprint('Dive "{}" already imported, nothing to do!'.format(diveid))
            return False  # Dive already in log

        self._current_dive_element = ETree.SubElement(dives_element, 'dive')
        self._current_computer_element = ETree.SubElement(self._current_dive_element, 'divecomputer')
        self._current_computer_element.set('model', self.dc_data['model'])
        self._current_computer_element.set('deviceid', self.dc_data['deviceid'])
        self._current_computer_element.set('diveid', diveid)
        return True

    def set_subsport(self, sub_sport):
        """Sets 'Freedive' attrib"""
        if sub_sport in ('apnea_diving', 'apnea_hunting'):  # Subsurface does not set this attrib for OC
            self._current_computer_element.set('dctype', 'Freedive')

    def set_travel_gas(self):
        """Sets travel gas index"""
        self.add_gas_change(self.travel_gas_index, self.activity_start_timestamp)

    def add_summary_temp(self, air):
        """Adds water/air temperature element
        hack used - missing reliable air temperature data in diving activity
        Garmin Connect uses internet weather service
        """
        if air is not None:
            new_temp = ETree.SubElement(self._current_computer_element, 'temperature')
            new_temp.set('water', str(self.lowest_temp) + ' C')
            new_temp.set('air', air)

    def add_surface_pressure(self, pressure):
        """Adds surface pressure
        hack used - missing reliable air pressure data in diving activity
        """
        if pressure is not None:
            new_press = ETree.SubElement(self._current_computer_element, 'surface')
            new_press.set('pressure', str(pressure) + ' bar')

    def add_summary_depths(self, mx, av):
        """Adds depths to dive summary"""
        if mx is not None and av is not None:
            new_depth = ETree.SubElement(self._current_computer_element, 'depth')
            new_depth.set('max', mx)
            new_depth.set('mean', av)

    def add_notes(self, note):
        """Add note to dive element"""
        new_note = self._current_dive_element.find('./notes')
        if new_note is None:
            new_note = ETree.SubElement(self._current_dive_element, 'notes')
        else:
            note = new_note.text + '\n' + note
        new_note.text = note

    def add_cylinder(self, cylinder_no, oxygen_perc, helium_perc):
        """Creates cylinders from MK1 mixes list"""
        new_cylinder = ETree.SubElement(self._current_dive_element, 'cylinder')

        if int(round(float(oxygen_perc.strip('%')))) >= 99:
            description = '%s. Oxygen' % cylinder_no

        elif int(round(float(helium_perc.strip('%')))) == 0 and int(round(float(oxygen_perc.strip('%')))) == 21:
            description = '%s. Air' % cylinder_no

        elif int(round(float(helium_perc.strip('%')))) == 0 and int(round(float(oxygen_perc.strip('%')))) > 21:
            description = '%s. EAN%i' % (cylinder_no,
                                         int(round(float(oxygen_perc.strip('%')))))

        else:
            description = '%s. %i/%i' % (cylinder_no,
                                         int(round(float(oxygen_perc.strip('%')))),
                                         int(round(float(helium_perc.strip('%')))))

        new_cylinder.set('description', description)
        new_cylinder.set('o2', oxygen_perc)
        new_cylinder.set('he', helium_perc)

    def add_extra_data(self, key, value):
        """Adds additional info as 'extradata' elements"""
        if key is not None and value is not None:
            new_extra = ETree.SubElement(self._current_computer_element, 'extradata')
            new_extra.set('key', key)
            new_extra.set('value', value)

    def add_water_data(self, salinity):
        """Adds water density"""
        if salinity is not None:
            new_water = ETree.SubElement(self._current_computer_element, 'water')
            new_water.set('salinity', salinity)

    def add_surfacetime(self, time):
        """Creates 'surfacetime' element
        ...but it seems that Subsurface doesn't care about it
        """
        if time is not None:
            new_surfacetime = ETree.SubElement(self._current_computer_element, 'surfacetime')
            new_surfacetime.text = time

    def add_event(self, event_type, timestamp, severity):
        """General events handling"""
        if event_type is not None and timestamp is not None:
            new_event = ETree.SubElement(self._current_computer_element, 'event')
            new_event.set('time', timestamps_diff(self.activity_start_timestamp, timestamp))
            new_event.set('type', '26')  # From libdivecomputer parser_sample_event_t enum SAMPLE_EVENT_STRING
            new_event.set('flags', str(severity << 2))  # From garmin_parser.c
            new_event.set('name', event_type)

    def add_gas_change(self, cylinder_index, timestamp):
        """Adds gas change events"""
        if cylinder_index is not None and timestamp is not None:
            new_event = ETree.SubElement(self._current_computer_element, 'event')
            new_event.set('time', timestamps_diff(self.activity_start_timestamp, timestamp))
            new_event.set('type', '25')  # From libdivecomputer parser_sample_event_t enum SAMPLE_EVENT_GASCHANGE2
            new_event.set('flags', str(int(cylinder_index) + 1))
            new_event.set('name', 'gaschange')
            new_event.set('cylinder', cylinder_index)
            # new_event.set('o2', '?%')
            # new_event.set('he', '?%')

    def save_dive(self):
        """Saves dive element data attributes"""
        for attr in self.dive_data:
            if self.dive_data[attr] is not None:
                self._current_dive_element.set(attr, self.dive_data[attr])

    def add_sample(self):
        """Adds sample record"""
        new_sample = ETree.SubElement(self._current_computer_element, 'sample')
        for key in self._sample_template:  # Should retain items order from template in python AFAIK >= 3.6
            curr_val = self.sample_data.get(key)
            prev_val = self._sample_cache.get(key)

            if key == 'in_deco':  # Relate current flag value to its previous state and required deco time
                next_stop_time = self.sample_data.get('stoptime')
                if next_stop_time is not None and next_stop_time != '0:00 min':  # Entering deco
                    if prev_val is None or prev_val == '0':
                        curr_val = '1'
                elif prev_val == '1' and next_stop_time == '0:00 min':  # Finishing deco
                    curr_val = '0'

            # Skip attribs that should always exist (time, depth) and update cached value if current differs.
            # Used to compress Subsurface log by adding only attribs changing previous (cached) state.
            if curr_val is not None:
                if key in ('time', 'depth') or prev_val != curr_val:
                    new_sample.set(key, curr_val)

                self._sample_cache[key] = curr_val  # Update values cache


class Settings(object):
    """Stores processing settings globally and provides access and checking functions"""
    def __init__(self):
        self.in_subslog = None  # Path to original Subsurface log
        self.out_subslog = None  # Path to destination Subsurface log
        self.fit_files = None  # Fit files list
        self.dump_dir = None  # Output dir for dumps
        self.big_circle = None  # From module constants
        self.small_circle = None  # From module constants
        self.apnea = None  # Include apnea dives
        self.min_time = None  # Minimal dive time
        self.list_only = None  # List FIT summaries only
        self.pipe_output = None  # Output everything to stdout for piping
        self.no_numbering = None  # Numbering of dives optionally disabled

    @staticmethod
    def _has_access(filepath, mode):
        """Checks file reading, cwriting and creating access"""
        if mode == 'r':  # Check binary reading access
            try:
                f = open(filepath, 'rb')
                _ = f.read(1)  # read 1 byte
                f.close()
                return True
            except IOError:
                errprint("File: '{}' cannot be opened for reading".format(filepath))
                return False

        if mode == 'w':  # Check writing access
            try:
                if os.path.exists(filepath):  # Only open for appending and then close if file exists
                    f = open(filepath, 'a')
                    f.close()
                else:
                    f = open(filepath, 'w')  # Create new and delete immediately
                    f.close()
                    os.remove(filepath)
                return True
            except (IOError, OSError):
                errprint("File: '{}' cannot be opened for writing".format(filepath))
                return False

    def settings_from_args(self, cmdparser):
        """Sets container vars with data from command line parser"""

        args = cmdparser.parse_args()

        self.fit_files = args.fitfiles
        self.in_subslog = args.inlog
        self.out_subslog = args.outlog
        self.dump_dir = args.dump
        self.apnea = args.apnea
        self.min_time = args.timelimit
        self.list_only = args.listdives
        self.pipe_output = args.pipeoutput
        self.no_numbering = args.nonumbering
        if args.circles is not None:
            self.big_circle = args.circles[0]
            self.small_circle = args.circles[1]

    def _check_params_pattern(self, ok_pattern):
        """Checks parameters combination pattern"""

        no_error_found = True
        for key, value in ok_pattern.items():
            setting = vars(self)[key]
            if setting is None:
                if value == 'set':
                    no_error_found = False
            elif setting is True:
                if value == 'unset':
                    no_error_found = False
            elif setting is False:
                if value == 'set':
                    no_error_found = False
            elif setting is not None:
                if value == 'unset':
                    no_error_found = False
        return no_error_found

    def check_settings(self):
        """Checks processing options"""

        # Check if wildcards have been expanded by shell (e.g. cmd is too dumb to do it)
        wildcards = '*?[]{}'
        temp_list = []

        for fit_file in settings.fit_files:
            # Check if wildcards are present in current item
            if any(w in fit_file for w in wildcards):  # Wildcards persisted - let's expand them internally
                exp_list = glob.glob(fit_file)
                if exp_list:
                    exp_list = sorted(exp_list)  # Hopefully filename contains structured date - provide proper import order
                    temp_list.extend(exp_list)
                else:  # In spite of all keep item in list to be eventually reported by checkers from below
                    temp_list.append(fit_file)
            else:  # Add file without wildcards as is
                temp_list.append(fit_file)

        self.fit_files = temp_list

        # Parameters consistency check
        if self.dump_dir is not None:  # Combination for dumping
            ok_pattern = {'in_subslog': 'unset', 'out_subslog': 'unset', 'big_circle': 'unset',
                          'small_circle': 'unset', 'apnea': 'set', 'min_time': 'unset',
                          'list_only': 'unset', 'pipe_output': 'unset', 'no_numbering': 'unset'}
            if not self._check_params_pattern(ok_pattern):
                errprint('Wrong parameters combination for dumping!')
                exit(2)

        elif self.list_only:  # Combination for listing
            ok_pattern = {'in_subslog': 'unset', 'out_subslog': 'unset', 'big_circle': 'unset',
                          'small_circle': 'unset', 'apnea': 'set', 'min_time': 'unset',
                          'dump_dir': 'unset', 'no_numbering': 'unset'}
            if not self._check_params_pattern(ok_pattern):
                errprint('Wrong parameters combination for dives listing!')
                exit(2)

        elif self.pipe_output:  # Combination for piping
            ok_pattern = {'out_subslog': 'unset', 'dump_dir': 'unset'}
            if not self._check_params_pattern(ok_pattern):
                errprint('Wrong parameters combination for piping output!')
                exit(2)

        elif self.out_subslog is not None:  # Combination for writing log file
            ok_pattern = {'pipe_output': 'unset', 'dump_dir': 'unset'}
            if not self._check_params_pattern(ok_pattern):
                errprint('Wrong parameters combination for writing log output!')
                exit(2)
        else:
            if self.out_subslog is None and not self.pipe_output:  # Log output selection must be set
                errprint('No output for log selected! File or pipe must be set!')
                exit(2)

        error_count = 0
        # Fit files access
        for filename in list(self.fit_files):  # Had to create local copy to allow enumerating lists with removed items
            if not self._has_access(filename, 'r'):
                self.fit_files.remove(filename)
                errprint('File removed from FITs list!\n')

        if not self.fit_files:  # Is FIT files list empty?
            errprint("No FIT files to process!\n")
            error_count += 1

        if self.out_subslog is not None and not self._has_access(self.out_subslog, 'w'):  # Is writing possible?
            errprint('Output log cannot be created!\n')
            error_count += 1

        if self.in_subslog is not None and not self._has_access(self.in_subslog, 'r'):  # Is reading possible?
            errprint('Input log cannot be opened!\n')
            error_count += 1

        if self.dump_dir is not None:  # Is writing dump files possible?
            test_file = os.path.join(self.dump_dir, 'x9wp3ui8t.txt')
            if not self._has_access(test_file, 'w'):
                errprint('Dump file cannot be created!\n')
                error_count += 1

        if error_count > 0:  # Error found in previous checks
            errprint("Cannot continue, exiting...")
            sys.exit(2)


def message_processor(dive_log, fit_file):
    """Interprets messages read from FIT files"""

    # Fit file opening
    try:
        fitparser = FitFile(fit_file)
    except Exception as err:  # Catches fit file parsing errors
        npprint(err)
        return False

    decoder = RecordDecoder()  # Record fields access object

    # Minimal dive time limit filtering
    if settings.min_time is not None:
        messages = fitparser.get_messages()  # Fitparse messages iterator object

        # As summary record is at the end of FIT files, extra record traversal is required (time consuming)
        for record in messages:
            if record.name == 'sport':
                val_dict = record.get_values()
                if val_dict['sub_sport'] in ('apnea_diving', 'apnea_hunting'):  # Skip limit if apnea dives are included
                    break

            if record.name == 'dive_summary':
                val_dict = record.get_values()
                if val_dict['reference_mesg'] == 'session':
                    bottom_time = val_dict['bottom_time']
                    if bottom_time < settings.min_time * 60:
                        npprint('discarded - bottom time: %.2f min '
                                'shorter than limit: %.2f min' % (bottom_time / 60, settings.min_time))
                        return False  # Skip futrher processing

    # Possible re-assignment because previous one uses (and remembers) dictionary fields access interface
    messages = fitparser.get_messages()  # Fitparse messages iterator object

    # Records processing
    for record in messages:  # Iterate through all data messages

        if record.name == 'device_info':  # Dive computer info [1]
            decoder.load_record(record)
            if decoder.field('device_index') == 'creator':
                fw = decoder.field('software_version')
                product = decoder.field('garmin_product')
                if product == '2859':
                    product = 'Garmin Descent MK1'
                product = '{} ({})'.format(product, fw)
                dive_log.dc_data['model'] = product
                dive_log.dc_data['serial'] = decoder.field('serial_number')

                # DC unique ID created by hashing 'model+serial+fw' fields
                dive_log.dc_data['deviceid'] = hashit([product, dive_log.dc_data['serial'], fw])

        if record.name == 'event':  # Catches 'event' type records [3]
            decoder.load_record(record)

            if decoder.field('event_type') == 'start':  # Activity start event
                dive_log.activity_start_timestamp = decoder.field('timestamp', raw=True)

            # Alerts =======
            elif decoder.field('event') == '56' and decoder.field('data') == '0':  # Deco required
                dive_log.add_event('Deco required', decoder.field('timestamp', raw=True), 2)

            elif decoder.field('event') == '56' and decoder.field('data') == '1':  # Gass switch prompted
                pass  # Don't waste graph

            elif decoder.field('event') == '56' and decoder.field('data') == '2':  # Surface
                pass  # Don't waste graph

            elif decoder.field('event') == '56' and decoder.field('data') == '3':  # Approaching NDL
                dive_log.add_event('Approaching NDL', decoder.field('timestamp', raw=True), 2)

            elif decoder.field('event') == '56' and decoder.field('data') == '4':  # ppO2 soft violation
                dive_log.add_event('ppO2 warning', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '5':  # ppO2 high critical
                dive_log.add_event('ppO2 critical high', decoder.field('timestamp', raw=True), 4)

            elif decoder.field('event') == '56' and decoder.field('data') == '6':  # ppO2 low critical
                dive_log.add_event('ppO2 critical low', decoder.field('timestamp', raw=True), 4)

            elif decoder.field('event') == '56' and decoder.field('data') == '7':  # Time alert
                dive_log.add_event('Time alert', decoder.field('timestamp', raw=True), 2)

            elif decoder.field('event') == '56' and decoder.field('data') == '8':  # Depth alert
                dive_log.add_event('Depth alert', decoder.field('timestamp', raw=True), 2)

            elif decoder.field('event') == '56' and decoder.field('data') == '9':  # Deco ceiling broken
                dive_log.add_event('Deco ceiling broken', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '10':  # Deco finished - free to exit
                dive_log.add_event('Deco completed', decoder.field('timestamp', raw=True), 1)

            elif decoder.field('event') == '56' and decoder.field('data') == '11':  # Safety stop ceiling broken
                dive_log.add_event('Safety stop ceiling broken', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '12':  # Safety stop completed
                pass  # Don't waste graph

            elif decoder.field('event') == '56' and decoder.field('data') == '13':  # CNS warning
                dive_log.add_event('CNS warning', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '14':  # CNS critical
                dive_log.add_event('CNS critical', decoder.field('timestamp', raw=True), 4)

            elif decoder.field('event') == '56' and decoder.field('data') == '15':  # OTU warning
                dive_log.add_event('OTU warning', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '16':  # OTU critical
                dive_log.add_event('OTU critical', decoder.field('timestamp', raw=True), 4)

            elif decoder.field('event') == '56' and decoder.field('data') == '17':  # Ascent speed too high
                dive_log.add_event('Ascent speed alert', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '18':  # Alert manualy deleted
                pass  # Don't waste graph

            elif decoder.field('event') == '56' and decoder.field('data') == '19':  # Alert timed out
                pass  # Don't waste graph

            elif decoder.field('event') == '56' and decoder.field('data') == '20':  # Battery low
                dive_log.add_event('Battery low', decoder.field('timestamp', raw=True), 3)

            elif decoder.field('event') == '56' and decoder.field('data') == '21':  # Battery critical
                dive_log.add_event('Battery critical', decoder.field('timestamp', raw=True), 4)

            elif decoder.field('event') == '56' and decoder.field('data') == '22':  # Safety stop begin
                dive_log.add_event('Safety stop begin', decoder.field('timestamp', raw=True), 1)

            elif decoder.field('event') == '56' and decoder.field('data') == '23':  # Approaching deco stop
                dive_log.add_event('Approaching first deco stop', decoder.field('timestamp', raw=True), 1)

            # Gas change ======
            elif decoder.field('event') == '57':  # Mix change - data contains gas number (cylinder index)
                dive_log.add_gas_change(decoder.field('data'), decoder.field('timestamp', raw=True))

            # Dive end events ======
            elif decoder.field('event') == '48':  # Dive finished (end GPS position fix maybe) ???
                pass  # Don't waste graph

            elif decoder.field('event') == '38':  # Dive finished (end GPS position fix maybe) ???
                pass  # Don't waste graph

            else:  # All other events (for reverse engineering only)
                dive_log.add_event('event %s, data %s' % (decoder.field('event'), decoder.field('data')),
                                   decoder.field('timestamp', raw=True))  # Fake event description

            # -- Sample Fields -- #
            # event: timer
            # event_group: 0
            # event_type: start
            # timer_trigger: manual
            # timestamp: 2018-04-11 08:56:59

        if record.name == 'activity':  # Activity record
            decoder.load_record(record)
            if decoder.field('event_type') == 'stop' and dive_log.dive_detected:  # Stop event - last message in log
                dive_log.save_dive()

            # -- Sample Fields -- #
            # event: activity
            # event_group: None
            # event_type: stop
            # local_timestamp: 2018-04-11 12:22:14
            # num_sessions: 1
            # timestamp: 2018-04-11 10:22:14
            # total_timer_time: 5115.432 [s]

        if record.name == 'sport':  # Contains sport type info - diving can be detected here
            decoder.load_record(record)
            if decoder.field('sport') in ('53', '54', '55', '56', '57'):  # Sport types - probably diving uses only #53
                sub_sport = decoder.field('sub_sport')
                if settings.apnea and sub_sport in ('apnea_diving', 'apnea_hunting'):
                    npprint('discarded - apnea dive')
                    return False

                if settings.no_numbering:  # No dive numbering when no input log (enables further import)
                    dive_nr = None
                else:
                    dive_nr = str(dive_log.last_dive_no + 1)
                dive_log.dive_data['number'] = dive_nr
                if not dive_log.start_dive():  # False if dive already imported
                    return False
                dive_log.update_dc()  # Dive computer info
                dive_log.set_subsport(sub_sport)  # Dive mode (OC, apnea, ...)
                dive_log.dive_detected = True  # Processing most of other messages will depend on this flag
            else:
                npprint('discarded - non-diving activity ({})'.format(decoder.field('sport')))
                return False  # Stop processing - no dive actvity found

            # -- Sample Fields -- #
            # name: Mieszanki
            # sport: 53
            # sub_sport: multi_gas_diving

        if record.name == 'dive_settings' and dive_log.dive_detected:  # Dive settings
            decoder.load_record(record)
            dive_log.add_water_data(decoder.field('water_density'))
            dive_log.add_extra_data('GFLow', decoder.field('gf_low'))  # Only global GF setting present in Subsurface
            dive_log.add_extra_data('GFHigh', decoder.field('gf_high'))  # Only global GF setting present in Subsurface
            dive_log.travel_gas_index = decoder.field('message_index')  # Travel gas setting (update from Garmin)

            # -- Sample Fields -- #
            # apnea_countdown_enabled: False
            # apnea_countdown_time: 120
            # backlight_brightness: 50
            # backlight_mode: always_on
            # backlight_timeout: 8
            # bottom_depth: None
            # bottom_time: None
            # gf_high: 95[percent]
            # gf_low: 45[percent]
            # heart_rate_local_device_type: 10
            # heart_rate_source_type: local
            # message_index: 0
            # model: zhl_16c
            # name: None
            # po2_critical: 1.6[percent]
            # po2_deco: 1.4[percent]
            # po2_warn: 1.4[percent]
            # repeat_dive_interval: 60[s]
            # safety_stop_enabled: True
            # safety_stop_time: 180[s]
            # water_density: 1025.0[kg/m^3]
            # water_type: salt

        if record.name == 'device_settings':  # Some important time settings
            decoder.load_record(record)
            dive_log.time_offset = decoder.field('time_offset', raw=True)  # UTC to local time offset value [s]

            # -- Sample Fields -- #
            # active_time_zone: 0
            # activity_tracker_enabled: True
            # auto_activity_detect: 2147483647
            # autosync_min_steps: 100 [steps]
            # autosync_min_time: 15 [minutes]
            # backlight_mode: auto_brightness
            # lactate_threshold_autodetect_enabled: True
            # mounting_side: left
            # move_alert_enabled: True
            # number_of_screens: None
            # tap_interface: auto
            # time_mode: hour24
            # time_offset: 7200 [s]
            # time_zone_offset: 0.0 [hr]
            # utc_offset: 0

        if record.name == 'dive_gas' and dive_log.dive_detected:  # Dive gases list (see message_index)
            decoder.load_record(record)
            # Adding all gases, even disabled to keep cylinders index consistent.
            # Subsurface should discard (present in XML but not displayed) unused ones.
            dive_log.add_cylinder(decoder.field('message_index'),
                                  decoder.field('oxygen_content'),
                                  decoder.field('helium_content'))

            # -- Sample Fields -- #
            # helium_content: 45[percent]
            # message_index: 0
            # oxygen_content: 16[percent]
            # status: enabled

        if record.name == 'dive_summary' and dive_log.dive_detected:  # Dive end summary
            decoder.load_record(record)
            if decoder.field('reference_mesg') == 'session':  # Take only last dive_summary (session not lap summary)
                dive_log.dive_data['duration'] = decoder.field('bottom_time')
                dive_log.add_summary_depths(decoder.field('max_depth'), decoder.field('avg_depth'))
                dive_log.add_extra_data('Start_N2', decoder.field('start_n2'))
                dive_log.add_extra_data('End_N2', decoder.field('end_n2'))
                dive_log.add_extra_data('Serial', dive_log.dc_data['serial'])  # Data present in Subsurface - why?
                dive_log.add_extra_data('O2_toxicity', decoder.field('o2_toxicity'))
                dive_log.add_surfacetime(decoder.field('surface_interval'))
                dive_log.add_notes('Computer reported bottom time: {}'.format(decoder.field('bottom_time')))

            # -- Sample Fields -- #
            # avg_depth: 24.867 [m]
            # bottom_time: 5114.713 [s]
            # dive_number: 5
            # end_cns: 70 [percent]
            # end_n2: 79 [percent]
            # max_depth: 68.884 [m]
            # o2_toxicity: 104 [OTUs]
            # reference_index: 0
            # reference_mesg: session
            # start_cns: 0 [percent]
            # start_n2: 4 [percent]
            # surface_interval: 82759 [s]
            # timestamp: 2018-04-11 10:22:14

        if record.name == 'session' and dive_log.dive_detected:  # Session end summary
            decoder.load_record(record)
            # Tuple of date and time strings in local time
            local_start_timestamp = date_time_loc(decoder.field('start_time', raw=True), dive_log.time_offset)
            dive_log.dive_data['date'] = local_start_timestamp[0]
            dive_log.dive_data['time'] = local_start_timestamp[1]
            dive_log.add_summary_temp(decoder.field('max_temperature'))
            dive_log.add_extra_data('Mean_HR', decoder.field('avg_heart_rate'))
            dive_log.add_extra_data('Max_HR', decoder.field('max_heart_rate'))

            # Entry / Exit GPS coordinates
            entry_lat = sc_to_degs(decoder.field('start_position_lat', noconv=True))
            entry_long = sc_to_degs(decoder.field('start_position_long', noconv=True))
            exit_lat = sc_to_degs(decoder.field('unknown_38', noconv=True))  # Field description missing in SDK 20.67
            exit_long = sc_to_degs(decoder.field('unknown_39', noconv=True))  # Field description missing in SDK 20.67

            # Add notes element with exit coordinates
            if exit_lat is not None and exit_long is not None:
                dive_log.add_notes('Exit coordinates: {}'.format(dd_to_dms(exit_lat, exit_long)))

            # If entry coordinates are missing (immersed before fix achieved), use exit as entry
            if entry_lat is None or entry_long is None:
                entry_lat = exit_lat
                entry_long = exit_long

            # Create site
            dive_log.update_site(entry_lat, entry_long)

            # -- Sample Fields -- #
            # avg_heart_rate: 71 [bpm]
            # avg_speed: 0
            # avg_temperature: 14 [C]
            # max_heart_rate: 92 [bpm]
            # max_temperature: 17 [C]
            # num_laps: 1
            # sport: 53
            # start_position_lat: 513203678 [semicircles]
            # start_position_long: 193472482 [semicircles]
            # start_time: 2018-04-11 08:56:59
            # sub_sport: multi_gas_diving
            # timestamp: 2018-04-11 10:22:14
            # total_calories: 66 [kcal]
            # total_elapsed_time: 5115.432 [s]
            # total_timer_time: 5115.432 [s]
            # total_training_effect: 0.0
            # total_work: None [J]
            # training_stress_score: None [tss]
            # trigger: activity_end
            #
            # -- EXIT coordinates -- #
            # unknown_38: 513206586
            # unknown_39: 193473160

        if record.name == 'record' and dive_log.dive_detected:  # Samples processing
            decoder.load_record(record)
            dive_log.reset_sample_vars()

            if dive_log.firstrecord:  # Stupid hack to calculate entry air pressure
                abs_press = float(decoder.field('absolute_pressure', noconv=True)) / 100000
                wtr_press = float(decoder.field('depth', noconv=True)) / 10
                dive_log.add_surface_pressure(abs_press - wtr_press)
                # Add first gas change to travel gas at the begining of the dive
                dive_log.set_travel_gas()
                dive_log.firstrecord = False

            dive_log.sample_data['heartbeat'] = decoder.field('heart_rate')
            dive_log.sample_data['depth'] = decoder.field('depth')
            dive_log.sample_data['temp'] = decoder.field('temperature')
            dive_log.sample_data['stopdepth'] = decoder.field('next_stop_depth')
            dive_log.sample_data['stoptime'] = decoder.field('next_stop_time')
            dive_log.sample_data['cns'] = decoder.field('cns_load')
            dive_log.sample_data['ndl'] = decoder.field('ndl_time')
            dive_log.sample_data['tts'] = decoder.field('time_to_surface')
            dive_log.sample_data['time'] = timestamps_diff(dive_log.activity_start_timestamp,
                                                           decoder.field('timestamp', raw=True))  # Sets sample offset
            dive_log.sample_data['temp'] = decoder.field('temperature')
            # There is no minimal temperature field in FIT - need to find it
            # TODO: This is stupid - have to check what if air temp is lower than water
            temp = float(decoder.field('temperature', noconv=True))
            if dive_log.lowest_temp is None or dive_log.lowest_temp < temp:
                dive_log.lowest_temp = temp

            dive_log.add_sample()  # Creates sample record

            # -- Sample Fields -- #
            # absolute_pressure: 117600[Pa]
            # altitude: 3981
            # cns_load: 0[percent]
            # depth: 1.499[m]
            # distance: 0.0[m]
            # enhanced_altitude: 296.20000000000005[m]
            # heart_rate: 76[bpm]
            # n2_load: 3[percent]
            # ndl_time: None[s]
            # next_stop_depth: 0.0[m]
            # next_stop_time: 0[s]
            # temperature: 17[C]
            # time_to_surface: 9[s]
            # timestamp: 2018-04-11 08:56:59

    dive_log.processed_count += 1  # Increase processed_count property - to be used by calling function
    return True


def start_processing():
    """Starts selected tasks"""

    # Eventually set circles to default values using global constants
    if settings.big_circle is None:
        settings.big_circle = NEW_SITE_DISTANCE_MIN
    if settings.small_circle is None:
        settings.small_circle = OLD_COORDS_DISTANCE_MAX

    # ---------------------------------------------------------------------------
    # Dump FIT files to text (for reverse engineering) --------------------------
    # ---------------------------------------------------------------------------
    if settings.dump_dir is not None:
        print('Dumping FIT files content...\n')
        success_count = 0
        for fit_file in settings.fit_files:
            print('\rFile: {}\n      '.format(fit_file), end='')
            dump_file = os.path.join(settings.dump_dir, os.path.splitext(os.path.basename(fit_file))[0]) + '.txt'
            if fit_dump(fit_file, dump_file):
                success_count += 1
        print("\nSucessfully dumped {} of {} FIT files".format(success_count, len(settings.fit_files)))
        exit(0)  # Nothing more to do when dumping

    # ---------------------------------------------------------------------------
    # Or lists summaries of selected FIT files ----------------------------------
    # ---------------------------------------------------------------------------
    if settings.list_only:
        npprint('FIT files summary...\n')
        npprint('{:_^23.23}|{:_^19.19}|{:_^19.19}|{:_^8.8}|{:_^8.8}'.format('File', 'Date', 'Activity',
                                                                            'Time', 'Depth'))
        success_count = 0
        for fit_file in settings.fit_files:
            summary = fit_summary(fit_file)  # Gets tuple of (status=bool, dict=content)
            if summary[0]:
                success_count += 1
            content = (summary[1]['file'], summary[1]['time'], summary[1]['sport'],
                       summary[1]['elapsed'], summary[1]['depth'])
            if settings.pipe_output:
                print('|'.join(content))
            else:
                elapsed = content[3]
                depth = content[4]
                if depth:
                    depth = depth + ' m'
                if elapsed:
                    elapsed = str(int(round(float(content[3]))))
                    elapsed = str(datetime.timedelta(seconds=int(elapsed))).lstrip('0:')

                print('{:23.23}|{:19.19}|{:19.19}|{:>8.8}|{:>8.8}'.format(content[0], content[1],
                                                                          content[2], elapsed, depth))

        npprint("\nSucessfully listed {} of {} FIT files".format(success_count, len(settings.fit_files)))
        exit(0)  # Nothing more to do when listing

    # ---------------------------------------------------------------------------
    # Or Subsurface logs processing ---------------------------------------------
    # ---------------------------------------------------------------------------
    npprint('Import of {} files started...\n'.format(len(settings.fit_files)))

    start_time = datetime.datetime.now()  # Startup timestamp
    dive_log = DiveLog()  # Subsurface log modify object

    success_count = 0

    for fit_file in settings.fit_files:
        npprint('\rFile: {}\n      '.format(fit_file), end='')

        if message_processor(dive_log, fit_file):  # Start sequential processing FIT files (True if OK)
            success_count += 1

        dive_log.reset_all_vars()  # Clears some variables of dive_log object preparing it for next file

    npprint("\nSucessfully processed {} of {} FIT files".format(success_count, len(settings.fit_files)))

    # Everything finished - now let's save Subsurface log back to the disk
    if dive_log.processed_count > 0:
        if not settings.pipe_output:
            dive_log.save_log(settings.out_subslog)
        else:
            dive_log.pipe_log()

    elapsed = datetime.datetime.now() - start_time  # Processing time
    npprint('\nImport done in {}.{} seconds'.format(elapsed.seconds, elapsed.microseconds))

    # All done, good bye...


# =================================================================================
# First function which is called only when script starts as standalone application
# =================================================================================
def main():

    # Help content for argparse epilog
    footer = "examples:\n" \
             " {0} -f 2018-04-03-12-07-44.fit -d dump-files\n\n"\
             "    Dumps content of file '2018-04-03-12-07-44.fit' to 'dump-files' directory\n\n" \
             " {0} -f myfits/*.fit -d\n\n"\
             "    Dumps content of all FIT files from 'myfits' directory to current directory\n\n" \
             " {0} -f 2018-04-03-12-07-44.fit -i mylog.ssrf -o new.xml -c 600 80 -a -t 1.5\n\n"\
             "    Appends dive from file '2018-04-03-12-07-44.fit' to 'mylog.ssrf' and saves\n" \
             "    result to 'new.xml'. Apnea dives are also added. Dives shorter than 90s (1.5min)\n" \
             "    will be discarded. New dive site will be created if distance to the nearest site\n" \
             "    found in log is longer than 600m. Site found in log will be reused if distance\n" \
             "    is shorter than 80m. If distance is between 80m and 600m, new site with new GPS\n" \
             "    coordinates will be created but found name of adjacent site will be reused.\n\n" \
             " {0} -f 2018-04-03-12-07-44.fit -o newlog.xml -n\n\n" \
             "    Creates completely new Subsurface log from FIT files content. Dive numbers won't\n" \
             "    be created. Subsurface 'import dive log' function will create missing numbers\n" \
             "    automatically.\n\n" \
             " {0} -f myfits/*.fit -l\n\n" \
             "    Lists FIT files summaries in tabular format.\n\n" \
             " {0} -f myfits/*.fit -l -p\n\n" \
             "    Lists FIT files summaries to stdout in pipe-character separated format, \n" \
             "    convenient for redirecting output to " \
             "(eventual) external GUI file picker.\n\n".format(os.path.basename(sys.argv[0])).replace('/', os.sep)

    cmdparser = argparse.ArgumentParser(epilog=footer, formatter_class=argparse.RawDescriptionHelpFormatter)

    cmdparser.add_argument('-f', '--fitfiles', metavar='FIT_SOURCE', required=True, nargs='+',
                           help='Source FIT file(s) list. Shell wildcards can be used.')

    cmdparser.add_argument('-o', '--outlog', metavar='OUTPUT_LOG', required=False,
                           help='Destination Subsurface log file. Can be the same as input log.')

    cmdparser.add_argument('-i', '--inlog', metavar='INPUT_LOG', required=False,
                           help='Source Subsurface log file. If not specified, output log will be created '
                                'from scratch.')

    cmdparser.add_argument('-d', '--dump', metavar='DUMP_DIR', required=False, nargs='?', const='.',
                           help='Dump FIT files content to text files. Optional destination directory can '
                                'be specified. If not - dump files will be created in current directory.')

    cmdparser.add_argument('-c', '--circles', metavar=('BIG_CIRCLE', 'SMALL_CIRCLE'), required=False,
                           nargs=2, type=int, help='Set custom radii in meters of circles used for existing '
                                                   'dive site reuse detection.')

    cmdparser.add_argument('-a', '--apnea', required=False, action='store_false',
                           help='Apnea dives will be also processed.')

    cmdparser.add_argument('-t', '--timelimit', metavar='MINUTES', required=False, type=float,
                           help='Discard dives shorter than this value. Fractional values can be used. '
                                'Does not affect apnea dives.')

    cmdparser.add_argument('-l', '--listdives', required=False, action='store_true',
                           help='Print only summary of chosen FIT files.')

    cmdparser.add_argument('-p', '--pipeoutput', required=False, action='store_true',
                           help='Dump data to stdout and suppress all processing messages.')

    cmdparser.add_argument('-n', '--nonumbering', required=False, action='store_true',
                           help='Dive numbering won\'t be set when output log is created from scratch. '
                                'Allows autonumbering in further joining output file with existing log '
                                'using Subsurface "import log" feature.')

    if '-p' not in sys.argv or '--pipeout' not in sys.argv:  # Suppress script description
        print('Garmin Descent MK1 data converter for Subsurface Dive Log (FIT to XML)\n')

    if len(sys.argv) < 2:  # To print help even if script is called without parameters
        cmdparser.print_help()
        sys.exit(1)

    settings.settings_from_args(cmdparser)  # Initialize settings container with data from argparser
    settings.check_settings()  # Check parameters cosistency

    start_processing()  # Start processing FIT files with parameters set above


# =======================================================================
settings = Settings()  # Global object for settings storage and checking
# =======================================================================

# If run as standalone application
if __name__ == '__main__':
    main()

# Steps required to use script as module:
#  1. Set 'settings' global object properties (look what's needed at 'settings.settings_from_args()' method)
#  2. Call 'settings.check_settings()' (optional)
#  3. Call 'start_processing()' function
