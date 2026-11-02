#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:31:18 2025

@author: daniellelindsay
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import geopy.distance
import pandas as pd
import pygmt as pygmt 
from scipy import stats
from mintpy.utils.utils0 import azimuth2heading_angle, calc_azimuth_from_east_north_obs

# dist_min = 0.1
# dist_max = 0.5
# x_length = dist_max + 0.1
# x_width = 0.1
# min_points_threshold = 10 
# region = "HW1"

########################################
# File Paths 
########################################

# fault_name = "Hayward"  # Change this as needed
# fault_sections = ["HW1", "HW2", "HW3", "HW4", "HW5", "HW6"]  # Sections to process

# base_dir = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output"
# geometry_dir = "/Volumes/X10Pro/OPERA/11115_Hayward/geometry"

# fig_dir = "/Volumes/X10Pro/OPERA/Figures"
# output_dir = "/Volumes/X10Pro/OPERA/Scripts"


# velocity_file = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output/velocity_HW3_msk.h5"
# ts_file = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output/timeseries_HW3_msk.h5"
# #fault_file = "/Volumes/X10Pro/OPERA/qgis_data/CF_HF_1.gmt"
# fault_profiles = "/Volumes/X10Pro/OPERA/Scripts/CF_HF_cross_profiles_sorted.csv"

# los_north_file = "/Volumes/X10Pro/OPERA/11115_Hayward/geometry/los_north_HW2.h5"
# los_east_file = "/Volumes/X10Pro/OPERA/11115_Hayward/geometry/los_east_HW1.h5"

# HW1_dic = {"ORBIT_DIRECTION" : "descending", "Path" : "11115",
#               "Heading" : -169.87667846679688, "Platform" : "OPERA", 
#               region : "HW1", 
#               velocity_file = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output/velocity_HW3_msk.h5"
#               ts_file = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output/timeseries_HW3_msk.h5",
#               east...
#               north...}

dist_min = 0.1
dist_max = 0.3
x_length = dist_max + 0.1
x_width = 0.2
min_points_threshold = 10 

fault_name = "Hayward"  # Change this as needed
fault_sections = ["HW1", "HW2", "HW3", "HW4", "HW5", "HW6"]  # Sections to process

base_dir = "/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output"
geometry_dir = "/Volumes/X10Pro/OPERA/11115_Hayward/geometry"
fig_dir = "/Volumes/X10Pro/OPERA/Figures"
output_dir = "/Volumes/X10Pro/OPERA/Scripts"
fault_profiles = "/Volumes/X10Pro/OPERA/Scripts/CF_HF_cross_profiles_sorted.csv"
# creep rate data
creep_rate_file='/Users/daniellelindsay/Google Drive/My Drive/Proj_1_MTJ_creeprates/data/SRL-2022186_Supplement_FINAL_2023_CREEP_Dataset.csv'

# Define the output CSV file path
xfault_csv_path = f"/Volumes/X10Pro/OPERA/11115_Hayward/mintpy_output/{fault_name}_fault_data.csv"

# Common properties
ORBIT_DIRECTION = "descending"
PATH = "11115"
HEADING = -169.87667846679688
PLATFORM = "OPERA"

# Individual dictionaries for each fault section
HW1_dic = {
    "region": "HW1",
    "velocity_file": f"{base_dir}/velocity_HW1_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW1_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW1.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW1.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

HW2_dic = {
    "region": "HW2",
    "velocity_file": f"{base_dir}/velocity_HW2_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW2_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW2.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW2.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

HW3_dic = {
    "region": "HW3",
    "velocity_file": f"{base_dir}/velocity_HW3_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW3_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW3.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW3.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

HW4_dic = {
    "region": "HW4",
    "velocity_file": f"{base_dir}/velocity_HW4_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW4_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW4.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW4.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

HW5_dic = {
    "region": "HW5",
    "velocity_file": f"{base_dir}/velocity_HW5_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW5_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW5.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW5.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

HW6_dic = {
    "region": "HW6",
    "velocity_file": f"{base_dir}/velocity_HW6_msk.h5",
    "ts_file": f"{base_dir}/timeseries_HW6_msk.h5",
    "los_north_file": f"{geometry_dir}/los_north_HW6.h5",
    "los_east_file": f"{geometry_dir}/los_east_HW6.h5",
    "ORBIT_DIRECTION": ORBIT_DIRECTION,
    "Path": PATH,
    "Heading": HEADING,
    "Platform": PLATFORM
}

# Create a list of dictionaries for iteration
insar_dic_list = [HW1_dic, HW2_dic, HW3_dic, HW4_dic, HW5_dic, HW6_dic]

########################################
# Define functions
########################################

# Function to convert date strings to decimal years
def date_to_decimal_year(date_str):
    """Convert date string (YYYYMMDD) to decimal year."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return year + (month - 1) / 12 + (day - 1) / 365.25

def get_azimuth_ang(dsDict):
    """Compute the azimuth angle from east/north coefficients"""
    east = dsDict["los_east"]
    north = dsDict["los_north"]
    azimuth_angle = calc_azimuth_from_east_north_obs(east, north)
    return azimuth_angle, east, north

# Function to read InSAR data and convert UTM to Lat/Lon
def read_insar_data(los_north_file, los_east_file, velocity_file, ts_file):
    """Reads InSAR data and converts UTM to Lat/Lon."""
    print("Loading InSAR data...")

    # # Read los_north data
    # with h5py.File(los_north_file, 'r') as hfn:
    #     print(f"Loading {los_north_file}")
    #     north = np.array(hfn["/band1"][:])   # Convert to mm/yr
        
    # # Read velocity data
    # with h5py.File(los_east_file, 'r') as hfvel:
    #     print(f"Loading {los_east_file}")
    #     east = np.array(hfn["/band1"][:]) 

    # Read velocity data
    with h5py.File(velocity_file, 'r') as hfvel:
        print(f"Loading {velocity_file}")
        velocity = np.array(hfvel["velocity"][:]) * 1000  # Convert to mm/yr
        velocity[velocity == 0] = np.nan  # Set zero values to nan
        x_start = float(hfvel.attrs['X_FIRST'])
        y_start = float(hfvel.attrs['Y_FIRST'])
        x_step = float(hfvel.attrs['X_STEP'])
        y_step = float(hfvel.attrs['Y_STEP'])
        length = int(hfvel.attrs['LENGTH'])
        width = int(hfvel.attrs['WIDTH'])

       

    # Read time series data
    with h5py.File(ts_file, 'r') as hfts:
        print(f"Loading {ts_file}")
        ts = np.array(hfts["timeseries"][:]) * 1000  # Convert to mm
        ts[ts == 0] = np.nan  # Set zero values to nan
        ts_dates_bytes = np.array(hfts["date"][:])

    # Convert dates to decimal years
    ts_dates = [date_to_decimal_year(d.decode('utf-8')) for d in ts_dates_bytes]

    # Generate longitude and latitude grids (UTM)
    utm_x = np.linspace(x_start, x_start + x_step * (width - 1), width)
    utm_y = np.linspace(y_start, y_start + y_step * (length - 1), length)
    utm_x, utm_y = np.meshgrid(utm_x, utm_y)
    
    # Convert UTM to Lat/Lon (EPSG:32610 -> EPSG:4326)
    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(utm_x, utm_y)
    
    # Compute the azimuth and incidence angles from east/north coefficients
    east = np.full_like(velocity, 0.65) # Set values fix when I know how to load east and north. 
    north = np.full_like(velocity, -0.11)  # Set values fix when I know how to load east and north. 
    azi = calc_azimuth_from_east_north_obs(east, north)

    up = np.sqrt(1 - east**2 - north**2)
    inc = np.rad2deg(np.arccos(up))

    # Return data as dictionary
    return {
        'lons': lons,
        'lats': lats,
        'inc': inc,
        'azi': azi,
        'vel': velocity,
        'ts': ts,
        'ts_dates': ts_dates
    }


def get_start_end_points(lat_ori, lon_ori, az, dist):
    start_lat, start_lon, start_z = geopy.distance.distance(kilometers=-dist).destination((lat_ori, lon_ori), bearing=az)
    end_lat, end_lon, end_z = geopy.distance.distance(kilometers=dist).destination((lat_ori, lon_ori), bearing=az)
    return start_lat, start_lon, end_lat, end_lon

def get_fault_pairs(points, dist_min, dist_max):
    w_subset = points [ (points.p > -dist_max) & (points.p < -dist_min )]
    e_subset = points [ (points.p > dist_min) & (points.p < dist_max )]
    return w_subset, e_subset


def los2fault_projection(observations, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Calculate the design matrix for converting LOS displacement to slope direction.
    Converts aspect from positive clockwise from north to positive anticlockwise from east.
    """
    # Convert aspect to match LOS azimuth reference
    aspect_deg = -aspect_deg + 90

    # LOS components
    los_E = np.sin(np.deg2rad(los_inc_angle_deg)) * np.sin(np.deg2rad(los_az_angle_deg)) * -1
    los_N = np.sin(np.deg2rad(los_inc_angle_deg)) * np.cos(np.deg2rad(los_az_angle_deg))
    los_U = np.cos(np.deg2rad(los_inc_angle_deg))
    
    # Downslope components
    fault_E = np.sin(np.deg2rad(slope_deg)) * np.sin(np.deg2rad(aspect_deg)) * -1
    fault_N = np.sin(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(aspect_deg))
    fault_U = np.cos(np.deg2rad(slope_deg))
    
    # Normalize vectors
    L = np.array([los_E, los_N, los_U])
    L = L / np.linalg.norm(L)

    F = np.array([fault_E, fault_N, fault_U])
    F = F / np.linalg.norm(F)

    # Compute design matrix (dot product)
    G = np.dot(L, F)
    
    fault_parallel = observations / G if G != 0 else np.nan
    
    return fault_parallel

    def get_azimuth_ang(dsDict):
        """Compute the azimuth angle from east/north coefficients"""
        east = dsDict["los_east"]
        north = dsDict["los_north"]
        azimuth_angle = calc_azimuth_from_east_north_obs(east, north)
        return azimuth_angle, east, north


########################################
# Load InSAR
########################################

# insar_data = read_insar_data(los_north_file, los_east_file, velocity_file, ts_file)

# insar_data["Path"] = des_dic["Path"]
# insar_data["Platform"] = des_dic["Platform"]

# insar_dic_list = [insar_data] #

########################################
# Load Creep Data
########################################


# Filter x_fault to keep only rows within the latitude range
x_fault =[]
x_fault = pd.read_csv(fault_profiles, sep=",", header=1, names = ['ft_lon','ft_lat', 'profile_az'])

#Load file
creep_rate_data = pd.read_csv(creep_rate_file, sep=",") # names = ['Fault',	'Longitude',	'Latitude',	'Location Description',	'Creep rate (mm/yr)',	'sig (mm/yr)',	'Type',	'Inst',	'Start',	'End',	'Years',	'Source',	'EQ Contamination',	'Temporal Variation Observed',	'Added or Updated from UCERF3',	'Used in 2021 Inversion'])
creep_rate_data = creep_rate_data.rename(columns={'Creep rate (mm/yr)': 'CreepRate', 'sig (mm/yr)': 'nSig'})
creep_rate_data.Longitude =  creep_rate_data.Longitude+360
creep_rate_data = creep_rate_data [creep_rate_data["Fault"] == "Hayward"]
creep_rate_data["Yerror"] = np.nan

source_mapping = {
    1: "Brown and Wallace (1968)",
    2: "Burford (1988)",
    3: "Burford and Harsh (1980)",
    4: "CRIES",
    5: "Galehouse and Lienkaemper (2003)",
    6: "Jin and Funning (2017)",
    7: "Lienkaemper et al. (1991)",
    8: "Lienkaemper et al. (2001)",
    9: "Lindsey et al. (2014)",
    10: "Lindsey et al. (2016)",
    11: "Lisowski and Prescott (1981)",
    12: "Louie et al.(1985)",
    13: "McFarland et al. (2009)",
    14: "McFarland et al. (2019)",
    15: "Oppenheimer et al. (1990)",
    16: "Prescott et al. (1981)",
    17: "Langbein (2021)",
    18: "Xu (2020)",
    19: "Schulz et al. (1982)",
    20: "Scott et al. (2020)",
    21: "Titus et al. (2005)",
    22: "Titus et al. (2006)",
    23: "Tong et al. (2013)",
    24: "Wallace and Roth (1967)",
    25: "Wilmesher and Baker (1987)",
    26: "Xu et al. (2018)"
}

creep_rate_data['Source'] = creep_rate_data['Source'].map(source_mapping)
print(creep_rate_data[['Source']].head())


########################################
# Calculate Creep Rate
########################################

plot_data_dict = {}

# Iterate over the list of dictionaries
for dic in insar_dic_list:
    print("")
    print(f"Calculating ts {dic['Path']}")

    print("")
    print(f"Loading InSAR data for {dic['region']} (Path: {dic['Path']})")

    # Load InSAR data for the current fault section
    insar_data = read_insar_data(
        dic["los_north_file"],
        dic["los_east_file"],
        dic["velocity_file"],
        dic["ts_file"]
    )

    # Add metadata to the loaded dictionary
    insar_data["Path"] = dic["Path"]
    insar_data["Platform"] = dic["Platform"]
    insar_data["region"] = dic["region"]

    # Extract relevant variables from the dictionary
    lons = insar_data["lons"]
    lats = insar_data["lats"]
    vel = insar_data["vel"]
    ts = insar_data["ts"]
    inc = insar_data["inc"]
    azi = insar_data["azi"]
    ts_dates = insar_data["ts_dates"]
    path = insar_data["Path"]
    segment = insar_data["region"]
    
    dic["vel"] = vel
    dic["lons"] = lons
    dic["lats"] = lats
    
    # Cut x_fault to just data
    lat_min = insar_data["lats"].min()
    lat_max = insar_data["lats"].max()
    print(lat_min, lat_max)

    # Initialize an empty DataFrame for this path
    plot_data = pd.DataFrame(columns=['latitude', 'time', 'west_east_ts'])

    # Flatten data
    original_index = np.arange(lons.size).reshape(
        lons.shape)  # 2D array tracking original indices
    xyz_dataframe = pd.DataFrame({
        'x': lons.ravel(),
        'y': lats.ravel(),
        'z': original_index.ravel()  # Use index as z
    })

    # Drop rows with NaN values
    xyz_dataframe = xyz_dataframe.dropna()

    # Use project to extract values along the line
    for index, row in x_fault.iterrows():
        # Skip fault locations outside InSAR data bounds
        if not (lat_min <= row['ft_lat'] <= lat_max):
            continue  # Skip this iteration and move to the next fault location

        print(f"Getting points... {row['ft_lon']}, {row['ft_lat']}")

        try:
            # Attempt to project points
            points = pygmt.project(
                data=xyz_dataframe,
                center=[row['ft_lon'], row['ft_lat']],
                length=[-x_length, x_length],
                width=[-x_width, x_width],
                azimuth=row['profile_az'],
                unit=True
            )
        except Exception as e:
            print(f"Error projecting points for {
                  row['ft_lon']}, {row['ft_lat']}: {e}")
            continue  # Skip to the next fault

        # Check if points DataFrame is empty or below the threshold
        if points.empty or len(points) < min_points_threshold:
            print(f"Insufficient points ({len(points)}) found for fault location {
                  row['ft_lon']}, {row['ft_lat']}. Skipping...")
            continue
 
        points.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
 
        # Get lat and lon of fault pairs for each step
        print(f"Getting fault pairs... {row['ft_lon']}, {row['ft_lat']}")
        w_subset, e_subset = get_fault_pairs(points, dist_min, dist_max)

        # Ensure subsets meet the minimum threshold
        if w_subset.empty or e_subset.empty or len(w_subset) < min_points_threshold or len(e_subset) < min_points_threshold:
            print(f"Insufficient valid fault pairs found for {row['ft_lon']}, {row['ft_lat']}. "
                  f"West subset: {len(w_subset)} points, East subset: {len(e_subset)} points. Skipping...")
            continue

        # Find 2D indices for projected points
        projected_indices_w_original_index = []
        for idx in w_subset['z'].astype(int).to_numpy():
            diff = np.abs(original_index - idx)
            if np.isnan(diff).all():  # Skip if all differences are NaN
                continue
            min_index = np.unravel_index(np.argmin(diff), original_index.shape)
            projected_indices_w_original_index.append(min_index)

        projected_indices_e_original_index = []
        for idx in e_subset['z'].astype(int).to_numpy():
            diff = np.abs(original_index - idx)
            if np.isnan(diff).all():  # Skip if all differences are NaN
                continue
            min_index = np.unravel_index(np.argmin(diff), original_index.shape)
            projected_indices_e_original_index.append(min_index)

        # Handle empty projected indices
        if not projected_indices_w_original_index or not projected_indices_e_original_index:
            print(
                "No valid projected indices for west or east points. Skipping fault segment...")
            continue

        # Separate into x and y indices
        try:
            row_indices_w, col_indices_w = zip(
                *projected_indices_w_original_index)
            row_indices_e, col_indices_e = zip(
                *projected_indices_e_original_index)
        except ValueError as e:
            print(f"Error unpacking indices: {
                  e}. Skipping this fault segment...")
            continue

        # Extract time series for the projected points
        try:
            # Extract time series for west points
            projected_ts_w = ts[:, row_indices_w, col_indices_w]
            # Extract time series for east points
            projected_ts_e = ts[:, row_indices_e, col_indices_e]
        except IndexError as e:
            print(f"Indexing error for time series: {e}")
            continue  # Skip this iteration if indices are invalid

        # Check for NaN values
        if np.all(np.isnan(projected_ts_w)):
            print("All values in projected_ts_w are NaN. Skipping this fault segment.")
            continue

        if np.all(np.isnan(projected_ts_e)):
            print("All values in projected_ts_e are NaN. Skipping this fault segment.")
            continue

        # Calculate median displacement for each date (along the spatial axis)
        try:
            # Median along spatial axis
            w_avg = np.nanmedian(projected_ts_w, axis=1)
            # Median along spatial axis
            e_avg = np.nanmedian(projected_ts_e, axis=1)
        except RuntimeWarning:
            print(f"Warning: NaN encountered in projected time series for fault segment {
                  row['ft_lon']}, {row['ft_lat']}")
            continue

        # Recalculate west-east time series
        west_east_ts = e_avg - w_avg

        # Validate lengths
        if len(west_east_ts) != len(ts_dates):
            print(f"Mismatch: west_east_ts length ({
                  len(west_east_ts)}) does not match ts_dates ({len(ts_dates)})")
            continue

        # Calculate mean incidence and azimuth angles
        mean_inc_w = np.nanmean(inc[row_indices_w, col_indices_w])
        mean_inc_e = np.nanmean(inc[row_indices_e, col_indices_e])
        mean_azi_w = np.nanmean(azi[row_indices_w, col_indices_w])
        mean_azi_e = np.nanmean(azi[row_indices_e, col_indices_e])

        slope = 90  # horizontal, measured from vertical
        aspect = row['profile_az'] + 90

        # Calculate the design matrix for the provided variables
        fault_parallel_w = los2fault_projection(
            w_avg, mean_inc_w, mean_azi_w, slope, aspect)
        fault_parallel_e = los2fault_projection(
            e_avg, mean_inc_e, mean_azi_e, slope, aspect)

        # Calculate fault-parallel displacement across the fault
        west_east_fault_parallel_ts = fault_parallel_e - fault_parallel_w

        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'x': ts_dates,
            'west_east_ts': west_east_ts,
            'west_east_fault_parallel_ts': west_east_fault_parallel_ts,
            'west': w_avg,
            'east': e_avg,
            'fault_parallel_w': fault_parallel_w,
            'fault_parallel_e': fault_parallel_e
        })
        df = df.dropna()

        # Compute the linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["x"], df["west_east_fault_parallel_ts"]
        )
        
        creep_rate = slope
        x_fault.loc[index, f'{segment}_fault_parallel_creep_rate'] = creep_rate
        x_fault.loc[index, f'{segment}_fault_parallel_std_err'] = std_err

        # Append to plot data
        latitudes = np.full_like(west_east_fault_parallel_ts, row['ft_lat'])
        temp_df = pd.DataFrame({
            'latitude': latitudes,
            'time': ts_dates,
            'west_east_ts': west_east_ts,
            'west_east_fault_parallel_ts': west_east_fault_parallel_ts,
        })
        plot_data = pd.concat([plot_data, temp_df], ignore_index=True)
    
    # Pivot the DataFrame
    plot_data_pivot = plot_data.pivot(
        index='latitude', columns='time', values='west_east_fault_parallel_ts')
    
    dic["plot_data"] = plot_data_pivot

x_fault.to_csv(xfault_csv_path, index=False)

# Iterate over the list of dictionaries
for dic in insar_dic_list:
    # Extract relevant variables from the dictionary
    path = dic["Path"]
    segment = dic["region"]
    vel = dic["vel"]
    lons = dic["lons"]
    lats = dic["lats"]
    
    
    plot_data_flat = dic["plot_data"].reset_index().melt(
        id_vars='latitude', var_name='time', value_name='west_east_fault_parallel_ts'
    ).dropna()
    
    # # Convert 'time' to numeric if necessary
    # plot_data_flat['time'] = pd.to_numeric(plot_data_flat['time'], errors='coerce')
    
    # # Find the middle time
    # unique_times = plot_data_flat['time'].unique()
    # middle_time = unique_times[len(unique_times) // 2]  # Pick the middle time value
    
    # # Find the corresponding value at the middle time
    # middle_values = plot_data_flat.loc[plot_data_flat['time'] == middle_time, 'west_east_fault_parallel_ts']
    # if not middle_values.empty:
    #     middle_value = middle_values.mean()  # Use mean if there are multiple entries
    # else:
    #     middle_value = 0  # Fallback if no value found for middle time
    
    # # Normalize the time-series data to make the middle time zero
    # plot_data_flat['west_east_fault_parallel_ts'] -= middle_value

    # Update font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 20
    })

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1, 1.5]})

    # First subplot (scatter plot for fault-parallel displacement)
    sc = axes[0].scatter(
        plot_data_flat['time'],
        plot_data_flat['latitude'],
        c=plot_data_flat['west_east_fault_parallel_ts'],  # Change to 'west_east_fault_parallel_ts' if plotting fault-parallel
        cmap='plasma',
        vmin=-35,
        vmax=35,
        s=8,
        marker='s',
        edgecolor='none'
    )
    axes[0].set_title(f'{dic["region"]} Relative Fault Parallel Displacement - {dic["Platform"]} {path}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Latitude')
    axes[0].set_xlim(2015.0, 2025.0)
    axes[0].set_ylim(dic["lats"].min(), dic["lats"].max())
    fig.colorbar(sc, ax=axes[0], label='Relative Cumulative Displacement (mm)')
    
    # Define the window size and minimum number of valid points
    window_size = 20  # Adjust as needed
    min_valid_points = 5  # Minimum number of non-NaN values required in the window
    
    # Compute the moving average, allowing for NaNs
    x_fault[f'{segment}_fault_parallel_creep_rate_moving_avg'] = x_fault[
        f'{segment}_fault_parallel_creep_rate'
    ].rolling(window=window_size, center=True, min_periods=min_valid_points).mean()

    axes[1].plot(
    x_fault[f'{segment}_fault_parallel_creep_rate_moving_avg'],
    x_fault['ft_lat'],
    color='darkred',  # Static color
    linewidth=2  # Thickness of the line
    )
    
    axes[1].scatter(
    x_fault[f'{segment}_fault_parallel_creep_rate'],
    x_fault['ft_lat'],
    color='grey',  # Static color
    marker='.'  # Thickness of the line
    )
    
    axes[1].set_title('Fault-Parallel Creep Rate')
    axes[1].set_xlabel('Creep Rate (mm/yr)')
    axes[1].set_ylim(dic["lats"].min(), dic["lats"].max())
    axes[1].set_xlim(-15, 15)

    # Add vertical line at zero for creep rates
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Add creep rate data with distinct markers
    markers = ['s', '^', 'o']  # Square, Triangle, Circle
    for source, marker in zip(creep_rate_data['Source'].unique(), markers):
        subset = creep_rate_data[creep_rate_data['Source'] == source]
        axes[1].scatter(
            subset['CreepRate'], subset['Latitude'],
            color='black', s=40, marker=marker, label=f'{source}', edgecolor='none'
        )
        
        # Add legend for the scatter points on the right subplot
    axes[1].legend(fontsize=10, loc='lower center', bbox_to_anchor=(0.5, -0.3))
        
    ###### **Third subplot: Velocity Map** ######
    
    vel = vel - np.nanmean(vel)

    # Velocity map with fault trace
    img = axes[2].pcolormesh(
        lons, lats, vel, cmap="RdBu_r", shading="auto", vmin=-5, vmax=5,
    )
    axes[2].plot(x_fault['ft_lon'], x_fault['ft_lat'], color="black", linewidth=1.5)

    # Aspect ratio correction to avoid distortion
    #axes[2].set_aspect(1.5)  # Adjust aspect ratio to ensure lat/lon scaling
    axes[2].set_xlim(dic["lons"].min(), dic["lons"].max())
    axes[2].set_ylim(dic["lats"].min(), dic["lats"].max())
    axes[2].set_title("Velocity Map")
    axes[2].set_xlabel("Longitude")
    fig.colorbar(img, ax=axes[2], label="Velocity (mm/yr)")

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{dic["region"]}_{path}_west_east_fault_parallel_x_width{x_width}_dist_min{dist_min}_dist_max{dist_max}_minpoints{min_points_threshold}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Iterate over the list of dictionaries
for dic in insar_dic_list:
    # Extract relevant variables from the dictionary
    path = dic["Path"]
    segment = dic["region"]
    vel = dic["vel"]
    lons = dic["lons"]
    lats = dic["lats"]
    
    # Detrend the data before plotting
    plot_data_pivot = dic["plot_data"]
    
    # Ensure time is numeric
    plot_data_pivot.columns = pd.to_numeric(plot_data_pivot.columns, errors='coerce')

    # Apply detrending row-wise (each latitude)
    def detrend_row(row):
        valid_mask = ~row.isna()  # Mask to ignore NaNs
        if valid_mask.sum() > 1:  # Ensure at least two points for regression
            x = row.index[valid_mask].astype(float)  # Convert time to numeric
            y = row[valid_mask]

            # Compute linear trend
            slope, intercept, _, _, _ = stats.linregress(x, y)

            # Remove trend
            detrended = y - (slope * x + intercept)

            # Replace the original values in the row
            row[valid_mask] = detrended
        
        return row

    # Apply function row-wise to remove long-term trends
    plot_data_pivot_detrended = plot_data_pivot.apply(detrend_row, axis=1)

    # Flatten the detrended DataFrame for plotting
    plot_data_flat = plot_data_pivot_detrended.reset_index().melt(
        id_vars='latitude', var_name='time', value_name='west_east_fault_parallel_ts'
    ).dropna()

    # Update font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 20
    })

    # **Single Figure**
    plt.figure(figsize=(9, 5.5))

    # Scatter plot for detrended fault-parallel displacement
    sc = plt.scatter(
        plot_data_flat['time'],
        plot_data_flat['latitude'],
        c=plot_data_flat['west_east_fault_parallel_ts'],  # Using detrended data
        cmap='RdBu',
        vmin=-13,
        vmax=13,
        s=9,
        marker='s',
        edgecolor='none'
    )
    plt.title(f'{dic["region"]} Fault Parallel Displacement (Detrended) - {dic["Platform"]} {path}')
    plt.xlabel('Time')
    plt.ylabel('Latitude')
    plt.xlim(2015.0, 2025.0)
    plt.ylim(dic["lats"].min(), dic["lats"].max())

    # Add colorbar
    plt.colorbar(sc, label='Detrended Displacement (mm)')

    # Save figure
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{dic["region"]}_{path}_detrended_fault_parallel.png', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
