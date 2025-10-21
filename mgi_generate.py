"""

Generates Mg II index from SUIT data for a series of dates specified. 

Options:
- Generate Mg II index files (2kx2k) and the absolute error files for specified dates. 
- Integrate the Counts from Each SUIT filter files and compute the Mg II index for the given date. Stores Single value  (Sun as a star)
- Integration process: Identify sets > Integrate Each File > Generate Index > Take Mean for the day
- Error calculated as Std. Dev. of Mg II index for a given Map. 

Input:
- SUIT files and location.  'src_base'
- Output location. 'dst_base'
- Subfolder to be specified 'normal_4k'
- Dates : Stard and End date # The Files need to be in the SUIT POC Folder Format. yyyy/mm/dd/normal_4k/SUIT_FILE.fits

Output:
- Creates similar folder structure as input in 'dst_base' path.
- Creates Mg II index map for each date specified.
- Generates a TXT file with Mg II disk integrated values for each Timestamp set with errors and other relevant data.

Author: Atul Bhat @atulbhats

Last Modified: 20 August, 2025 



"""

# Debug Options
debug = 0
break_one = 0 # Break after first iteration in For Loop for dates. 
one_set_only = 0 # Generates only First Map for the day. - For Testing
echo_op = 0 # Echo Output lines. 
show_math_error = 0

# Fitting Options
circle_fit = 1
preview_fit = 0

# Processing Options
parallel = 1 # Serial / Parallel Processing
leave_out_cores = 2 # Spare these many cores while parallel processing

# Binning & Alignment methods to be used
rebin = 1 # Rebin 2x2 to 1 : Resulting Image in 2k

"""
Alignment Options:
1 : Aign by Template
2 : Align by Relative Images
3 : No Alignment Required.
4 : Align by CRPIX [Best for Sun Centered]
5 : Align by First Image
"""
m = 4 # Alignment Methods

# Mg II image generation
generate_mg_images = 1 # Generate Mg II images for each set per day.
save_align_jpg = 0 # Saves the aligned results as jpg
check_aligned = 0  # Outputs the Aligned Files

# Mg II day Integration

integrate_mg_index = 1  
save_integrate_file = 1  # If 0, displays output on screen. 
integration_method = 0 # 0: Mean 1: Integrated(Simpson) 2: Integrated(Gaussian)
generate_only_for_new = 0 # Generates only for those dates where file is missing.

if integration_method == 0:
	integration_file_name = 'magnesium_index_compared_summed.txt'
	integration_name = 'Summed'
elif integration_method == 1: # Don't Use
	integration_file_name = 'magnesium_index_compared_simp.txt'
	integration_name = 'Integrated (Simpson)'
elif integration_method == 2: # Don't Use
	integration_file_name = 'magnesium_index_compared_quad.txt'
	integration_name = 'Integrated (Gaussian Quadrature)'


# Modules Required 
import re
import os
import sys
import glob
import socket
import requests
import platform
import sunpy.map
import time
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sunpy.net import Fido
from astropy.io import fits
from scipy.ndimage import shift
from scipy.integrate import simpson
from datetime import datetime,timedelta
from astropy.coordinates import SkyCoord
from sunkit_image.coalignment import mapsequence_coalign_by_match_template as mc_coalign
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import mgi_functions as mgi
import atul_settings as atul


# Required Variables. Don't Touch
filters =	['NB01','NB02','NB03','NB04','NB05','NB06','NB07','NB08','BB01','BB02','BB03']
label = ['214nm Continuum','276.7nm Continuum','279.6nm Mg II K','280.3nm Mg II h','283.2nm Continuum','300nm Continuum','388nm Continuum','388nm CN Band','396.85nm Ca II h','220nm Herzberg','277nm Harley','340nm Huggins']
go = 1
dstr = None
mgI_dat_arr = []
mgI_err_dat_arr = []
mgI_arr = []
mgI_err_arr = []
num_arr = []
den_arr = []
num_err_arr = []
den_err_arr = []
# Obtain Dates

# Option 1 : Through Argument. 
# Option 2 : Specified here.

# Collect Inputs 

if len(sys.argv) > 1:
	dstr = sys.argv[1]

if dstr is not None:
	date = dstr
elif dstr is None:
	date = '01-04-2025'
	dates = []
	start_date = datetime(2025, 3, 20)		 # ← Start date
	end_date = datetime(2025, 4, 23)
	if debug:
		end_date = start_date
	skip_dates = [] #['08-06-2025'] #'29-01-2025','22-02-2025','19-03-2025','21-03-2025','25-12-2025']
	date_list = []
	current_date = start_date

	while current_date <= end_date:
		date_list.append(current_date.strftime('%d-%m-%Y'))
		current_date += timedelta(days=1)

if len(date_list) > 1 and preview_fit == 1:
	print('Multiple Dates with Preview Enabled. Exiting')
	input = ('Press 1 to exit. 2 to continue withour previews. [1/2] : ')
	if int(input) == 2:
		preview_fit = 0

if str(date).find(",") > 1:
	date = date.replace("[","").replace("]","") # For Atul's Stupidity
	dates = str(date).split(",")
else:
	dates = date_list

mgi.debug = debug
mgi.preview_fit = preview_fit
mgi.echo_align = echo_op
iterations = 0


def tsnow():
	global dt
	return dt.now().strftime("%H:%M:%S")

# THE Function

def mgi_generate(date):

	# Initialize
	global go, generate_mg_images,integrate_mg_index,save_align_jpg,check_aligned,iterations
	mgI_arr = []
	mgI_err_arr = []
	num_arr = []
	den_arr = []
	num_err_arr = []
	den_err_arr = []
	day_mg_II = []
	timestamp = datetime.strptime(date, "%d-%m-%Y")				
	print()
	print(f'Calculating Mg II Index for {date}:')
	
	# Extract Date Components
	if date.find('-') > -1:
		datestr = date.split("-")
	elif date.find('/') > -1:
		datestr = date.split("/")
	else:
		datestr = date.split(".")

	if len(datestr[0]) == 2:
		dd = datestr[0]
		mm = datestr[1]
		yyyy = datestr[2]
	elif len(datestr[0] == 4):
		yyyy = datestr[0]
		mm = datestr[1]
		dd = datestr[2]
	else:
		print(f'Date Error')
		go = 0

	# Folders
	src_base = '/atul_ssd/data/SUIT/data/'
	dst_base = '/atul_ssd/data/SUIT/mgi/'
	subfolder = 'normal_4k'

	
	if generate_only_for_new == 1 :
		if os.path.exists(os.path.join(dst_base,yyyy,mm,dd,'normal_4k',integration_file_name)):
			go = 0
			print(f'Skipping {date} Integrated already.')
		else:
			go = 1

		

	filt_array = ['nb4_f','nb3_f','nb2_f','nb5_f']

		# Check if Date Valid and Folder Structure Exists. If not Create

	if go == 1:
		dat	= f"Magnesium Index Files for {yyyy}-{mm}-{dd}: \n"
		dat += "Generated using SUIT Data \n"
		dat += "-----------------------------------------------\n"
		input_folder = os.path.join(src_base,yyyy,mm,dd,subfolder)
		output_folder = os.path.join(dst_base,yyyy,mm,dd,subfolder)
		
		# Create Date Folders to store files generated. 

		if os.path.exists(input_folder):
			if echo_op == 1:
				print('Date Valid. Checking Files')
			files = os.listdir(input_folder)
			if len(files) > 0:
				if not os.path.exists(os.path.join(dst_base,yyyy)):
					os.mkdir(os.path.join(dst_base,yyyy))
					os.mkdir(os.path.join(dst_base,yyyy,mm))
					os.mkdir(os.path.join(dst_base,yyyy,mm,dd))
				elif not os.path.exists(os.path.join(dst_base,yyyy,mm)):
					os.mkdir(os.path.join(dst_base,yyyy,mm))
					os.mkdir(os.path.join(dst_base,yyyy,mm,dd))
				elif not os.path.exists(os.path.join(dst_base,yyyy,mm,dd)):
					os.mkdir(os.path.join(dst_base,yyyy,mm,dd))

				if not os.path.exists(os.path.join(dst_base,yyyy,mm,dd,subfolder)):
					os.mkdir(os.path.join(dst_base,yyyy,mm,dd,subfolder))
				
				if echo_op == 1:				
					print(f'Not Empty: Files found within {yyyy}/{mm}/{dd}/{subfolder}')

				f_set = [] #Array with 4 filters for a given window. 
				files.sort()
				z = 0

				# initialize data and header variables

				nb4_f, nb3_f, nb2_f, nb5_f = None,None,None,None
				f_window, nb4_ts = None,None
				nb4_h, nb3_h, nb2_h, nb5_h = None,None,None,None


				"""
					Here we Find the Mg II index for a Filter Cycle. 
				"""
				if generate_mg_images == 1:

					nb4_f, nb3_f, nb2_f, nb5_f, f_window, nb4_ts = None,None,None,None,None,None # Initial Initializations
					exposure_corrected = 0
					
					print(f'Initiating Mg II index Images Generation for {date}')
					
					for f in files:

						# Check for Scatter Corrections
						if f.endswith('.fits'):
							with fits.open(os.path.join(input_folder,f)) as h:
								fheader = h[0].header
							if str(fheader['SCAT_CF']) == 'NA':
								print(f'ALERT! : {f} has not been Scatter Corrected. Please Look into it.')

						# Let's first find the Filter Set!  
						# Collect a Set
						if f.endswith('NB04.fits'):
							#SUT_T24_1411_000574_Lev1.0_2024-09-30T05.25.32.390_0971NB02.fits
							f_ts = f.split("_")[5]
							set_ts = f_ts.split("T")
							nb4_f = os.path.join(input_folder,f)
							nb4_ts = datetime.strptime(f_ts, '%Y-%m-%dT%H.%M.%S.%f')
							# Now add Delta t to timestamp and find nb3
							f_window = nb4_ts + timedelta(hours=1)
						elif nb4_ts is not None and f_window is not None:
							start_dt = nb4_ts
							end_dt = f_window
							f_ts = datetime.strptime(f.split("_")[5],'%Y-%m-%dT%H.%M.%S.%f')
							if f.endswith('NB03.fits') and start_dt <= f_ts <= end_dt:
								nb3_f = os.path.join(input_folder,f)
							elif f.endswith('NB05.fits') and start_dt <= f_ts <= end_dt:
								nb5_f = os.path.join(input_folder,f)
							elif f.endswith('NB02.fits') and start_dt <= f_ts <= end_dt:
								nb2_f = os.path.join(input_folder,f)

						if nb4_f is not None and nb2_f is not None and nb3_f is not None and nb5_f is not None:

							# A set is Collected. Let's Generate the Index Now.
							print()
							iterations += 1
							if parallel :
								print(f'Set {z+1} Gathered Files. Aligning FITS. || {date}')
							else:
								print(f'Set {z+1} Gathered Files. Aligning FITS. | {date}')
							#print(f'Timestmap: {f_ts}')

							# rebin 2x2 into 1

							fpath = [nb4_f,nb3_f,nb2_f,nb5_f]
							#print(fpath)

							if rebin == 1:
								if echo_op == 1:
									print(f'Rebinning | {date}')
								with fits.open(nb4_f) as h:
									data = h[0].data
								original_shape = data.shape

								nb4_f,nb4_h = mgi.rebin_fits_2x2_sum(nb4_f)
								nb3_f,nb3_h = mgi.rebin_fits_2x2_sum(nb3_f)
								nb2_f,nb2_h = mgi.rebin_fits_2x2_sum(nb2_f)
								nb5_f,nb5_h = mgi.rebin_fits_2x2_sum(nb5_f)
								rebin_size = 2
								rebinned_shape = nb4_f.shape # Rebin Done

								files = [(nb4_f,nb4_h),(nb3_f,nb3_h),(nb2_f,nb2_h),(nb5_f,nb5_h)]
							else:
								files = fpath

							if circle_fit == 1:
								aligned_maps = mgi.align_filters_by_fitting(files,z,rebin_size)
							else: # For Final Version remove the code below. 
								if m == 1:
									aligned_maps = mgi.align_filters_by_template(files,z)			
								elif m ==2:
									aligned_maps = mgi.align_filters_by_relative(files,z)
								elif m == 3:
									aligned_maps = mgi.align_filters_centered(files)
								elif m == 4 :
									aligned_maps = mgi.align_filters_by_crpix(files,z)
								elif m == 5:
									aligned_maps = mgi.align_filters_by_first(files,z)

							# Retrive Alignment Results

							nb4_a = aligned_maps[0].data
							nb3_a = aligned_maps[1].data
							nb2_a = aligned_maps[2].data
							nb5_a = aligned_maps[3].data

							nb4_h = aligned_maps[0].meta
							nb3_h = aligned_maps[1].meta
							nb2_h = aligned_maps[2].meta
							nb5_h = aligned_maps[3].meta

							header = fits.Header() # Defining Header for Map
							
							for key, value in nb2_h.items(): # We store NB2 for map header
								if isinstance(value, (int, float, str)):
									header[str(key)[:8]] = value

							set_timestamp = nb4_h['T_OBS']
							
							# Sample Code to Check Alignment 
							
							filter_index = [4,3,2,5]
							
							if check_aligned == 1:
								print(f'Alingment Check Requested. Loading Image Differences.')
								def check_align(f1_a,f2_a):
									f1_d = (f1_a - np.min(f1_a))/( np.max(f1_a) - np.min(f1_a))
									f2_d = (f2_a - np.min(f2_a))/ (np.max(f2_a) - np.min(f2_a))
									sub = f1_d - f2_d
									plt.imshow(sub, cmap='gray', origin='lower')
									plt.colorbar()
									plt.show()

								check_align(nb4_a,nb3_a)
								check_align(nb4_a,nb2_a)
								check_align(nb4_a,nb5_a)

							if save_align_jpg == 1:
								print(f'Alingment Check Requested. Saving JPGs as the following.')
								i = 0
								jpg_paths = []
								for amap in aligned_maps:
									data = amap.data
									# Normalize the data for better visualization
									normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
									plt.imshow(normalized_data)
									plt.show()
									# Plot the data using matplotlib and save it as a JPG
									plt.imshow(normalized_data, cmap='gray', origin='lower')
									
									if i == 0: lev = [0.08]
									elif i == 1: lev = [0.24]
									elif i == 2: lev = [0.14]
									elif i == 0: lev = [0.14]
									plt.contour(normalized_data, levels=lev, colors='red', linewidths=0.5, origin='lower')
									plt.axis('off')	# Remove axis for a cleaner image
									jpg_f = fpath[i].replace(".fits",".jpg")
									jpg_f = jpg_f.replace("SUIT/data/","SUIT/mgi/")
									print(f'{i}.Saving as {jpg_f}')
									jpg_paths.append(jpg_f)
									plt.savefig(jpg_f, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
									plt.close()
									i+=1

								from PIL import Image, ImageDraw, ImageFont
								font_path = '/atul_ssd/dump/Open_Sans/static/OpenSans-Bold.ttf'
								font = ImageFont.truetype(font_path, size=56)
								frames = []
								labels = [4,3,2,5]
								# Save as GIF
								output_path = os.path.join(jpg_f.replace(jpg_f.split("/")[-1],""),f'{yyyy}-{mm}-{dd}_alignment_{m}.gif')
								for path, label in zip(jpg_paths, labels):
									img = Image.open(path).convert("RGB")
									draw = ImageDraw.Draw(img)
									position = (10, 10)
									draw.text(position, f'NB0{label}', font=font, fill=(255, 255, 255))
									frames.append(img)

								frames[0].save(
									output_path,
									save_all=True,
									append_images=frames[1:],
									duration=500,	 # duration between frames in milliseconds
									loop=0			# loop=0 means infinite loop
								)

								print(f"GIF saved to {output_path}")

							# Add Core to Remove Off Disk Values at this point!!!

							# Code Replace Negative Values with Zeros. 

							nb4_a = np.where(nb4_a < 0, 0, nb4_a)
							nb3_a = np.where(nb3_a < 0, 0, nb3_a)
							nb5_a = np.where(nb5_a < 0, 0, nb5_a)
							nb2_a = np.where(nb2_a < 0, 0, nb2_a)

							nb4_exp = float(nb4_h['MEAS_EXP'])/1000
							nb3_exp = float(nb5_h['MEAS_EXP'])/1000
							nb2_exp = float(nb2_h['MEAS_EXP'])/1000
							nb5_exp = float(nb5_h['MEAS_EXP'])/1000

							"""
							AMP_G_E = '3.04    '           / Amplifier gain of channel E                    
							AMP_G_F = '3.09    '           / Amplifier gain of channel F                    
							AMP_G_G = '3.01    '           / Amplifier gain of channel G                    
							AMP_G_H = '2.95    '           / Amplifier gain of channel H 
							"""

							nb4_gain = [nb4_h['AMP_G_E'], nb4_h['AMP_G_F'], nb4_h['AMP_G_G'], nb4_h['AMP_G_H']]
							nb4_gain = [float(x) for x in nb4_gain]
							g = np.average(nb4_gain)

							"""
							Error Calculation: 

							Each Filter: 

							σₙᵦᵢ² = σdₙᵢ² + σₚᵣₙᵤ²  

							σₙᵦᵢ² = (√(g × DNₙᵦᵢ))² + (DNₙᵦᵢ × g × 0.56%)²  

							σₙᵦᵢ² = DNₙᵦᵢ × 3 + K² × DNₙᵦᵢ²  

							σₙᵦᵢ² = DNₙᵦᵢ (√3 + K² × DNₙᵦᵢ)  

							"""

							nb4_a = nb4_a * g # Converting to Electron per Photon 
							nb3_a = nb3_a * g
							nb2_a = nb2_a * g
							nb5_a = nb5_a * g

							# Define Each Error Component for each Filter. 

							nb4_pois = np.sqrt(nb4_a) # Poissonian Error
							nb2_pois = np.sqrt(nb2_a)
							nb3_pois = np.sqrt(nb3_a)
							nb5_pois = np.sqrt(nb5_a)

							# Adding Scatter Residue to each filter error term
							# Scatter Block Begins

							# quadrant n = [center_x,center_y] [ width, height]
							# quadrant 1 = [3300,3100] 200x200
							# quadrant 2 = [850,3100] 200x200
							# quadrant 3 = [850, 990] 200x200
							# quadrant 4 = [3300, 990] 200x200

							box_1 = [1650,1550] # x,y
							box_2 = [425,1550]
							box_3 = [400,450]
							box_4 = [1650,450]
							w = 100 / 2
							h = 100 / 2 

							nb4_s1 = nb4_a[int(box_1[1]-h):int(box_1[1]+h),int(box_1[0]-w):int(box_1[0]+w)] # Pick Boxes in each quadrant
							nb4_s2 = nb4_a[int(box_2[1]-h):int(box_2[1]+h),int(box_2[0]-w):int(box_2[0]+w)]
							nb4_s3 = nb4_a[int(box_3[1]-h):int(box_3[1]+h),int(box_3[0]-w):int(box_3[0]+w)]
							nb4_s4 = nb4_a[int(box_4[1]-h):int(box_4[1]+h),int(box_4[0]-w):int(box_4[0]+w)]

							nb4_sd1 = np.std(nb4_s1)
							nb4_sd2 = np.std(nb4_s2)
							nb4_sd3 = np.std(nb4_s3)
							nb4_sd4 = np.std(nb4_s4)

							if 0: #preview_fit:
								import matplotlib.patches as patches
								fig, ax = plt.subplots()
								ax.imshow(nb4_a, origin='lower')
								print(f'Box1 at 1: x{box_1[1]},y{box_1[0]} , StdDev: {nb4_sd1}, sum: {np.max(nb4_s1)}, Size: {nb4_s1.shape}')
								print(f'Box1 at 2: x{box_2[1]},y{box_2[0]} , StdDev: {nb4_sd2}, sum: {np.max(nb4_s2)}, Size: {nb4_s2.shape}')
								print(f'Box1 at 3: x{box_3[1]},y{box_3[0]} , StdDev: {nb4_sd3}, sum: {np.max(nb4_s3)}, Size: {nb4_s3.shape}')
								print(f'Box1 at 4: x{box_4[1]},y{box_4[0]} , StdDev: {nb4_sd4}, sum: {np.max(nb4_s4)}, Size: {nb4_s4.shape}')
								rect1 = patches.Rectangle((box_1[0]-(w/2),box_1[1]-(h/2)), w*2, h*2, linewidth=1, edgecolor='r', facecolor='none')
								rect2 = patches.Rectangle((box_2[0]-(w/2),box_2[1]-(h/2)), w*2, h*2, linewidth=1, edgecolor='r', facecolor='none')
								rect3 = patches.Rectangle((box_3[0]-(w/2),box_3[1]-(h/2)), w*2, h*2, linewidth=1, edgecolor='r', facecolor='none')
								rect4 = patches.Rectangle((box_4[0]-(w/2),box_4[1]-(h/2)), w*2, h*2, linewidth=1, edgecolor='r', facecolor='none')
								ax.add_patch(rect1)
								ax.add_patch(rect2)
								ax.add_patch(rect3)
								ax.add_patch(rect4)
								plt.show()


							nb4_std = [nb4_sd1,nb4_sd2,nb4_sd3,nb4_sd4]
							nb4_std = np.array(nb4_std)
							nb4_err_scat = np.max(nb4_std) #**2)) # Mean of Std. Dev

							if debug:
								print(f'NB4 Std. Devs: {nb4_std}')
								print(f'NB4 Scatter Error = {nb4_err_scat}')


							# -- Do this for all filters 

							nb3_s1 = nb3_a[int(box_1[1]-h):int(box_1[1]+h),int(box_1[0]-w):int(box_1[0]+w)] # Pick Boxes in each quadrant
							nb3_s2 = nb3_a[int(box_2[1]-h):int(box_2[1]+h),int(box_2[0]-w):int(box_2[0]+w)]
							nb3_s3 = nb3_a[int(box_3[1]-h):int(box_3[1]+h),int(box_3[0]-w):int(box_3[0]+w)]
							nb3_s4 = nb3_a[int(box_4[1]-h):int(box_4[1]+h),int(box_4[0]-w):int(box_4[0]+w)]

							nb3_sd1 = np.std(nb3_s1)
							nb3_sd2 = np.std(nb3_s2)
							nb3_sd3 = np.std(nb3_s3)
							nb3_sd4 = np.std(nb3_s4)
							nb3_std = [nb3_sd1,nb3_sd2,nb3_sd3,nb3_sd4]
							nb3_std = np.array(nb3_std)
							nb3_err_scat = np.max(nb3_std) #**2)) 

							nb2_s1 = nb2_a[int(box_1[1]-h):int(box_1[1]+h),int(box_1[0]-w):int(box_1[0]+w)] # Pick Boxes in each quadrant
							nb2_s2 = nb2_a[int(box_2[1]-h):int(box_2[1]+h),int(box_2[0]-w):int(box_2[0]+w)]
							nb2_s3 = nb2_a[int(box_3[1]-h):int(box_3[1]+h),int(box_3[0]-w):int(box_3[0]+w)]
							nb2_s4 = nb2_a[int(box_4[1]-h):int(box_4[1]+h),int(box_4[0]-w):int(box_4[0]+w)]

							nb2_sd1 = np.std(nb2_s1)
							nb2_sd2 = np.std(nb2_s2)
							nb2_sd3 = np.std(nb2_s3)
							nb2_sd4 = np.std(nb2_s4)
							nb2_std = [nb2_sd1,nb2_sd2,nb2_sd3,nb2_sd4]
							nb2_std = np.array(nb2_std)
							nb2_err_scat = np.max(nb2_std) #**2)) 

							nb5_s1 = nb5_a[int(box_1[1]-h):int(box_1[1]+h),int(box_1[0]-w):int(box_1[0]+w)] # Pick Boxes in each quadrant
							nb5_s2 = nb5_a[int(box_2[1]-h):int(box_2[1]+h),int(box_2[0]-w):int(box_2[0]+w)]
							nb5_s3 = nb5_a[int(box_3[1]-h):int(box_3[1]+h),int(box_3[0]-w):int(box_3[0]+w)]
							nb5_s4 = nb5_a[int(box_4[1]-h):int(box_4[1]+h),int(box_4[0]-w):int(box_4[0]+w)]

							nb5_sd1 = np.std(nb5_s1)
							nb5_sd2 = np.std(nb5_s2)
							nb5_sd3 = np.std(nb5_s3)
							nb5_sd4 = np.std(nb5_s4)
							nb5_std = [nb5_sd1,nb5_sd2,nb5_sd3,nb5_sd4]
							nb5_std = np.array(nb5_std)
							nb5_err_scat = np.max(nb5_std) #**2)) 

							### Scatter Block Ends 

							# Correct for Varying Exposure times. SUIT Exposure times for each filter are different. So dividing by Measured Exposure makes the counts / data per unit time.

							if nb4_h['MEAS_EXP'] != nb5_h['MEAS_EXP']:
								exposure_corrected = 1

								nb4_a = np.array(nb4_a)/(nb4_exp)
								nb3_a = np.array(nb3_a)/(nb3_exp)
								nb2_a = np.array(nb2_a)/(nb2_exp)
								nb5_a = np.array(nb5_a)/(nb5_exp)
								
								nb4_pois = np.sqrt(nb4_pois)/(nb4_exp) # If Not used, remove
								nb2_pois = np.sqrt(nb2_pois)/(nb2_exp)
								nb5_pois = np.sqrt(nb5_pois)/(nb5_exp)
								nb3_pois = np.sqrt(nb3_pois)/(nb3_exp)

								nb4_err_scat = np.sqrt(nb4_err_scat)/(nb4_exp) # If Not used, remove
								nb2_err_scat = np.sqrt(nb2_err_scat)/(nb2_exp)
								nb5_err_scat = np.sqrt(nb5_err_scat)/(nb5_exp)
								nb3_err_scat = np.sqrt(nb3_err_scat)/(nb3_exp)

							else:
								nb4_exp = nb3_exp = nb2_exp = nb5_exp = 1
								exposure_corrected = 0

													
							nb4_err = nb4_pois**2 + nb4_err_scat**2 # Adding Scatter term to Error
							nb4_err = np.sqrt(nb4_err) # Sigma for each filter.

							nb3_err = nb3_pois**2 + nb3_err_scat**2 # Adding Scatter term to Error
							nb3_err = np.sqrt(nb3_err) # Sigma for each filter.
							
							nb2_err = nb2_pois**2 + nb2_err_scat**2 # Adding Scatter term to Error
							nb2_err = np.sqrt(nb2_err) # Sigma for each filter.

							nb5_err = nb5_pois**2 + nb5_err_scat**2 # Adding Scatter term to Error
							nb5_err = np.sqrt(nb5_err) # Sigma for each filter.
							

							num = nb3_a + nb4_a
							denom = nb2_a + nb5_a

							num_err = np.sqrt(nb4_err**2 + nb3_err**2)
							denom_err = np.sqrt(nb2_err**2 + nb5_err**2)

							mgI = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
							#mgI_err = mgI * np.sqrt((num_err/num)**2 + (denom_err/denom)**2)
							term1 = np.divide(num_err, num, out=np.zeros_like(num_err), where=num>0)
							term2 = np.divide(denom_err, denom, out=np.zeros_like(denom_err), where=denom>0)
							mgI_err = mgI * np.sqrt(term1**2 + term2**2)

							radius = header["R_SUN"]   # Solar radius in pixels
							x_center = header["CRPIX1"]   # X center (pixel)
							y_center = header["CRPIX2"]   # Y center (pixel)

							if debug:
								print(radius,x_center,y_center)

							# Now Labelling the Off Disk Values as Zero

							ny, nx = mgI.shape
							Y, X = np.ogrid[:ny, :nx]
							dist = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
							mask = np.zeros_like(mgI, dtype=np.uint8)
							mask[dist <= radius] = 1
							emask = np.zeros_like(mgI_err, dtype=np.uint8)
							emask[dist <= radius] = 1
							emask[np.isnan(mgI_err)] = 0
							mgI = mgI * mask
							mgI_err = mgI_err * emask

							if debug == 1 and show_math_error == 1:
								indices = np.where(
									(num == 0) | (denom == 0) | np.isnan(num) | np.isnan(denom)
								)
								for idx in zip(*indices):
									print(f"Index: {idx}, num: {num[idx]}, denom: {denom[idx]}, 4: {nb4_a[idx]}, 3: {nb3_a[idx]}, 2: {nb2_a[idx]}, 5: {nb5_a[idx]}")

							print(f'Set {z+1} : Saving Mgi FITS	| {date}')

							# If header needs to be different, this would be a place to change it.

							# Modifying Header and Filename

							mgi_f = fpath[0].replace(src_base,dst_base)
							mgi_f = mgi_f.replace("NB04",f".m{m}_mg_index")
							mgi_f = mgi_f.replace("remote/","")
							if rebin == 1:
								mgi_f = mgi_f.replace(".fits","_binned.fits")
								if echo_op == 1:
									print(f'Writing Rebin Header Values')
								header['RBIN'] = (1, 'Image Rebinned')
								header['RBIN_O1'] = (original_shape[0], 'Original Width of Data')
								header['RBIN_O2'] = (original_shape[1], 'Original Height of Data')
								header['RBIN_N1'] = (rebinned_shape[0], 'Width after Rebin')
								header['RBIN_N2'] = (rebinned_shape[1], 'Height after Rebin')

							# Saving the Maps
							
							hdu = fits.PrimaryHDU(mgI, header=header)
							hdul_new = fits.HDUList([hdu])
							mgi_f = fpath[0].replace(src_base,dst_base)
							mgi_f = mgi_f.replace("NB04",f".m{m}_mg_index")
							mgi_f = mgi_f.replace("remote/","")
							if rebin == 1:
								mgi_f = mgi_f.replace(".fits","_binned.fits")
							hdul_new.writeto(mgi_f, overwrite=True)

							mgI_dat_arr = mgI
							
							# If header needs to be different, this would be a place to change it.
							mgi_f = mgi_f.replace("mg_index","mg_index_err")
							hdu = fits.PrimaryHDU(mgI_err, header=header)
							hdul_new = fits.HDUList([hdu])
							hdul_new.writeto(mgi_f, overwrite=True)
							print(f'Set {z+1} : Mg II Index Map saved | {mgi_f.split("/")[-1]}')

							mgI_err_dat_arr = mgI_err

							# Integrate Value here onwards

							nb4_i = nb4_i_err = nb2_i = nb2_i_err = nb3_i = nb3_i_err = nb5_i = nb5_i_err = None

							nb4_i, nb4_i_err = mgi.integrate_mgI_disk(nb4_a, header, nb4_err, integration_method)
							nb3_i, nb3_i_err = mgi.integrate_mgI_disk(nb3_a, header, nb3_err, integration_method)
							nb2_i, nb2_i_err = mgi.integrate_mgI_disk(nb2_a, header, nb2_err, integration_method)
							nb5_i, nb5_i_err = mgi.integrate_mgI_disk(nb5_a, header, nb5_err, integration_method)

							"""
							nb4_i_err = np.sqrt(np.sum(nb4_err**2)) # Ideally, this should be sent in the function above.  Done
							nb3_i_err = np.sqrt(np.sum(nb3_err**2)) # So that it can be processed for on-disk values. 
							nb2_i_err = np.sqrt(np.sum(nb2_err**2))
							nb5_i_err = np.sqrt(np.sum(nb5_err**2))
							"""

							num_i = nb3_i + nb4_i
							denom_i = nb2_i + nb5_i

							num_i_err = np.sqrt(nb4_i_err**2 + nb3_i_err**2)
							denom_i_err = np.sqrt(nb2_i_err**2 + nb5_i_err**2)

							mgI_i = np.divide(num_i, denom_i)
							mgI_i_err = mgI_i * np.sqrt((num_i_err/num_i)**2 + (denom_i_err/denom_i)**2)
							
							mgI_arr.append(mgI_i)
							mgI_err_arr.append(mgI_i_err)
							num_arr.append(num_i)
							den_arr.append(denom_i)
							num_err_arr.append(num_i_err)
							den_err_arr.append(denom_i_err)

							timestamp = datetime.strptime(re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2})', mgi_f).group(1), "%Y-%m-%dT%H.%M")				
												
							dat += f'NB04 [{iterations}] : {str(nb4_i)} \n'
							dat += f'NB03 [{iterations}] : {str(nb3_i)} \n'
							dat += f'NB02 [{iterations}] : {str(nb2_i)} \n'
							dat += f'NB05 [{iterations}] : {str(nb5_i)} \n\n'
							
							dat += f'NB04 Error [{iterations}] : {str(nb4_i_err)} \n'
							dat += f'NB03 Error [{iterations}] : {str(nb3_i_err)} \n'
							dat += f'NB02 Error [{iterations}] : {str(nb2_i_err)} \n'
							dat += f'NB05 Error [{iterations}] : {str(nb5_i_err)} \n\n'

							dat += f'Numer [{iterations}] : {str(num_i)} \n'
							dat += f'Denom [{iterations}] : {str(denom_i)} \n\n'
							dat += f'Numer_Err [{iterations}] : {str(num_i_err)} \n'
							dat += f'Denom_Err [{iterations}] : {str(denom_i_err)} \n\n'

							dat += f'SUIT Mg II Index [{iterations}] : {str(mgI_i)} \n'
							dat += f'SUIT Mg II Error [{iterations}] : {str(mgI_i_err)} \n'
							dat += f'SUIT Mg II Tstamp[{iterations}] : {str(set_timestamp)} \n\n'
							
							day_mg_II.append(mgI_i)
							# Integration Ends

							print(f'Set {z+1} processed. | {tsnow()}')
							z = z+1
							
							nb4_f, nb3_f, nb2_f, nb5_f, f_window, nb4_ts = None,None,None,None,None,None # Initial Initializations								

							if one_set_only:
								break
				else:
					dat += f'Skipped: Mg II Index Generation Skipped \n'
				if exposure_corrected == 1:
					dat+= f'Exposures : Corrected \n'
				else:
					dat+= f'Exposures : Equal \n'
				mgI_arr = np.array(mgI_arr)
				mgI_err_arr = np.array(mgI_err_arr)
				num_arr = np.array(num_arr)
				den_arr = np.array(den_arr)
				num_err_arr = np.array(num_err_arr)
				den_err_arr = np.array(den_err_arr)
				dat += '\n'
				
				nan_mask = ~np.isnan(num_arr)
				dat += f'Numerator : {np.mean(num_arr[nan_mask])}\n'
				nan_mask = ~np.isnan(den_arr)
				dat += f'Denominator : {np.mean(den_arr[nan_mask])}\n'
				nan_mask = ~np.isnan(num_arr)
				dat += f'Numerator Err: {np.mean(num_err_arr[nan_mask])}\n'
				nan_mask = ~np.isnan(den_arr)
				dat += f'Denominator Err: {np.mean(den_err_arr[nan_mask])}\n'
				nan_mask = ~np.isnan(mgI_arr)
				dat += f'SUIT Magnesium Index : {np.mean(mgI_arr[nan_mask])}\n'
				dat += f'SUIT Magnesium Index Error : {np.sqrt(np.sum(mgI_err_arr[nan_mask]*mgI_err_arr[nan_mask])) / len(mgI_arr[nan_mask])}\n'
				if iterations > 1:
					print(f'{day_mg_II}')
					dat += f'SUIT Magnesium Std. Error: {np.std(day_mg_II)}\n'
				if rebin == 1:
					dat += f'\nBinned : Yes'
				else:
					dat += 	f'\nBinned : 0'
				dat += f'\nMethod: {integration_name}'
				dat += f'\n\n ------------------------------------------------\n'

				dat += f'GOME2B : {mgi.get_mg_index("GOME2B",timestamp)}\n'
				dat += f'GOME2B TS : {mgi.get_index_tstamp("GOME2B",timestamp)}\n'
				dat += f'GOME2C : {mgi.get_mg_index("GOME2C",timestamp)}\n'
				dat += f'GOME2C TS : {mgi.get_index_tstamp("GOME2C",timestamp)}\n'
				dat += f'Bremen Composite : {mgi.get_mg_index("composite",timestamp)}\n'
				dat += f'Bremen Composite TS : {mgi.get_index_tstamp("composite",timestamp)}\n'
				dat += f"Bremen Composite Error : {mgi.get_bremen_error('composite',timestamp)}\n"
				dat += f'Bremen Extended : {mgi.get_mg_index("extended",timestamp)}\n'
				dat += f'Bremen Extended TS : {mgi.get_index_tstamp("extended",timestamp)}\n'
				iterations = 0
				if save_integrate_file == 1:	
					current_folder = os.path.join(dst_base,yyyy,mm,dd,subfolder)
					dat_f = open(os.path.join(current_folder,integration_file_name),'w')
					dat_f.write(dat)
					dat_f.close()
					dat = ''
					if echo_op == 1:
						print(f'Data file for {date} saved as {integration_file_name}.')
				else:
					print('*******************************************')
					print(dat)

				nb4_f, nb3_f, nb2_f, nb5_f = None,None,None,None
				f_window, nb4_ts = None,None
				nb4_h, nb3_h, nb2_h, nb5_h = None,None,None,None

				if break_one == 1:
					sys.exit()
			else:
				print(f'Empty normal_4k Folder | {date}')
		 
		 # Clear all user-defined variables (not recommended in most cases)
		else:
			print(f'No Data Folder | {date}')
		for var in list(globals().keys()):
				if var not in ['__builtins__', '__name__', '__file__', '__doc__']:
						#del globals()[var]
						xx = 1
	else:
		print('No Data')
if parallel == 0:
	print(f'Intiating Serial Mode')
	for date in dates:
		if date not in skip_dates:
			mgi_generate(date)
		if break_one ==1:
			break
	print(f'Execution Complete	| {tsnow()}')
else:
	t1 = time.time()
	cores_use = int((atul.cores - leave_out_cores))# / 2)
	print(f'Initiating Parallel Processing using {cores_use} Cores')
	with multiprocessing.Pool(processes=cores_use) as pool:	# Set processes to number of CPUs or whatever limit you prefer
		pool.map(mgi_generate, date_list)
	t2 = time.time()
	print(f'Execution Time:{t2-t1} seconds	| {tsnow()}')

	