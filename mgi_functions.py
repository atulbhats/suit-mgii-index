"""

All the necessary functions to generate the Mg II index in mgi_generate.py

Author: Atul

Last Modified: 22nd May 2025

"""

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
import multiprocessing as mp
import atul_settings as atul

from skimage.draw import line
from scipy.optimize import leastsq

debug = 0
preview_fit = 0
echo_align = 0 # mgi_generate will set its own value and process it. 

def disk_integrated_flux_2D(data, disk_mask, mu):

	# pixel solid angle: convert arcsec^2 -> steradians
	pixel_area = 0.7 * 0.7
	sr_per_pixel = pixel_area * (4.8481e-6)**2
	
	# sum: data * mu * (solid angle)
	flux_2D = np.nansum(data[disk_mask] * mu[disk_mask])/100. * sr_per_pixel
	
	return flux_2D

def __disk_integrated_flux_1D(data, mu, disk_mask, nbins=100):
	
	mu_on_disk = mu[disk_mask].ravel()
	data_on_disk = data[disk_mask].ravel()

	
	mu_bins = np.linspace(0, 1, nbins+1)
	mu_centers = 0.5*(mu_bins[1:] + mu_bins[:-1])  # midpoints


	I_of_mu = np.zeros(nbins, dtype=np.float64)
	for i in range(nbins):
		bmin, bmax = mu_bins[i], mu_bins[i+1]
		in_bin = (mu_on_disk >= bmin) & (mu_on_disk < bmax)
		if np.any(in_bin):
			I_of_mu[i] = np.mean(data_on_disk[in_bin])
		else:
			I_of_mu[i] = 0.0
	# Now integrate 2 pi * ∫[0->1] I(mu)*mu dmu
	flux_1D = 2.0 * np.pi * simps(I_of_mu * mu_centers, mu_centers)


	return flux_1D

def fractional_year_to_datetime(fractional_year):
	year = int(fractional_year)
	# Compute the fractional part in days (using a leap-aware method)
	start_of_year = datetime(year, 1, 1)
	start_of_next_year = datetime(year + 1, 1, 1)
	year_duration = (start_of_next_year - start_of_year).total_seconds()
	
	seconds_into_year = (fractional_year - year) * year_duration
	final_date = start_of_year + timedelta(seconds=seconds_into_year)
	
	return final_date

def get_index_tstamp(ver, timestamp):	
	local_files = {
		'GOME2C': 'GOME2C_Index_classic.dat',
		'extended': 'MgII_extended.dat',
		'composite': 'MgII_composite.dat',
		'GOME2B': 'GOME2B_Index_classic.dat'
	}

	url = 'https://www.iup.uni-bremen.de/gome/solar/'

	index_columns = {
		'GOME2C': -1,
		'extended': -2,
		'composite': 3,
		'GOME2B': -1
	}

	year_ind = 0
	mont_ind = 1
	date_ind = 2
	mg2i_ind = index_columns[ver]

	file_name = os.path.join('ext_data',local_files[ver])
	found = False
	index_val = 0
	frac_year = 0
	def find_index_in_lines(lines):
		nonlocal index_val, found
		for l in lines:
			l = l.strip().replace("  ", " ")
			n = l.split(" ")
			if len(n) < max(year_ind, mont_ind, date_ind, abs(mg2i_ind)) + 1:
				continue
			y = n[year_ind].split(".")[0]
			if y == timestamp.strftime('%Y'):
				ts_m = timestamp.strftime('%-m') if len(n[mont_ind]) < 2 else timestamp.strftime('%m')
				if n[mont_ind] == ts_m:
					ts_d = timestamp.strftime('%-d') if len(n[date_ind]) < 2 else timestamp.strftime('%d')
					if n[date_ind] == ts_d:
						index_val = n[year_ind]
						found = True
						break

	# Check if file exists locally
	if os.path.exists(file_name):
		with open(file_name, 'r') as f:
			lines = f.readlines()
			find_index_in_lines(lines)

	if not found:
		msg = f'Fetching {ver} data from the web...'
		sys.stdout.write(msg)
		sys.stdout.flush()
		response = requests.get(url+local_files[ver])
		response.raise_for_status()
		content = response.text
		lines = content.split("\n")

		# Save to local file
		with open(file_name, 'w') as f:
			f.write(content)

		find_index_in_lines(lines)

		# Clear the line after fetching
		sys.stdout.write('\r' + ' ' * len(msg) + '\r')
		sys.stdout.flush()

	return fractional_year_to_datetime(float(index_val))

def get_mg_index(ver, timestamp):
	local_files = {
		'GOME2C': 'GOME2C_Index_classic.dat',
		'extended': 'MgII_extended.dat',
		'composite': 'MgII_composite.dat',
		'GOME2B': 'GOME2B_Index_classic.dat'
	}

	url = 'https://www.iup.uni-bremen.de/gome/solar/'

	index_columns = {
		'GOME2C': -1,
		'extended': -2,
		'composite': 3,
		'GOME2B': -1
	}

	year_ind = 0
	mont_ind = 1
	date_ind = 2
	mg2i_ind = index_columns[ver]

	file_name = local_files[ver]
	found = False
	index_val = 0
	frac_year = 0
	def find_index_in_lines(lines):
		nonlocal index_val, found
		for l in lines:
			l = l.strip().replace("  ", " ")
			n = l.split(" ")
			if len(n) < max(year_ind, mont_ind, date_ind, abs(mg2i_ind)) + 1:
				continue
			y = n[year_ind].split(".")[0]
			if y == timestamp.strftime('%Y'):
				ts_m = timestamp.strftime('%-m') if len(n[mont_ind]) < 2 else timestamp.strftime('%m')
				if n[mont_ind] == ts_m:
					ts_d = timestamp.strftime('%-d') if len(n[date_ind]) < 2 else timestamp.strftime('%d')
					if n[date_ind] == ts_d:
						index_val = n[mg2i_ind]
						found = True
						break

	# Check if file exists locally
	if os.path.exists(file_name):
		with open(file_name, 'r') as f:
			lines = f.readlines()
			find_index_in_lines(lines)

	# If not found locally, fetch from the web
	if not found:
		msg = f'Fetching {ver} data from the web...'
		sys.stdout.write(msg)
		sys.stdout.flush()
		response = requests.get(url+local_files[ver])
		response.raise_for_status()
		content = response.text
		lines = content.split("\n")

		# Save to local file
		with open(file_name, 'w') as f:
			f.write(content)

		find_index_in_lines(lines)

		# Clear the line after fetching
		sys.stdout.write('\r' + ' ' * len(msg) + '\r')
		sys.stdout.flush()

	return index_val

def get_bremen_error(ver, timestamp):
	local_files = {
		'composite': 'MgII_composite.dat',
	}

	url = 'https://www.iup.uni-bremen.de/gome/solar/'

	index_columns = {
		'composite': 3,
	}

	year_ind = 0
	mont_ind = 1
	date_ind = 2
	mg2i_err = index_columns[ver] + 1
	mg2i_ind = index_columns[ver]
	file_name = local_files[ver]
	found = False
	index_err = 0
	frac_year = 0
	
	def find_index_in_lines(lines):
		nonlocal found, index_err
		for l in lines:
			l = l.strip().replace("  ", " ")
			n = l.split(" ")
			if len(n) < max(year_ind, mont_ind, date_ind, abs(mg2i_ind)) + 1:
				continue
			y = n[year_ind].split(".")[0]
			if y == timestamp.strftime('%Y'):
				ts_m = timestamp.strftime('%-m') if len(n[mont_ind]) < 2 else timestamp.strftime('%m')
				if n[mont_ind] == ts_m:
					ts_d = timestamp.strftime('%-d') if len(n[date_ind]) < 2 else timestamp.strftime('%d')
					if n[date_ind] == ts_d:
						index_err = n[mg2i_err]
						found = True
						break

	with open(file_name, 'r') as f:
		lines = f.readlines()
		find_index_in_lines(lines)

	if not found:
		msg = f'Fetching {ver} data from the web...'
		sys.stdout.write(msg)
		sys.stdout.flush()
		response = requests.get(url+local_files[ver])
		response.raise_for_status()
		content = response.text
		lines = content.split("\n")

		# Save to local file
		with open(file_name, 'w') as f:
			f.write(content)

		find_index_in_lines(lines)

		# Clear the line after fetching
		sys.stdout.write('\r' + ' ' * len(msg) + '\r')
		sys.stdout.flush()
	return index_err

def get_diagonal_scan_coords(data, spacing=10, x_cutoff=0, y_cutoff=2500, scan_lines = []):
	"""Generate 45° diagonal lines from (0, y) or (x, 0) but avoid lower-right origin scans"""
	h, w = data.shape
	coords = []

	for y in range(1,h,spacing):
		coords.append(((h-y,1),(h-1,y)))

	for x in range(1,500,spacing):
		coords.append(((1,x),(h-x,w-1)))
	scan_lines = coords
	return coords

l = 0
lno = 0
fil = 0
def detect_limb_edges_along_line(image, p1, p2, threshold):
	global l,lno,fil
	fil +=1
	"""Detect two sharp transitions along the line"""
	rr, cc = line(p1[0], p1[1], p2[0], p2[1])
	values = image[rr, cc]
	limb_indices = []
	
	gradient = values #np.gradient(values)
	#threshold = 2500 #threshold_ratio * (np.max(values) - np.min(values))
	limb_indices = np.where(values >= threshold)[0]
	
	diff = np.diff(limb_indices)
	breaks = np.where(diff > 1)[0]
	groups = np.split(limb_indices, breaks + 1) # Split into groups of consecutive indices
	limb_indices = []
	limb_indices = max(groups, key=len)# Find the longest group
	

	l+=1
	if len(limb_indices) > 150:
		"""
		d = []
		deli = []
		lno+=1
		for i in range(0,len(limb_indices)-1):
			x1 = cc[limb_indices[i]]
			y1 = rr[limb_indices[i]]
			x2 = cc[limb_indices[i+1]]
			y2 = rr[limb_indices[i+1]]
			d.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))
			if i > 1:
				d2 = d[i-2]
				d1 = d[i-1]
				if (d[i] - d1) > (2*(d2-d1)):
					deli.append(i+1)

		if len(deli) > 0:
			limb_indices = np.delete(limb_indices,deli)

		d = []
		deli = []
		for i in range(0,len(limb_indices)-2):
			x1 = cc[limb_indices[i]]
			y1 = rr[limb_indices[i]]
			x2 = cc[limb_indices[i+1]]
			y2 = rr[limb_indices[i+1]]
			d.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))

		for i in range(0,len(d)-2):
			d2 = d[i+2]
			d1 = d[i+1]
			if (d1 - d[i]) > (2*(d2-d1)):
				deli.append(i)

		if len(deli) > 0:
			limb_indices = np.delete(limb_indices,deli)

		"""
		return [(cc[limb_indices[0]], rr[limb_indices[0]]), (cc[limb_indices[-1]], rr[limb_indices[-1]])]
	else:
		return []

def fit_circle(xs, ys):
	"""Fit a circle to given x and y using least squares"""
	def calc_R(xc, yc):
		return np.sqrt((xs - xc)**2 + (ys - yc)**2)

	def f(c):
		Ri = calc_R(*c)
		return Ri - Ri.mean()

	x_m = np.mean(xs)
	y_m = np.mean(ys)
	center, _ = leastsq(f, (x_m, y_m))
	radius = calc_R(*center).mean()
	return center[0], center[1], radius

c = 0
def fit_for_file(file,spacing,rebin_size):
	scan_lines = []
	global c
	# File is sent like this. files = [(nb4_f,nb4_h)] so file = (data, header)
	data = file[0]
	header = file[1]
	fltr = header['FTR_NAME']
	x = 1
	if fltr == 'NB04':
		threshold = 3000
	elif fltr == 'NB03':
		threshold = 2000
	elif fltr == 'NB02':
		threshold = 2000
	elif fltr == 'NB05':
		threshold = 2000
	else:
		print('NOT SUIT FILE. Exiting')
		sys.exit() 

	threshold = threshold*rebin_size*rebin_size
	scan_lines = []
	coords = get_diagonal_scan_coords(data, spacing,scan_lines)
	limb_points = []
	for (p1, p2) in coords:
		points = detect_limb_edges_along_line(data, p1, p2, threshold)
		if len(points) == 2:
			limb_points.extend(points)
	#print(limb_points)
	if len(limb_points) < 10:
		print("Not enough limb points detected.")
		return 0
	xs, ys = zip(*limb_points)
	x0, y0, r = fit_circle(np.array(xs), np.array(ys))
	xf = header['CRPIX1']
	yf = header['CRPIX2']
	rf = header['R_SUN']
	ts = header['T_OBS']
	if debug:
		print()
		print(f"File: {fltr}")
		print(f"Timestamp: {ts}")
		print(f"Estimated Center: x = {x0}, y = {y0:.2f}, radius = {r:.2f}")
		print(f"Header Center : x = {xf}, y = {yf}, radius = {rf}")

	if debug and preview_fit:
		plt.figure(figsize=(8, 8))
		plt.imshow(data, origin='lower', cmap='gray') #, norm='log')
		plt.plot(x0, y0, 'rx', markersize=10, label='Fitted Center')
		x = 0
		for (p1, p2) in coords:
				rr, cc = line(p1[0], p1[1], p2[0], p2[1])
				plt.plot(cc, rr, color='cyan', linewidth=0.5)
		plt.scatter(xs, ys, s=5, c='yellow', label='Limb Points')
		circle = plt.Circle((x0, y0), r, color='red', fill=False, linestyle='--', label='Fitted Circle')
		plt.gca().add_patch(circle)
		plt.legend()
		plt.title(f"Fitted Solar Disk Center | {fltr}")
		if debug:
			plt.show()
		plt.close()
	return x0,y0,r

def align_filters_by_fitting(files, z=0, rebin_size=2):
	print(f'Set {z+1} : Aligning by Fitting Circle')
	fname = []
	cenx = []
	ceny = []
	radi = []
	x,y,r = None,None,None
	spacing = 50
	for f in files:
		x,y,r = fit_for_file(f,spacing,rebin_size)
		if x == None or y == None:
			break
		fname.append(f)
		cenx.append(x)
		ceny.append(y)
		radi.append(r)


		hdr = f[1]
		hdr['CP1_OL'] = hdr['CRPIX1']
		hdr['CP2_OL'] = hdr['CRPIX2']
		hdr['R_SUNO'] = hdr['R_SUN']
		hdr['R_SUN'] = r #(f'{r}', 'Updated Radius by Fitting')
		hdr['CRPIX2'] = y #(f'{y}', 'Updated Center by Fitting')
		hdr['CRPIX1'] = x #(f'{x}', 'Updated Center by Fitting')


	maps = sunpy.map.Map(files, sequence=True)
	ref_map = maps[0]
	ref_crpix1 = float(ref_map.meta['CRPIX1'])
	ref_crpix2 = float(ref_map.meta['CRPIX2'])
	aligned_maps = []
	i = 0
	fil_seq = [4,2,3,5]
	for mp in maps:
		cur_crpix1 = float(mp.meta['CRPIX1'])
		cur_crpix2 = float(mp.meta['CRPIX2'])
		shift_x = ref_crpix1 - cur_crpix1
		shift_y = ref_crpix2 - cur_crpix2
		data = mp.data
		shifted_data = shift(data, shift=(shift_y, shift_x), mode='nearest')  # shift in (y, x)
		new_meta = mp.meta.copy()
		new_map = sunpy.map.Map(shifted_data, new_meta)
		aligned_maps.append(new_map)

	print()
	return aligned_maps



def align_filters_by_template(files, z = 0):
	global echo_align
	Sequence = sunpy.map.Map(files, sequence=True)
	if echo_align == 1: 
		print(f'Set {z+1} : Aligning Images with respect to first image with Template.')
	ref_img=sunpy.map.Map(files[0])
	top_right = SkyCoord(550 * u.arcsec, 1100 * u.arcsec, frame=ref_img.coordinate_frame)
	bottom_left = SkyCoord(-550 * u.arcsec, 800 * u.arcsec, frame=ref_img .coordinate_frame)
	ref_submap = ref_img.submap(bottom_left, top_right=top_right)
	#ref_submap.peek()
	aligned_maps=mc_coalign(Sequence,template=ref_submap,clip=False)
	return aligned_maps

def align_filters_by_first(files, z = 0):
	global echo_align
	Sequence = sunpy.map.Map(files, sequence=True)
	if echo_align == 1: 
		print(f'Set {z+1} : Aligning Images with respect to first image.')
	aligned_maps=mc_coalign(Sequence,layer_index=0,clip=False)
	return aligned_maps

def align_filters_by_relative(files, z = 0):
	#files = nb4_f,nb3_f,nb2_f,nb5_f
	global echo_align
	if echo_align == 1: 
		print(f'Aligning.')
	nb4_f = files[0]
	nb3_f = files[1]
	nb2_f = files[2]
	nb5_f = files[3]


	print(f'Set {z+1} : Sequential Alignment with Template ')
	a_files = [nb2_f,nb5_f]
	Sequence = sunpy.map.Map(a_files, sequence=True)
	print(f'Set {z+1} : Aligning NB05 from NB02')
	ref_img=sunpy.map.Map(a_files[0])
	top_right = SkyCoord(550 * u.arcsec, 1100 * u.arcsec, frame=ref_img.coordinate_frame)
	bottom_left = SkyCoord(-550 * u.arcsec, 800 * u.arcsec, frame=ref_img .coordinate_frame)
	ref_submap = ref_img.submap(bottom_left, top_right=top_right)
	aligned_maps=mc_coalign(Sequence,template=ref_submap,clip=False)

	nb2_a = aligned_maps[0]
	nb5_a = aligned_maps[1]

	a_files = [nb2_f,nb4_f]
	Sequence = sunpy.map.Map(a_files, sequence=True)
	print(f'Set {z+1} : Aligning NB04 from NB02')
	ref_img=sunpy.map.Map(a_files[0])
	top_right = SkyCoord(550 * u.arcsec, 1100 * u.arcsec, frame=ref_img.coordinate_frame)
	bottom_left = SkyCoord(-550 * u.arcsec, 800 * u.arcsec, frame=ref_img .coordinate_frame)
	ref_submap = ref_img.submap(bottom_left, top_right=top_right)
	aligned_maps=mc_coalign(Sequence,template=ref_submap,clip=False)

	hdu = fits.PrimaryHDU(aligned_maps[1].data, header=aligned_maps[1].fits_header) #Saving dummy fits
	hdul_new = fits.HDUList([hdu])
	dump = os.path.join(atul.base_path,'dump')
	if not os.path.exists(dump): os.mkdir(dump)
	hdul_new.writeto(os.path.join(dump,'dummy.fits'), overwrite=True)
	if is_path_or_data(nb3_f) == 'data':
		a_files = [(aligned_maps[1].data,aligned_maps[1].fits_header),nb3_f]
	else:
		a_files = [dummy,nb3_f]
	Sequence = sunpy.map.Map(a_files, sequence=True)
	print(f'Set {z+1} : Aligning NB03 from NB04 [aligned]')
	ref_img=sunpy.map.Map(files[0])
	top_right = SkyCoord(550 * u.arcsec, 1100 * u.arcsec, frame=ref_img.coordinate_frame)
	bottom_left = SkyCoord(-550 * u.arcsec, 800 * u.arcsec, frame=ref_img .coordinate_frame)
	ref_submap = ref_img.submap(bottom_left, top_right=top_right)
	aligned_maps=mc_coalign(Sequence,template=ref_submap,clip=False)

	os.remove(os.path.join(dump,'dummy.fits'))

	nb4_a = aligned_maps[0]
	nb3_a = aligned_maps[1]

	aligned_maps = [nb4_a,nb3_a,nb2_a,nb5_a]
	
	return aligned_maps

def align_filters_centered(files,z = 0):
	nb4_f = files[0]
	nb3_f = files[1]
	nb2_f = files[2]
	nb5_f = files[3]
	print(f'Set {z+1} : Sun Centre, Already Aligned.')
	
	nb4_a = sunpy.map.Map(nb4_f)
	nb5_a = sunpy.map.Map(nb5_f)
	nb3_a = sunpy.map.Map(nb3_f)
	nb2_a = sunpy.map.Map(nb2_f)

	aligned_maps = []
	aligned_maps.append(sunpy.map.Map(nb4_f))
	aligned_maps.append(sunpy.map.Map(nb3_f))
	aligned_maps.append(sunpy.map.Map(nb2_f))
	aligned_maps.append(sunpy.map.Map(nb5_f))

	return aligned_maps

def align_filters_by_crpix(files,z=0):
	
	maps = sunpy.map.Map(files, sequence=True)
	# Choose the first image as the reference
	ref_map = maps[0]
	ref_crpix1 = ref_map.meta['CRPIX1']
	ref_crpix2 = ref_map.meta['CRPIX2']

	aligned_maps = []
	i = 0
	fil_seq = [4,3,2,5]
	for mp in maps:
		if echo_align == 1:
			print(f'Set {z+1} : Aligning NB0{fil_seq[i]} Based on CRPIX values.')
		i+=1
		# Current image CRPIX
		cur_crpix1 = mp.meta['CRPIX1']
		cur_crpix2 = mp.meta['CRPIX2']
		
		# Compute the shift in pixels needed to align with reference
		shift_x = ref_crpix1 - cur_crpix1
		shift_y = ref_crpix2 - cur_crpix2
		
		# Apply shift to the image data
		shifted_data = shift(mp.data, shift=(shift_y, shift_x), mode='nearest')  # shift in (y, x)

		# Copy metadata and update if needed
		new_meta = mp.meta.copy()
		new_map = sunpy.map.Map(shifted_data, new_meta)
		
		aligned_maps.append(new_map)

	return aligned_maps

def patch_header(h):
	# Insert or correct all needed keys
	h['CTYPE1'] = h.get('CTYPE1', 'HPLN-TAN')
	h['CTYPE2'] = h.get('CTYPE2', 'HPLT-TAN')
	h['CUNIT1'] = 'arcsec'
	h['CUNIT2'] = 'arcsec'
	h['CRVAL1'] = h.get('CRVAL1', 0.0)
	h['CRVAL2'] = h.get('CRVAL2', 0.0)
	h['CRPIX1'] = h.get('CRPIX1', (h.get('NAXIS1', 1) + 1) / 2.0)
	h['CRPIX2'] = h.get('CRPIX2', (h.get('NAXIS2', 1) + 1) / 2.0)
	h['CDELT1'] = h.get('CDELT1', 0.6)
	h['CDELT2'] = h.get('CDELT2', 0.6)
	return h

def align_filter_by_crpix(files,headers,filter_name):
	path = 0
	ffiles = []
	if is_path_or_data(files[0]) == 'path':
		#print(f'Aligning Files by path')
		path = 1
		for f in files:
			with fits.open(f) as hdul:
				data = hdul[0].data
				header = hdul[0].header
			ffiles.append((data,header))
		#print(ffiles[0])
	else:
		#print(f'Aligning Files by data')
		def_CDELT = def_CRVAL = def_CUNIT = def_CTYPE = None
		hvars = []
		sunpy_required_keys = [
			'CTYPE1', 'CTYPE2','CUNIT1', 'CUNIT2','CRVAL1', 'CRVAL2','CRPIX1', 'CRPIX2','CDELT1', 'CDELT2','DSUN_OBS', 'RSUN_REF','HGLN_OBS', 'HGLT_OBS','DATE-OBS','CROTA2'	 # If PC-matrix not present			
		]
		reference_header_data = {}
		for idx, (f, h) in enumerate(zip(files, headers)):
			if idx == 0:
				for key in sunpy_required_keys:
					if key in h:
						reference_header_data[key] = h[key]
					else:
						print(f"Warning: Key {key} missing in first header!")
			else:
				# Subsequent headers: fill in missing keys from reference_header_data
				for key in sunpy_required_keys:
					if key not in h:
						h[key] = reference_header_data[key]
						#print(f"Filled missing key {key} from reference_header_data for file {f}")

			for key in sunpy_required_keys:
				if key in h:
					reference_header_data[key] = h[key]				

			ffiles.append((f, h))
		
		ref_crpix1 = headers[0]['CRPIX1']
		ref_crpix2 = headers[0]['CRPIX2']
		obs_date = headers[0]['DATE-OBS']

	maps = sunpy.map.Map(ffiles, sequence=True)
	ref_map = maps[0]

	if path == 1:
		ref_crpix1 = ref_map.meta['CRPIX1']
		ref_crpix2 = ref_map.meta['CRPIX2']
		obs_date = ref_map.meta['DATE-OBS']
	else:
		ref_crpix1 = ref_crpix1
		ref_crpix2 = ref_crpix2

	
	aligned_maps = []
	if echo_align == 1:
		print(f'Alignment of {filter_name} Underway. | {obs_date[8:10]}-{obs_date[5:7]}-{obs_date[0:4]} | {dt.now().strftime("%H:%M:%S")}')
	for m in maps:
		cur_crpix1 = m.meta['CRPIX1']
		cur_crpix2 = m.meta['CRPIX2']
		shift_x = ref_crpix1 - cur_crpix1
		shift_y = ref_crpix2 - cur_crpix2
		shifted_data = shift(m.data, shift=(shift_y, shift_x), mode='nearest')	# shift in (y, x)
		new_meta = m.meta.copy()
		new_map = sunpy.map.Map(shifted_data, new_meta)
		aligned_maps.append(new_map)
	return aligned_maps

def align_filter_by_template(files,filter_name):
	Sequence = sunpy.map.Map(files, sequence=True)
	print(f'Aligning {filter_name} Images with respect to first image with Template.')
	ref_img=sunpy.map.Map(files[0])
	top_right = SkyCoord(550 * u.arcsec, 1100 * u.arcsec, frame=ref_img.coordinate_frame)
	bottom_left = SkyCoord(-550 * u.arcsec, 800 * u.arcsec, frame=ref_img .coordinate_frame)
	ref_submap = ref_img.submap(bottom_left, top_right=top_right)
	#ref_submap.peek()
	aligned_maps=mc_coalign(Sequence,template=ref_submap,clip=False)
	return aligned_maps

def align_filter_by_first(files,filter_name):
	Sequence = sunpy.map.Map(files, sequence=True)
	print(f'Aligning {filter_name} Images with respect to first image.')
	aligned_maps=mc_coalign(Sequence,layer_index=0,clip=False)
	return aligned_maps

def integrate_mgI_disk(data, header, error_data,integration_method):
	ny, nx = data.shape
	y_vals, x_vals = np.ogrid[:ny, :nx]
	center_x = header['CRPIX1']
	center_y = header['CRPIX2']
	radius_pixels = header['R_SUN']
	radius_arcsec = header['RSUN_OBS']
	radius_sun = radius_pixels
	d = 1
	distance_from_center = np.sqrt((x_vals - center_x)**2 + (y_vals - center_y)**2)
	disk_mask = (distance_from_center <= radius_sun)
	mu = np.zeros_like(data, dtype=np.float64)
	mu[disk_mask] = np.sqrt(1.0 - (distance_from_center[disk_mask]/radius_sun)**2)
	if d is not 1:
		raise NotImplementedError("Only d=1 (1D mu integration) is supported.")
	else:
		flux, flux_error = disk_integrated_flux(data, mu, disk_mask, integration_method, error_data)

	return flux, flux_error

# Below Function is Deprecated / Not Called
def integrate_filter(files,d,rebin, method): # This takes the single filter files and integrates each and takes mean. 
	fluxes = []
	errors = []
	path = 0
	if is_path_or_data(files[0]) == 'path':
		path = 1
		maps = sunpy.map.Map(files, sequence=True)
	else:
		maps = files

	for smap in maps:
		if path == 1:
			data = smap.data  # This might be a bug in future, when rebinning is not done. 
			header = smap.fits_header
		else:
			data = smap.data
			header = smap.fits_header # This is giving the SunpyMetaWarning
		ny, nx = data.shape
		y_vals, x_vals = np.ogrid[:ny, :nx]
		center_x = header['CRPIX1']
		center_y = header['CRPIX2']
		solar_radius_pix = header['R_SUN'] #pixels
		#print(f'Solar Radius {solar_radius_pix}')
		radius_sun = solar_radius_pix	# Adjust as needed #arcseconds
		
		distance_from_center = np.sqrt((x_vals - center_x)**2 + (y_vals - center_y)**2)
		disk_mask = (distance_from_center <= radius_sun)
		
		mu = np.zeros_like(data, dtype=np.float64)
		mu[disk_mask] = np.sqrt(1.0 - (distance_from_center[disk_mask]/solar_radius_pix)**2)
		#print(header['DATE-OBS'])
		plt.imshow(mu, cmap='gray')
		#plt.show()
		if d == 1:
				flux, flux_error = disk_integrated_flux_1D(data, mu, disk_mask,method)
		else:
				raise NotImplementedError("Only d=1 (1D mu integration) is supported.")

		fluxes.append(flux)
		errors.append(flux_error)
		break

	mean_flux = np.mean(fluxes)
	combined_error = np.sqrt(np.sum(np.array(errors)**2)) / len(errors)

	return mean_flux, combined_error

# Below Function is Deprecated / Not Called
def disk_integrated_flux_2D(data, mu, disk_mask):

	# pixel solid angle: convert arcsec^2 -> steradians
	pixel_area = 0.7 * 0.7
	sr_per_pixel = pixel_area * (4.8481e-6)**2
	
	# sum: data * mu * (solid angle)
	flux_2D = np.nansum(data[disk_mask] * mu[disk_mask])/100. * sr_per_pixel
	
	return flux_2D

#1D version which infact loses feature-to-feature variations and just gives out average value.
mu_on_disk = []
data_on_disk = []
I_of_mu = []
I_error = []
mu_bins = []
mu_bins = []
nbins_c= [10] #[10000,100000,1000000,10000000] #[10,100,1000,10000] #,100000,1000000]
show_tables = 1
xx = 0 

# Below Function is Deprecated / Not Called
def integrate_Imu_counts(I_of_mu, mu_centers, I_error):
	n = len(mu_centers)

def disk_integrated_flux(data, mu, disk_mask, method, error_data = None):
	global mu_on_disk, data_on_disk,I_of_mu,I_error,mu_bins,show_tables
	#mu_on_disk = mu[disk_mask].ravel()
	data_on_disk = data[disk_mask].ravel()
	if method == 0:
		flux = np.sum(data_on_disk)
		if error_data is not None:
			error_on_disk = error_data[disk_mask].ravel()
			flux_err = np.sqrt(np.sum(error_on_disk**2)) 
		else:
			flux_err = np.sqrt(flux)
	else:
		print(f'But Sankar and Sreekumar say Add them!')
		print(f'So yea! That must be discussed before we can finalize it.')
		sys.exit()
		"""
		for nbins in nbins_c:
			I_of_mu = []
			I_error = []
			mu_bins = []
			mu2_bins = np.linspace(0, 1, nbins + 1)
			mu_bins = np.sqrt(mu2_bins)
			mu_centers = 0.5 * (mu_bins[1:] + mu_bins[:-1])
			I_of_mu = np.zeros(nbins)
			I_error = np.zeros(nbins)
			if show_tables == 1:
				print()
				print('Data for I and I_err calculation')
				print('# I \t\t mean \t\t std \t\t n \t\t I_err')
			for i in range(nbins):
				bmin, bmax = mu_bins[i], mu_bins[i+1]
				in_bin = (mu_on_disk >= bmin) & (mu_on_disk < bmax)
				if np.any(in_bin):
					values = data_on_disk[in_bin]
					errors = error_on_disk[in_bin]
					I_of_mu = values/len(nbins)
					I_of_mu_err = np.sqrt(np.sum(errors))/len(nbins)


			if method == 1:
				flux , flux_err = integrate_Imu_simpson(I_of_mu,combined_I_err,mu_centers)
			if method == 2:
				flux , flux_err = integrate_I_mu_gaussian_quad(I_of_mu,combined_I_err,mu_centers)					
		"""
	return flux,flux_err

# Below Function is Deprecated / Not Called
def disk_integrated_flux_1D(data, mu, disk_mask,method, error_data = None):
	global mu_on_disk, data_on_disk,I_of_mu,I_error,mu_bins,show_tables
	mu_on_disk = mu[disk_mask].ravel()
	data_on_disk = data[disk_mask].ravel()
	error_on_disk = error_data[disk_mask].ravel()
	if method == 0:
		flux = np.mean(data_on_disk)
		flux_err_poisson = np.sqrt(np.sum(data_on_disk)) / len(data_on_disk)
		if error_data == None:
			flux_err = flux_err_poisson
		else:
			flux_err_from_error = np.sqrt(np.sum(error_data**2))/len(data_on_disk)
			flux_err = np.sqrt(flux_err_from_error**2 + flux_err_poisson**2)
	else:
		for nbins in nbins_c:
			I_of_mu = []
			I_error = []
			mu_bins = []
			

			mu2_bins = np.linspace(0, 1, nbins + 1)  # since μ ∈ [0,1]
			mu_bins = np.sqrt(mu2_bins)			 # convert μ² back to μ

			# Compute bin centers
			mu_centers = 0.5 * (mu_bins[1:] + mu_bins[:-1])
			delta_mu = mu_bins[1] - mu_bins[0]
			

			I_of_mu = np.zeros(nbins)
			I_error = np.zeros(nbins)
			
			if show_tables == 1:
				print()
				print('Data for I and I_err calculation')
				print('# I \t\t mean \t\t std \t\t n \t\t I_err')
			for i in range(nbins):
				bmin, bmax = mu_bins[i], mu_bins[i+1]
				in_bin = (mu_on_disk >= bmin) & (mu_on_disk < bmax)
				if np.any(in_bin):
					values = data_on_disk[in_bin]
					errors = error_on_disk[in_bin]

					I_of_mu[i] = np.mean(values)
					I_error[i] = np.sqrt(I_of_mu[i]/len(values))	
					if error_data is not None:
						pixel_errors = error_on_disk[in_bin]
						bin_error = np.sqrt(np.sum(pixel_errors**2)) / len(pixel_errors) 
					else:
						bin_error = np.zeros()
				else:
					I_of_mu[i] = 0.0
					I_error[i] = 0.0
				if show_tables == 1: # Debug Line
					print(f"{i} {I_of_mu[i]:.4f} \t {np.mean(values):.4f} \t {np.std(values):.4f} \t {len(values):.4f}  \t {I_error[i]}")  # Debug Line
			combined_I_err = np.sqrt(I_error**2 + bin_error**2)
			if method == 1:
				flux , flux_err = integrate_Imu_simpson(I_of_mu,combined_I_err,mu_centers)
			if method == 2:
				flux , flux_err = integrate_I_mu_gaussian_quad(I_of_mu,combined_I_err,mu_centers)

	return flux, flux_err

# Below Function is Deprecated / Not Called
def integrate_Imu_simpson(I_of_mu,I_error, mu_centers):
	print(f'Solving Integral by Simpson Method')
	global nbins_c, show_tables
	nbins = len(mu_centers)
	#global mu_on_disk, data_on_disk,I_of_mu,I_error,mu_bins,xx
	f = I_of_mu * mu_centers
	h = (mu_centers[len(mu_centers)-1] - mu_centers[0])/len(mu_centers)
	#h = 1 / nbins
	w = []
	for i in range(0,nbins):
		if i == 0 or i == nbins-1:
			w.append((h/3) * 1)
		elif i % 2 == 0:
			w.append((h/3) * 2) # Even
		else:
			w.append((h/3) * 4) # Odd
	
	flux = 2.0 * np.pi * simpson(f, mu_centers)
	#print(len(w),len(mu_centers))
	print(w[0],mu_centers[0],I_error[0],I_of_mu[0])
	flux_int_err = 2 * np.pi * np.sqrt( np.sum((w * mu_centers * I_error)**2 ))
	#flux_int_err = 2 * np.pi * np.sqrt(np.sum((w * mu_centers * combined_I_err)**2))

	f4_values = fourth_derivative(f, h)

	if show_tables == 1:  # Debug Line
		print()  # Debug Line
		print('Data for Flux Error')  # Debug Line
		print('# w \t mu \t sigma I ')  # Debug Line
		for i in range(len(f)):  # Debug Line
			print(f"{i} {w[i]:.4f} \t {mu_centers[i]:.4f} \t {I_error[i]:.4f} ")  # Debug Line

	max_f4 = np.max(np.abs(f4_values))
	n = len(mu_centers) - 1
	b = mu_centers[nbins-1]
	a = mu_centers[0]
	simp_error = 2*np.pi*(b-a)*(1/180)*max_f4*(h**4)
	
	if show_tables == 1:	  # Debug Line
		print()	  # Debug Line
		print('Data for Simpson Error')  # Debug Line
		print('# I(mu)  \tmu \tf \t\tf⁴(μ)')  # Debug Line
		for i in range(len(f)):  # Debug Line
			print(f"{i} {I_of_mu[i]:.4f} \t{mu_centers[i]:.4f} \t{f[i]:.4f}  \t{f4_values[i]:.4f}")  # Debug Line

	flux_err = np.sqrt(flux_int_err**2 + simp_error**2)
	
	if show_tables == 1:	
		print()
		print(f'Bins: {nbins} \t\t a: {a} \t\t b: {b}')  # Debug Line
		print(f'h: {h} \t\t h⁴: {h**4}')  # Debug Line
		print(f'Max Abs f⁴(μ): {np.max(np.abs(f4_values))}')  # Debug Line
		print(f'Flux Value: {flux} | 2 pi 0∫1 I(μ) μ dμ ')  # Debug Line
		print(f'Flux Error: {flux_int_err} | 2 pi sqrt( sum( (w * mu * I_error)**2)))')  # Debug Line
		print(f'Simp Error: {simp_error} | 2 pi (b-a) (1/180) max(abs(f⁴(μ))) (h⁴)')  # Debug Line
		print(f'Total Error: {flux_err}')
		print('----------------------------------------')  # Debug Line
		
	return flux, flux_err
	
# Below Function is Deprecated / Not Called
def integrate_I_mu_gaussian_quad(I_of_mu, mu_centers, I_error):
	print(f'Solving Integral by Gaussian Quadrature')
	n = len(mu_centers)
	delta_mu = 1.0 / n
	factor = 2 * np.pi
	
	integrand = I_of_mu * mu_centers
	integral = factor * np.sum(integrand) * delta_mu

	abs_error_meas = factor * np.sqrt(np.sum((mu_centers * delta_mu * I_error) ** 2))
	rel_error_meas = abs_error_meas / integral

	f_mu = factor * integrand 

	second_deriv = np.zeros(n)
	for i in range(1, n - 1):
		second_deriv[i] = (f_mu[i+1] - 2*f_mu[i] + f_mu[i-1]) / (delta_mu ** 2)

	max_fpp = np.max(np.abs(second_deriv[1:-1]))
	abs_error_quad = (1.0 / (24 * n**2)) * max_fpp
	rel_error_quad = abs_error_quad / integral

	abs_error_total = np.sqrt(abs_error_meas**2 + abs_error_quad**2)
	rel_error_total = abs_error_total / integral
	if show_tables == 1:	  # Debug Line
		print(f'Bins: {n} | a {mu_centers[len(mu_centers)-1]} | b {mu_centers[0]}')
		print(f'Flux Value: {integral}')
		print(f'Measured Error: {abs_error_meas}')
		print(f'Quad Error: {abs_error_quad}')
		print(f'Flux Error: {abs_error_total}')
		print()
	return integral, rel_error_total

# Below Function is Deprecated / Not Called
def rebin_fits_2x2_sum(file, header=None):

	if is_path_or_data(file) == 'path':

		with fits.open(file) as hdul:
			data = hdul[0].data
			header = hdul[0].header

	else:
		data = file
		header = {
		    "NAXIS1": data.shape[0],
		    "NAXIS2": data.shape[1],
		    "CRPIX1": 0,
		    "CRPIX2": 0,
		    "CDELT1": 0,
		    "CDELT2": 0,
		    "R_SUN" : 0
		}
	 
	# Check if the dimensions are even
	ny, nx = data.shape
	if ny % 2 != 0 or nx % 2 != 0:
		raise ValueError("Image dimensions must be even for 2x2 rebinning.")
	if debug:
		print(">>>>", file)
	# Reshape and sum 2x2 blocks
	rebinned_data = data.reshape(ny//2, 2, nx//2, 2).sum(axis=(1, 3))
	# Update header (you might want to adjust pixel scale and CRPIX if needed)
	if 'NAXIS1' in header:
		header['NAXIS1'] = rebinned_data.shape[1]
		header['NAXIS2'] = rebinned_data.shape[0]
	if 'CRPIX1' in header:
		header['CRPIX1'] /= 2.0
	if 'CRPIX2' in header:
		header['CRPIX2'] /= 2.0
	if 'CDELT1' in header:
		header['CDELT1'] *= 2.0
	if 'CDELT2' in header:
		header['CDELT2'] *= 2.0
	if 'R_SUN' in header:
		header['R_SUN'] /= 2.0

	if 'CUNIT1' not in header or 'CUNIT2' not in header:
		header['CUNIT1'] = 'arcsec'
		header['CUNIT2'] = 'arcsec'
	if 'CTYPE1' not in header or 'CTYLE2' not in header:
		header['CTYPE1'] = 'HPLN-TAN'
		header['CTYPE2'] = 'HPLT-TAN'
	if 'CDELT1' not in header or 'CDELT2' not in header:
		header['CDELT1'] = 0.698
		header['CDELT2'] = 0.698

	return rebinned_data,header


def is_path_or_data(var):
	if isinstance(var, str) and os.path.exists(var):
		return 'path'
	elif isinstance(var, (np.ndarray, list, tuple, float, int)):
		return 'data'
	else:
		return 'unknown'

# Below Function is Deprecated / Not Called
def fourth_derivative(f, h):
	"""
	Estimate the 4th derivative using 5-point finite difference.
	Returns an array of f'''' at valid interior points.
	"""
	f4 = np.zeros(len(f))
	for i in range(2, len(f) - 2):
		f4[i] = (f[i + 2] - (4*f[i + 1]) + 6*f[i] -4*f[i - 1] + f[i - 2]) / (h ** 4)
		#f4[i] = (-f[i-2]+ (16*f[i-1]) -(30*f[i]) + (16*f[i+1])-f[i+2]) / (h**4)
	#return (f[:-4] - 4*f[1:-3] + 6*f[2:-2] - 4*f[3:-1] + f[4:]) / (h**4)
	return f4

