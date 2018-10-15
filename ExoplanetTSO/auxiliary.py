def cross_correlation_HST_diff_NDR():
	# Cross correlated the differential non-destructive reads from HST Scanning mode with WFC3 and G141
	import image_registration as ir
	from glob import glob
	from pylab import *;ion()
	from astropy.io import fits

	fitsfiles = glob("*ima*fits")

	ylow   = 50
	yhigh  = 90
	nExts  = 36
	extGap = 5
	shifts_ndr = zeros((len(fitsfiles), (nExts-1)//extGap, 2))
	fitsfile0 = fits.open(fitsfiles[0])
	for kf, fitsfilenow in enumerate(fitsfiles):
			 fitsnow = fits.open(fitsfilenow)
			 for kndr in range(extGap+1,nExts+1)[::extGap][::-1]:
					 fits0_dndrnow = fitsfile0[kndr-extGap].data[ylow:yhigh]- fitsfile0[kndr].data[ylow:yhigh]
					 fitsN_dndrnow = fitsnow[kndr-extGap].data[ylow:yhigh]  - fitsnow[kndr].data[ylow:yhigh]
					 shifts_ndr[kf][(kndr-1)//extGap-1] = ir.chi2_shift(fits0_dndrnow, fitsN_dndrnow)[:2]
					 #ax.clear()
					 #plt.imshow(fitsN_dndrnow)
					 #ax.set_aspect('auto')
					 #plt.pause(1e-3)

	plot(shifts_ndr[:,:-1,0],'o') # x-shifts 
	plot(shifts_ndr[:,:-1,1],'o') # y-shifts
