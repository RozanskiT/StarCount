#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
(Polish version below)
Simple script for counting stars in hemisphere,
based on several images.

Example usage with images in example_data:
python StarCount.py example_data/picture_*.JPG

------------------------------------------------
Skrypt zliczający gwiazdy widoczne na sferze
niebieskiej na podstawie zliczeń w kilku obrazach.

Użycie ze zdjęciami w folderze example_data:
python StarCount.py example_data/picture_*.JPG

"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from PIL import Image
from astropy.stats import mad_std, sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus
from photutils import aperture_photometry

def read_image(file_name):
    image = Image.open(file_name)
    image = np.sum(image, axis=2)
    return image.astype(np.float64)

def get_circular_mask(data):
    mask = np.zeros(data.shape, dtype=bool)
    r2 = (np.min(data.shape)/4.)**2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (i-int(data.shape[0]/2))**2 + (j-int(data.shape[1]/2))**2 < r2:
                mask[i,j] = True
    return ~mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='+',
                        help="Pictures used for star counting")
    args = parser.parse_args()

    minimum_luminosity = None
    images = []
    counts = []
    for file_name in args.file:
        image = read_image(file_name)
        images.append(image)

    for i, image in enumerate(images):
        std = np.std(image)
        daofind = DAOStarFinder(fwhm=5.0, threshold=6.*std)
        mask = get_circular_mask(image)
        sources = daofind(image - np.median(image),mask=mask)
        # for col in sources.colnames:
        #     sources[col].info.format = '%.8g'  # for consistent table output
        # print(sources)

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        aperture = CircularAperture(positions, r=5)
        annulus_aperture = CircularAnnulus(positions, r_in=6., r_out=8.)
        annulus_masks = annulus_aperture.to_mask(method='center')

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        bkg_median = np.array(bkg_median)
        phot = aperture_photometry(image, aperture)
        phot['annulus_median'] = bkg_median
        phot['aper_bkg'] = bkg_median * aperture.area()
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
        for col in phot.colnames:
            phot[col].info.format = '%.8g'  # for consistent table output
        print(phot)

        if i==0:
            fig, ax = plt.subplots()
            ax.imshow(image,
                      cmap=plt.get_cmap('binary'),
                      norm=colors.PowerNorm(gamma=1./2.)
                     )
            aperture.plot(color='b', lw=1, alpha=0.5)
            annulus_aperture.plot(color='r', lw=1, alpha=0.5)
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()


def onclick(event):
    tb = plt.get_current_fig_manager().toolbar
    if event.button==1 and event.inaxes and tb.mode == '':
        # event.x, event.y
        # minimum_luminosity
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        plt.close()




if __name__ == '__main__':
	main()
