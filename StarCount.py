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
import numpy as np

from PIL import Image
from astropy.stats import mad_std
from photutils import DAOStarFinder

def read_image(file_name):
    image = Image.open(file_name)
    image = np.sum(image, axis=2)
    return image.astype(np.float64)


    # image -= np.median(image)
    # bkg_sigma = mad_std(image)
    # daofind = DAOStarFinder(fwhm=fwhm, threshold=thresholdSigmaNo*bkg_sigma)
    # sources = daofind(image).to_pandas()
    # sources.sort_values("flux", axis=0, ascending=False, inplace=True)
    # sources = sources.head(3*topNo)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='+',
                        help="Pictures used for star counting")
    args = parser.parse_args()

    images = []
    for file_name in args.file:
        image = read_image(file_name)
        images.append(image)

    for image in images:
        std = np.std(image)
        daofind = DAOStarFinder(fwhm=5.0, threshold=6.*std, ratio=0.3, theta=0)
        sources = daofind(image - np.median(image))
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
        print(sources)

        from astropy.visualization import SqrtStretch
        from astropy.visualization.mpl_normalize import ImageNormalize
        from photutils import CircularAperture

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=4.)
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(image, cmap='Greys', origin='lower', norm=norm)
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
        plt.show()




if __name__ == '__main__':
	main()
