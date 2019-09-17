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
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import distance

from PIL import Image
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus
from photutils import aperture_photometry

class StarCount(object):
    """
    Simple class for counting stars, brighter than
    the one pointed with cursor, in central circle
    (of diameter = half of size of shorted image edge)
    of several images of the same size in pixels.

    Uses aperture phototometry go get the flux,
    and DAOFIND algoritm to find the stars positions in image.
    """
    def __init__(self, FWHM, threshold, aperture_radius):
        self.file_names = []
        self.images = []
        self.sources_positions = []
        self.fluxes_in_all_images = []
        self.reference_star_idx = None

        self.FWHM = FWHM
        self.threshold = threshold
        self.aperture_radius = aperture_radius

        self._irad_annulus = self.aperture_radius + 3
        self._orad_annulus = self.aperture_radius + 8

        self.init_plot()

    def read_images(self, file_names):
        self.images = []
        self.file_names = []
        for file_name in file_names:
            image = Image.open(file_name)
            image = np.sum(image, axis=2)
            self.images.append(image.astype(np.float64))
            self.file_names.append(os.path.split(file_name)[-1])
        if self.images:
            std = np.std(self.images[0])
            self.daofind_ = DAOStarFinder(fwhm=self.FWHM, threshold=self.threshold*std)
            # Compute size of circle where stars will be counted and add it to plot
            im_shape = np.shape(self.images[0])
            r = np.min(im_shape)/4
            self.ax.add_patch(Circle((im_shape[1]/2.,im_shape[0]/2.), r, fill=False, edgecolor='k'))


    def find_sources_in_all_images(self):
        if self.images:
            for image in self.images:
                positions = self.get_positions(image)
                self.sources_positions.append(positions)
        else:
            print("First load images!")

    def do_aperture_flux_measurement_in_all_images(self):
        if self.images and  (len(self.images) == len(self.sources_positions)):
            self.fluxes_in_all_images = []
            for image, positions in zip(self.images, self.sources_positions):
                fluxes, backgrounds = self.aperture_photometry(image,
                                            positions,
                                            r=self.aperture_radius,
                                            r_int=self._irad_annulus,
                                            r_ext=self._orad_annulus)
                self.fluxes_in_all_images.append(fluxes)

    def aperture_photometry(self, image, positions, r, r_int, r_ext):
        # Define apertures and anulluses at all found sources:
        object_aperture = CircularAperture(positions, r)
        background_annulus = CircularAnnulus(positions, r_int, r_ext)

        # Calculate the flux in each aperture (star, and background, and subtract)
        backgound_solution = aperture_photometry(image, background_annulus)
        object_solution = aperture_photometry(image, object_aperture)
        background_per_pixel = backgound_solution["aperture_sum"] / background_annulus.area()
        object_solution["aperture_sum"] -= background_per_pixel * object_aperture.area()

        return np.array(object_solution["aperture_sum"]), background_per_pixel

    def check_adjustments(self):
        if self.images:
            image = self.images[0]
            positions = None
            if len(self.images) == len(self.sources_positions):
                positions = self.sources_positions[0]
            else:
                positions = self.get_positions(image)
            # ------------------------------------------------------------------
            object_aperture = CircularAperture(positions, self.aperture_radius)
            background_annulus = CircularAnnulus(positions, self._irad_annulus, self._orad_annulus)
            self.ax.imshow(image, norm=colors.PowerNorm(gamma=0.5),cmap='PuBu_r')
            object_aperture.plot(color='white', lw=1)
            background_annulus.plot(color='white', lw=1)
            self.interactive = False
            plt.show()

    def run_counting(self):
        if self.images:
            self.find_sources_in_all_images()
            self.do_aperture_flux_measurement_in_all_images()

            print("Choose the reference star on image that popped on the screen!")
            print("Use left mouse button. If you made a mistake simply click on correct star.")
            print("Close the window or hit Enter to proceed...\n")

            image = self.images[0]
            positions = self.sources_positions[0]
            object_aperture = CircularAperture(positions, self.aperture_radius)
            background_annulus = CircularAnnulus(positions, self._irad_annulus, self._orad_annulus)

            self.ax.imshow(image, norm=colors.PowerNorm(gamma=0.5),cmap='PuBu_r')
            object_aperture.plot(color='white', lw=1)
            background_annulus.plot(color='white', lw=1)
            self.interactive = True
            plt.show()

            # Continue when image is closed
            threshold_flux = self.fluxes_in_all_images[0][self.reference_star_idx]
            counts = self.count_stars(threshold_flux)
            for f,c in zip(self.file_names, counts):
                print("{:5d} counts in central circle of image : {}".format(c, f))
            print("\nAverage number of stars in central circle = {:.1f}\n".format(np.mean(counts)))

    def count_stars(self, threshold_flux):
        counts = []
        for positions, fluxes in zip(self.sources_positions, self.fluxes_in_all_images):
            counts.append(0)
            for p, f in zip(positions,fluxes):
                if self.is_in_central_circle(p) and f > threshold_flux:
                    counts[-1] +=1 # incremet last element
        return counts

    def is_in_central_circle(self, p):
        im_shape = np.shape(self.images[0])
        d = np.amin(im_shape)
        r2 = (d/4.)**2
        return ((p[0] - im_shape[0]/2.)**2 + (p[1] - im_shape[1]/2.)**2 ) < r2

    def get_positions(self, image):
        sources = self.daofind_(image - np.median(image))
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        return positions

    def closest_point(self, point, points):
        closest_index = distance.cdist([point], points).argmin()
        return closest_index, points[closest_index]

    def init_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.sc = self.ax.scatter([], [], color='red', marker='x', zorder=10)
        self.interactive = False
        self.fig.canvas.mpl_connect('button_press_event', self.click_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

    def click_callback(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if event.inaxes and tb.mode == '' and self.interactive:
            if event.button==1:
                self.reference_star_idx = None
                positions = self.sources_positions[0]
                idx, point_coordinates = self.closest_point([event.xdata,event.ydata], positions)
                self.reference_star_idx = idx
                self.sc.set_offsets(point_coordinates)
                self.fig.canvas.draw()
            else:
                pass

    def key_press_callback(self,event):
        if event.key == 'enter' and self.interactive:
            plt.close()


def main():
    # Configure command line arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument("file", nargs='+', type=str,
                        help="Pictures used for star counting")
    parser.add_argument("-t","--threshold", default=4.0, type=float,
                        help="Threshold in standard diviations used by DAOFIND algorithm")
    parser.add_argument("-f","--fwhm", default=3.0, type=float,
                        help="FWHM in pixels used by DAOFIND algorithm")
    parser.add_argument("-ar","--aperture_radius", default=4.0, type=float,
                        help="Radius in pixels of aperture used for aperture photometry")
    parser.add_argument("-i","--inspect_settings", action='store_true',
                        help="Displays single image to check the star finding algorithm accuracy and size of apertures")

    args = parser.parse_args()

    # Create StarCount object
    sc = StarCount(FWHM=args.fwhm,
                   threshold=args.threshold,
                   aperture_radius=args.aperture_radius
                   )
    sc.read_images(args.file)

    if args.inspect_settings:
        print("------------------------------ Inspect settings ------------------------------\n")
        sc.check_adjustments()
    else:
        print("------------------------------ Run analysis ------------------------------\n")
        sc.run_counting()

    print("================================== End ==================================\n")



if __name__ == '__main__':
	main()
