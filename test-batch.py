'''
File: 	test-batch.py
Author: 	Shazia Akbar

Description:
Test file for reading patches on-the-fly for Keras. This example passes a low 
resolution mask when we use the extract tool so that we only extract patches 
containing tissue.
'''

import extract
import numpy as np
import matplotlib
from openslide import open_slide

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

# create a boolean mask (at resolution mask_level) which can be used to locate tissue in whole slide
# if you have more levels in the pyramid, I suggest increasing mask_level
def _get_tissue_mask(filename, mask_level=2):
    _slide = open_slide(filename)
    downscale_dims = _slide.level_dimensions[mask_level]
    downscale_image = np.asarray(_slide.read_region((0, 0), mask_level, downscale_dims)).astype(np.uint8)
    downscale_image = matplotlib.colors.rgb_to_hsv(downscale_image[:, :, :3])
    _slide.close()

    # otsu thresholding (using hue color channel)
    val = filters.threshold_otsu(downscale_image[:, :, 0])
    thresholded_mask = downscale_image[:, :, 0] > val

    tissue_mask = thresholded_mask.astype('bool')
    # unclear why the following needs to be performed...?
    tissue_mask = np.flipud(np.rot90(tissue_mask))
    return tissue_mask



# retrieve only patches containing tissue from filename alongwith the location of those patches
def get_tissue_patch_locations_from_tif(filename, tile_size):
    _slide = open_slide(filename)
    orig_dims = _slide.dimensions
    _slide.close()
    tissue_mask = _get_tissue_mask(filename)

    '''
    plt.subplot(121)
    plt.imshow(downscale_image[:,:,0])
    plt.subplot(122)
    plt.imshow(tissue_mask)
    plt.colorbar()
    plt.show()
    '''

    einst = extract.TissueLocator(filename, tile_size, mode="all", mask=tissue_mask)
    location_of_patches = einst.get_coordinates_as_list(orig_dims)
    return location_of_patches


def get_patch_from_locations(filename, locations, patch_size):
    einst = extract.TissueLocator(filename, patch_size, mode="all")
    image_as_patches = einst.get_tissue_patches(locations)
    return image_as_patches

test_file = "./svsdir/"

# retrieve a list of points that we wish to extract
all_points_as_list = get_tissue_patch_locations_from_tif(test_file, (512, 512))

#randomly shuffle list and iteratively extract patches on-the-fly
np.random.shuffle(all_points_as_list)
num_patches_per_loop = 10
for i in range(10):
	print 'loop', str(i)
	patches = get_patch_from_locations(test_file, all_points_as_list[i*num_patches_per_loop:min((i+1)*num_patches_per_loop, len(all_points_as_list))], (512, 512))

print("Completed experiment.")
