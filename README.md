Author: Shazia Akbar
Date last updated: 10th April 2017

--------------------------------------------------------------------------- 
Prerequisites:
* Python (tested on version 2.7) 
* openslide
* h5py

--------------------------------------------------------------------------- 
### Description: 

Extract.py provides functionality for extracting patches from pathology slides
using openslide. Slides should be provided as .svs files and the location to
these files is determined when you set up an instance of the TissueLocator
class. 

Extracted files are saved in .h5 (by default) compressed files containing one 
variable, 'x'. By default patches are extracted on a regular grid defined by 
patch_size. There are three additional modes which may also use:

### Modes:
* ["all"] (default): all tiles are extracted from the slide
* ["random"]: a random subset of tiles are extracted from the slide; num_tiles_per slide must be provided.
* ["mask"]: a mask is provided which determines where tiles should be extracted; mask must be provided


Usage: To use TissueLocator, in the constructor provide the location of the slide to be processed and the size
of the tiles to extract (i.e. tile_size). You then have two options for extracting patches:
* ["extract_patches_and_save"]: Use this function if you would like to store the patches externally. By default .h5 files are generated but you can override this to save "numpy" or "jpg" files instead.
* ["get_tissue_patches"]: Use this function if you would like the patches to be returned as a numpy array; useful for a pipeline in which you want to call this method multiple times.

Here is a very simple example of how to use the code below:
```
    import extract 
    einst = extract.TissueLocator(svsfile, tile_size = (512, 512), mode="random", num_tiles_per_slide=102)
	einst.extract_patches_and_save(out_location = save_location)
```
    
To read the tiles back in again, simple load the h5 file as follows
```
	import h5py
	meta = h5py.File('name of .h5 file', 'r')
	patches = meta['x'][:]
	meta.close()
```

---------------------------------------------------------------------------
Using Tissue Finder code:

Functionality also exists for extracting regions containing tissue only. This is switched off by default. Set use_tissue_finder to True to enable this.

Note: if you are extracting random patches which contains tissue i.e. mode == "random" AND use_tissue_finder = True, then you must perform a check after setting up the constructor. An example is given below.

```
einst = extract.TissueLocator(filename, tile_size, mode="random", num_tiles_per_slide=num_patches, use_tissue_finder=True)
extracted_points = einst.get_coordinates_as_list(dims=(512, 512))

einst.extract_patches_and_save(out_location = save_location, workers=1, list_points=extracted_points)
```

--------------------------------------------------------------------------- 
### Additional parameters:

* [level]: If you would like to extract at the full resolution leave scale=1.0 (this hasn't been tested for other scales yet) 
* [offset]: If you don't want to extract tiles on a regular grid, set the step size for both x and y direction.
* [export_format]: format of images/files to be saved (default: "h5").
* [workers]: defines how many parallel processes are operatign when saving externally. Increase this to extract patches faster.
* [MAX_SAMPLE_PER_BATCH_FILE]: determines the maximum number of patches to be stored in a single .npz file. By default this is 500.

--------------------------------------------------------------------------- 
### Save in alternative formats: 
 If you don't wish to save the patches as .h5 files there is an option to  change this to numpy files. If you would like to implement your own method (e.g. to create images) overwride the save() function in TileWorker class.
