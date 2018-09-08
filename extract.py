from __future__ import print_function

import os
from multiprocessing import Process, JoinableQueue

try:
    import cPickle as pickle
except:
    import pickle

from openslide import open_slide
import numpy as np
import h5py
import math
import scipy.misc

MAX_SAMPLE_PER_BATCH_FILE = 500


# threshold at grayscale to determine background
# todo: this algorithm is not reliable
def check_tissue_region(patch):
    patch = np.asarray(patch.getdata())[:, 0]
    val = np.histogram(patch, bins=[100, 235, 255])[0]
    if val[0] < val[1]:
        return False
    else:
        return True


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, filename, tile_size, check_tissue_bool, id, format="h5"):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self.id = id
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._slide = None
        self.save_out = filename + "_" + str(self.id)

        self.file_count = 0
        self.data_arr = []
        self.check_bool = check_tissue_bool
        self.format = format

    def run(self):
        self._slide = open_slide(self._slidepath)
        while True:
            data = self._queue.get()
            if data is None:
                self.save()
                self._queue.task_done()
                break

            level, address, last = data
            tile = self._slide.read_region(address, level, self._tile_size)
            if check_tissue_region(tile) or self.check_bool == False:
                if last % (MAX_SAMPLE_PER_BATCH_FILE-1) == 0:
                    self.store(np.asarray(tile)[:, :, :-1])
                    self.save()
                else:
                    self.store(np.asarray(tile)[:, :, :-1])
                self._queue.task_done()
            else:
                self._queue.put((level, address, last))

    def store(self, data):
        if self.data_arr == []:
            self.data_arr = data[np.newaxis, ...]
        else:
            self.data_arr = np.append(self.data_arr, data[np.newaxis, ...], axis=0)

    def save(self):
        filename = self.save_out + "_" + str(self.file_count) + "." + self.format
        if len(self.data_arr) > 0:
            self.data_arr = self.data_arr.transpose(0, 3, 1, 2)

            if self.format == "h5":
                with h5py.File(filename, "w") as hf:
                    hf.create_dataset("x", data=self.data_arr.astype('float32') / 255.0)
            elif self.format == "numpy":
                np.savez_compressed(filename, x=self.data_arr.astype('float32') / 255.0)
            elif self.format == "jpg":
                for idx in range(self.data_arr.shape[0]):
                    filename = self.save_out + "_" + str(idx) + "." + self.format
                    scipy.misc.toimage(self.data_arr[idx].transpose(2,0,1), cmin=0.0, cmax=1.0).save(filename)
            else:
                print("Unrecognized file format.")

            self.file_count += 1
            self.data_arr = []


'''
TissueLocator has three settings:
	- "all": all tiles are extracted from the slide
	- "random": a random subset of tiles are extracted from the slide; num_tiles_per slide must be provided.
	- "mask": a mask is provided which determines where tiles should be extracted; mask must be provided
'''
class TissueLocator():
    def __init__(self, slidepath, tile_size, mode="all", offset=None, level=0, mask=None, num_tiles_per_slide=None, region_locations=None, use_tissue_finder=False):

        self._tile_size = tile_size	#set the width/height of tile (tuple)
        self._slidepath = slidepath	#set the location of slide

        # save offset, if given
        if(offset == None):
            self.offset = tile_size
        else:
            self.offset = offset

        self._level = level  # 0 is the highest resolution and increments to lowest resolution

        self.mode = mode
        self._mask = mask
        self.region_locations = region_locations
        self.random_extract = num_tiles_per_slide
        self.use_tissue_finder = use_tissue_finder

    def get_coordinates_as_list(self, dims):
        list_points = []
        if self.mode == "all" or self.mode == "random":
            list_points = self.get_coordinates(dims[0], dims[1])

            # if we are only selecting a subset, then randomize and select first n points
            if self.mode == "random":
                np.random.shuffle(list_points)

        if self.mode == "mask" and self._mask is not None:
            mask_scale = dims[1] / float(self._mask.shape[0])
            new_list_points = []
            for point in list_points:
                if self._mask[int(math.floor(point[0] / mask_scale)), int(math.floor(point[1] / mask_scale))] == 1:
                    new_list_points.append(point)
            list_points = np.array(new_list_points)
                print("Note: mask mode is yet to be fully tested.")

        return list_points

    # Added by Shazia
    # This method performs a pass to determine how many random regions are required to fulfil "num_required".
    # Only the tissue finder check is performed here to save time. Once the "valid" tissue-like regions are determined
    # they are returned to be used later for saving/extracting
    def get_list_of_random_points(self):
        _slide = open_slide(self._slidepath)
        dims = _slide.dimensions

        list_points = self.get_coordinates_as_list(dims)

        num_accumulated = 0
        new_list = []
        for random_idx in range(len(list_points)):
            loc = (list_points[random_idx, 0], list_points[random_idx, 1])
            this_tile = _slide.read_region(loc, self._level, self._tile_size)

            result = check_tissue_region(this_tile)
            if(result == True):
                num_accumulated += 1
                new_list.append(np.array([list_points[random_idx, 0], list_points[random_idx, 1]]))
            if(num_accumulated == self.random_extract):
                return np.array(new_list)

        return list_points

    # Extract patches and saves them externally to h5 files. This method uses threading in order to speed up the tile
    # extraction process, required for large images
    def extract_patches_and_save(self, out_location, workers=2, export_format="h5", list_points=None):

        self._filename = os.path.basename(self._slidepath)[:-4]
        self._output = out_location
        if not os.path.exists(self._output):
            os.makedirs(self._output)

        # create a queue of workers to increase processing speed
        queue = JoinableQueue(workers)
        pool = [TileWorker(queue, self._slidepath, self._output + "/" + self._filename, self._tile_size, self.use_tissue_finder, _i, export_format)
                for _i in range(workers)]
        for thread in pool:
            thread.start()

        _slide = open_slide(self._slidepath)
        dims = _slide.dimensions
        _slide.close()

        # determine the patches to be extracted by openslide
        if list_points is None:
            list_points = self.get_coordinates_as_list(dims)

        i = 0
        stop_criteria = len(list_points)
        if (self.mode == "random"):
            stop_criteria = self.random_extract

        #todo: fix problem with randomly selected patches, we can't keep track of number extracted in queue WITH tissue finder

        while (i < stop_criteria):
            loc = (list_points[i, 0], list_points[i, 1])

            # pushes the location and size of tile to queue
            queue.put((self._level, loc, stop_criteria - i))

            i += 1

        #forces a last save before ending all workers
        for _j in range(len(pool)):
            queue.put(None)
        queue.join()

    # Instead of queuing patch extraction jobs (extract_patches_and_save), this method is simply a tile extractor which
    # processes one tile at a time
    def get_tissue_patches(self, list_points=None):
        #open the image
        _slide = open_slide(self._slidepath)
        dims = _slide.dimensions

        # determine the patches to be extracted by openslide
        if list_points is None:
            list_points = self.get_coordinates_as_list(dims)

        i, loc_idx = 0, 0
        tiles = []
        stop_criteria = len(list_points)
        if(self.mode == "random"):
            stop_criteria = self.random_extract

        while (i < stop_criteria):
            loc = (list_points[loc_idx, 0], list_points[loc_idx, 1])

            # select each tile one at a time and append to numpy array
            # this takes longer that the queuing and storing externally (extract_tissue) however may be required in a pipeline
            this_tile = _slide.read_region(loc, self._level, self._tile_size)
            if self.use_tissue_finder == False or check_tissue_region(this_tile):
                # flip the last channel and reshuffle dim order
                tiles.append(np.asarray(this_tile)[:, :, :-1].transpose(2, 0, 1).astype('float32') / 255.0)
                i += 1

            loc_idx += 1

        _slide.close()
        return np.array(tiles)

    # Creates a regular grid of points and saves into a list of [x, y] coordinates
    def get_coordinates(self, width, height):
        x_ind = np.arange(0, width - self._tile_size[0], self.offset[0])
        y_ind = np.arange(0, height - self._tile_size[1], self.offset[1])

        xv, yv = np.meshgrid(x_ind, y_ind)
        list_points = np.asarray([xv.ravel(), yv.ravel()])
        list_points = list_points.transpose(1, 0)

        return list_points
