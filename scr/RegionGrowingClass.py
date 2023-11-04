from skimage.feature import peak_local_max
import numpy as np
from collections import deque


class RegionGrowingPython:

    def __init__(self, image, seeds_lth=0, seeds_hth=255, seeds_dist=10):
        """
        Initialize the class by giving the initial seeds and the input image
        @:param image: Input image
        @:type numpy.ndarray

        @:param seeds: Input seeds, must be a list of tuples
        @:type list[tuple]

        @:param seeds_lth: Lower threshold for automatic seeds selection
        @:type int or float

        @:param seeds_hth int or float: Higher threshold for automatic seeds selection
        @:type int or float

        @:param seeds_dist: Minimum distance between seeds for automatic seeds selection
        @:type int
        """
        self.image = image
        self.seeds = RegionGrowingPython.set_seeds(self, seeds_lth, seeds_hth, seeds_dist)

    """
    Automatically choose the seeds based on the histogram of you image. My image had two region with a slight 
    pixel intensity difference. This function uses a single intensity threshold to create a binary image mask, 
    where pixels with intensity above the threshold are set to 1 and the others to 0. Then, labelled connected 
    components in the mask and calculated the centroid of each component. The centroids are used as seeds for the 
    growth of the region, and a minimum distance between the seeds is applied to prevent them from being too close to 
    each other.
    """
    def set_seeds(self, low_threshold, high_threshold, min_distance):
        """
        Automatically set seeds for region growing based on intensity thresholds.

        @param low_threshold: The lower bound for the intensity threshold. Pixels with intensity equal to or higher than
         this value will be considered potential seed points.
        @type low_threshold: int or float

        @param high_threshold: The upper bound for the intensity threshold. Pixels with intensity equal to or higher
        than this value but less than the high threshold will be considered for seed selection.
        @type high_threshold: int or float

        @param min_distance: The minimum allowed distance between any two seeds. This parameter ensures that seeds are
        sufficiently spaced out in the image.
        @type min_distance: int
        """
        image = self.image
        # Identify regions of interest based on intensity thresholds
        bone_region = np.where(image >= high_threshold, 1, 0)
        jaw_region = np.where((image >= low_threshold) & (image < high_threshold), 1, 0)

        # Find local maxima in each region as seeds
        bone_seeds = peak_local_max(image, min_distance=min_distance, labels=bone_region)
        jaw_seeds = peak_local_max(image, min_distance=min_distance, labels=jaw_region)

        # Combine seeds from both regions
        seeds = list(map(tuple, np.vstack((bone_seeds, jaw_seeds))))

        return seeds

    """
    This function iterates over the seeds that the __init()__ method got, calculating the different regions for each of
    them and combining the results to compute the final image.
    """
    def region_grow(self, threshold, std_max_variation):
        """
        Implement Region Growing Algorithm for multiple seeds, treating each seed separately.

        @param threshold: The intensity difference threshold for growing the region. This value determines how different
        a pixel's intensity can be from the seed point's intensity for it to be considered part of the region.
        @type threshold: int or float

        @param std_max_variation: The maximum allowed standard deviation increase when adding a new pixel to the region.
        This parameter helps in controlling the homogeneity of the grown region.
        @type std_max_variation: float

        @:return segmented: 2D array with the combined segmented regions.
        @:type numpy.ndarray
        """
        image = self.image
        seed_points = self.seeds
        # Initialize segmented output image
        final_segmented = np.zeros_like(image, dtype=np.uint8)

        for seed in seed_points:
            # Segment the image based on the current seed
            segmented = RegionGrowingPython.region_growing_adaptive(self, seed, threshold, std_max_variation)
            # Combine the results
            final_segmented = np.logical_or(final_segmented, segmented)

        return final_segmented.astype(np.uint8)

    """
    This function applies an adaptive region growing algorithm to segment an image based on local pixel 
    intensities and standard deviation variations. It starts from a given seed point and expands to adjacent pixels, 
    incorporating those that do not exceed a specified intensity threshold and whose inclusion does not cause the 
    standard deviation of the region's intensities to increase beyond a defined limit. This process continues 
    iteratively, assessing the local neighborhood of each new pixel added to the region, until no further pixels meet 
    the criteria for inclusion. The result is a segmented binary image where the region of interest connected to the 
    seed point is differentiated from the background.
    """
    def region_growing_adaptive(self, seed_point, initial_threshold, max_std_dev_increase):
        """
        Implement Adaptive Region Growing Algorithm based on pixel intensity and local standard deviation
        with optimizations to improve efficiency.

        @param seed_point: The coordinates (x, y) of the initial seed point from which to start growing the region.
        @type seed_point: tuple

        @param initial_threshold: The starting intensity difference threshold for growing the region. As the region
        grows, this threshold can adapt based on local variations.
        @type initial_threshold: int or float

        @param max_std_dev_increase: The maximum permissible increase in the standard deviation of the region's
        intensity values when a new pixel is added. This ensures the region maintains a degree of intensity homogeneity.
        @type max_std_dev_increase: float


        @:return segmented: 2D array with the segmented region.
        @:type numpy.ndarray
        """
        image = self.image.astype(np.float64)

        # Initialize segmented output image
        segmented = np.zeros_like(image, dtype=np.uint8)

        # Initialize a deque with seed point
        seed_deque = deque([seed_point])

        # Get seed point intensity and convert to float
        seed_intensity = float(image[seed_point])

        # Initialize region intensity list as deque for efficient operations
        region_intensities = deque([seed_intensity])

        # Initialize region intensity sum and sum of squares for efficient standard deviation calculation
        region_intensity_sum = seed_intensity
        region_intensity_sum_sq = seed_intensity ** 2

        # Loop until no more pixels to be added
        while seed_deque:
            s = seed_deque.popleft()
            x, y = s

            # Check if the point is already segmented
            if segmented[x, y] == 1:
                continue

            # Mark the point as segmented
            segmented[x, y] = 1

            # Loop through the 8-neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy

                    # Check if the neighbor is within bounds
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        neighbor_intensity = float(image[nx, ny])

                        # Calculate local mean and standard deviation within a window around the neighbor pixel
                        window = image[max(0, nx - 1):min(image.shape[0], nx + 2), max(0, ny - 1):min(image.shape[1], ny + 2)]
                        local_mean = np.mean(window)

                        # Calculate new sum and sum of squares for efficient standard deviation calculation
                        new_region_intensity_sum = region_intensity_sum + neighbor_intensity
                        new_region_intensity_sum_sq = region_intensity_sum_sq + neighbor_intensity ** 2

                        # Calculate new standard deviation of the region
                        n = len(region_intensities) + 1
                        new_std_dev = np.sqrt((new_region_intensity_sum_sq - (new_region_intensity_sum ** 2) / n) / n)

                        # Check if the neighbor should be added based on the intensity difference and standard deviation
                        if abs(local_mean - neighbor_intensity) <= initial_threshold and new_std_dev <= max_std_dev_increase:
                            seed_deque.append((nx, ny))
                            region_intensities.append(neighbor_intensity)
                            region_intensity_sum = new_region_intensity_sum
                            region_intensity_sum_sq = new_region_intensity_sum_sq

        return segmented.astype(np.uint8)

