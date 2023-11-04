import numpy as np


"""
Created by Mario Pascual González, a bioinformatics student, as part of a project for the course "Biomedical 
Images". 

The SplitAndMerge class encapsulates an image segmentation algorithm using a split-and-merge technique 
adapted for biomedical image processing. The algorithm is based on quadtree decomposition, segmenting image regions 
according to specified intensity thresholds.

This class is open-source and may be freely used and modified. Users are encouraged to adapt and improve the 
functions herein to meet their specific requirements or to contribute to its evolution in the spirit of collaborative 
development.
"""


class SplitAndMerge:
    """
    The SplitAndMerge class encapsulates a segmentation algorithm for grayscale images based on quadtree
    decomposition, guided by intensity thresholds and region dimensions. Upon instantiation, the class initializes
    with an image and parameters including minimum and maximum intensity thresholds (min_th and max_th) for region
    segmentation and a minimum dimension (min_dim) for subregions in the quadtree. It ensures that these thresholds
    are within valid ranges and that the min_dim is feasible for splitting.
    """
    def __init__(self, image, min_th, max_th, min_dim):
        """
        Initialize the SplitAndMerge class.

        This constructor sets up the SplitAndMerge object with the necessary attributes for the image segmentation
        process. It validates and assigns the minimum and maximum thresholds for intensity values and the minimum
        dimension size for quadtree regions.

        :param image: Grayscale image to be segmented.
        :type image: numpy.ndarray

        :param min_th: Minimum threshold for intensity values to consider in the region of interest.
                       Must be non-negative.
        :type min_th: int

        :param max_th: Maximum threshold for intensity values to consider in the region of interest.
                       Must be within 0-255 range.
        :type max_th: int

        :param min_dim: Minimum dimension (width and height) for quadtree regions. Must be a positive integer.
        :type min_dim: int

        :raises ValueError: If thresholds are out of acceptable range or if min_dim is not positive.
        """
        self.image = image
        if min_th < 0:
            raise ValueError(f'Minimum threshold cant be lower than zero: {min_th}')
        elif max_th < 0:
            raise ValueError(f'Maximum threshold cant be lower than zero: {max_th}')
        elif max_th > 255:
            raise ValueError(f'Maximum threshold cant be higher than 255: {max_th}')
        elif max_th < min_th:
            raise ValueError(f'Maximum threshold cant be lower than the minimum: {max_th, min_th}')
        elif min_dim < 1:
            raise ValueError(f'The minimum dimension of each node cant be lower than 1x1: {min_dim}')
        self.min_th = min_th
        self.max_th = max_th
        self.min_dim = min_dim

    """
    The extract_statistical_info method processes the given image, identifying pixels within the specified 
    intensity thresholds to calculate and return the mean and standard deviation of these selected pixels, 
    which represent the region of interest, such as bone tissue in a medical image.
    """
    def extract_statistical_info(self):
        """
        Extract statistical information from the image within given intensity thresholds.

        This method identifies pixels within the specified intensity range and calculates the mean and standard
        deviation of these pixels' intensity values. The calculated values represent the statistical profile of the
        region of interest in the image, which is used to guide the segmentation process.

        :returns: Tuple containing the mean and standard deviation of the selected pixels' intensity.
        :rtype: tuple
        """

        img = self.image
        lower_bound = self.min_th
        upper_bound = self.max_th
        threshold_image = np.zeros_like(img)
        threshold_image[(img >= lower_bound) & (img <= upper_bound)] = 255  # Setting pixels in the ROI to white

        bone_pixels = img[threshold_image > 0]
        # Compute descriptive metrics for target tissue
        mean_intensity = np.mean(bone_pixels)
        std_intensity = np.std(bone_pixels)

        return mean_intensity, std_intensity

    """
    This function serves as a decision rule for the segmentation process. Given a region of the image and pre-computed mean and 
    standard deviation of the region of interest, it determines whether the region is likely to be of the target 
    tissue based on its intensity profile.
    """
    @staticmethod
    def predicate_func(region, mean_intensity, std_intensity):
        """
        Static predicate function to decide if a region is likely to be the target tissue.

        This function evaluates a region's mean and standard deviation against predefined metrics to determine if it
        matches the characteristics of the target tissue. It returns True if the region's intensity profile falls
        within the acceptable range dictated by the mean and standard deviation of the target tissue.

        :param region: Image region to evaluate.
        :type region: numpy.ndarray

        :param mean_intensity: Pre-computed mean intensity for the target tissue.
        :type mean_intensity: float

        :param std_intensity: Pre-computed standard deviation for the target tissue.
        :type std_intensity: float

        :returns: Boolean indicating whether the region is likely to be the target tissue.
        :rtype: bool
        """
        # Compute mean and standard deviation of the region's intensity
        region_mean = np.mean(region)
        region_std = np.std(region)

        # Criteria based on pre-computed descriptive metrics for bone tissue
        is_bone_tissue = (
                (region_mean > mean_intensity - std_intensity) and
                (region_mean < mean_intensity + std_intensity) and
                (region_std < std_intensity)
        )

        return is_bone_tissue

    """
    This is the core functionality of the class is implemented in the split_and_merge method, which orchestrates 
    the segmentation process. It starts by labeling different regions of the image using an initial label count. A 
    private, recursive helper function, recursive_split_and_merge, is defined within this method to handle the 
    splitting of the image into quadrants and their potential merging based on the predicate function's outcome. If a 
    region's intensity characteristics satisfy the predicate, it is marked with a unique label; otherwise, 
    it is split into smaller quadrants. This process continues until the minimum allowable region size (min_dim) is 
    reached. The resulting segmented_image is a labeled matrix where each uniquely identified region is marked with a 
    different integer, representing the segmented portions of the original image according to the specified criteria.
    """
    def split_and_merge(self):
        """
        Segment the image using the split-and-merge algorithm.

        This method applies the split-and-merge segmentation process to the image using a quadtree decomposition
        approach. It recursively divides the image into smaller regions and merges them based on the predicate
        function's decision. The segmented image has uniquely labeled regions that meet the criteria defined by the
        statistical information of the target tissue.

        :returns: A segmented image with uniquely labeled regions.
        :rtype: numpy.ndarray
        """

        image = self.image
        min_dim = self.min_dim
        # Initialize the segmented image with a larger integer type
        segmented_image = np.zeros_like(image, dtype=np.int32)
        label = 0  # Initial label
        # Compute the statistical characteristics of the tissue
        mean_intensity, std_intensity = SplitAndMerge.extract_statistical_info(self)

        def recursive_split_and_merge(x, y, dim):
            """
            Recursively apply the split-and-merge algorithm to the image region specified by
            the top-left corner (x, y) and dimension dim.
            """
            nonlocal label  # Access the label variable from the outer function

            # Extract the region of interest
            region = image[x:x + dim, y:y + dim]

            # Apply the predicate function to the region
            if SplitAndMerge.predicate_func(region, mean_intensity, std_intensity):
                # If the predicate is True, assign a label to this region
                segmented_image[x:x + dim, y:y + dim] = label
                label += 1  # Increment the label for the next region
            elif dim > min_dim:
                # If the predicate is False and the region is larger than min_dim,
                # split it into 4 sub-regions and apply the algorithm recursively
                new_dim = dim // 2  # Half the dimension
                for dx in [0, new_dim]:
                    for dy in [0, new_dim]:
                        recursive_split_and_merge(x + dx, y + dy, new_dim)

        # Start the recursive split-and-merge algorithm from the top-left corner of the image
        recursive_split_and_merge(0, 0, image.shape[0])

        return segmented_image
