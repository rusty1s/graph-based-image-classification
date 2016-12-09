# from .segment import Segment


# def get_image_from_superpixels(segments):
#     """S"""
#     # Compute segments with all its attributes.
#     segments = Segment.generate(image, superpixels)

#     # Create a blank new image.
#     height, width = superpixels.shape
#     superpixel_image = np.zeros((height, width, 3), np.uint8)

#     # Iterate over all segment values.
#     for s in segment.values():
#         mask = s.mask
#         mean = s.mean

#     @staticmethod
#     def write(segments, image_name, suffix=None):
#         """Writes the generated dictionary of segments to disk."""

#         # Compute the correct file name.
#         suffix = '' if suffix is None else '-{}'.format(suffix)
#         file_name = '{}_segments{}.pkl'.format(image_name, suffix)

#         with open(file_name, 'wb') as output:
#             pickle.dump(segments, output, -1)

#     @staticmethod
#     def read(file_name):
#         """Reads the written dictionary of segments from disk. The file name
#         shouldn't have a file extension."""

#         with open('{}.pkl'.format(file_name), 'rb') as input:
#             return pickle.load(input)

#     @staticmethod
