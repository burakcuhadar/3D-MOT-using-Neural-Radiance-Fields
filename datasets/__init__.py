from .blender import BlenderDataset
from .carla_static import CarlaStaticDataset
from .carla_star import CarlaStarDataset

dataset_dict = {'blender': BlenderDataset, 'carla_static': CarlaStaticDataset, 'carla_star': CarlaStarDataset}

