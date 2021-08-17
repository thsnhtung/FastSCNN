from .cityscapes import CitySegmentation
from .simulation import Simulation

datasets = {
    'citys': CitySegmentation,
    'simulation': Simulation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
