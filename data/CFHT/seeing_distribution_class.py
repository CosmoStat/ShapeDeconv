import numpy as np
class seeing_distribution(object):
    """ Seeing distribution
    Provide a seeing following CFIS distribution. Seeing generated from
    scipy.stats.rv_histogram(np.histogram(obs_seeing)). Object already
    initialized and saved into a numpy file.
    Parameters
    ----------
    path_to_file: str
        Path to the numpy file containing the scipy object.
    seed: int
        Seed for the random generation. If None rely on default one.
    """
    def __init__(self, path_to_file, seed=None):
        self._file_path = path_to_file
        self._load_distribution()
        self._random_seed = None
        if seed != None:
            self._random_seed = np.random.RandomState(seed)
    def _load_distribution(self):
        """ Load distribution
        Load the distribution from numpy file.
        """
        self._distrib = np.load(self._file_path, allow_pickle=True).item()
    def get(self, size=None):
        """ Get
        Return a seeing value from the distribution.
        Parameters
        ----------
        size: int
            Number of seeing value required.
        Returns
        -------
        seeing: float (numpy.ndarray)
            Return the seeing value or a numpy.ndarray if size != None.
        """
        return self._distrib.rvs(size=size, random_state=self._random_seed)
