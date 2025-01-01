class CacheConf:
    """
    Cache configurations (checkpoints)
    These are the checkpoints that we used in our experiments,
    documented in Table II of our paper. The corresponding results for each
    configuration can be found in Table III.

    CONFIG_3 demonstrated the best performance in our experiments.
    For detailed analysis, please refer to our paper.

    Example usage:
        from ca_tome.utils import CacheConf
        conf = CacheConf.ADAPTIVE
    """
    CONFIG_1 = [0, 1, 2, 3, 5, 10, 15, 25, 35]
    CONFIG_2 = [0, 10, 11, 12, 15, 20, 25, 30, 35, 45] 
    CONFIG_3 = [0, 8, 11, 13, 20, 25, 30, 35, 45, 46, 47, 48, 49]  # Best performing CA-ToMe configuration
    CONFIG_4 = [0, 9, 13, 14, 15, 28, 29, 32, 36, 45]
    ADAPTIVE = [0, 1, 5, 7, 10, 12, 15, 35, 40, 45, 46, 47, 48, 49, 50, 51]
    CONFIG_Five = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
