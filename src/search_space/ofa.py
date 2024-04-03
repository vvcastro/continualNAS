import numpy as np


class OFASearchSpace:
    """ 
    Holds the wrapper for the OFA net search space.
    Some standard descriptors:
    @params
    - `num_blocks`: max number of units in the CNN sequence.
    - `block_depths`: number of layers on each block.
    - `block_widths`: expansion rate for the number of channels on each layer.
    - `block_ksizes`: kernel size of each layer.
    - `input_resolutions`: varies the resolution of the input image
    """

    def __init__(self, family: str):
        match family:
            case 'mobilenetv3':
                self.num_blocks = 5
                self.block_depths = [2, 3, 4]
                self.block_widths = [3, 4, 6]
                self.block_ksizes = [3, 5, 7]
                self.input_resolutions = list(range(192, 257, 4))
            case _:
                raise KeyError(f"OFA family type: '{family}' not implemented!")

    def sample(
        self,
        n_samples: int = 1,
        n_blocks: int | None = None,
        depths: int | None = None,
        widths: int | None = None,
        ksizes: int | None = None,
        resolutions: int | None = None,
    ):
        """ 
        Randombly sample one architecture from the maximal OFA setting loaded 
        when initialising the object. 
        @params:
        - (n_blocks, kernel_size, exp_rate, depth, resolution) 
        @returns
        - A list with the settings for each architecture.
        """

        # Variables to use
        _blocks = self.num_blocks if not n_blocks else n_blocks
        _depths = self.block_depths if not depths else depths
        _widths = self.block_widths if not widths else widths
        _ksizes = self.block_ksizes if not ksizes else ksizes
        _resolutions = self.input_resolutions if not resolutions else resolutions

        samples = []
        for n in range(n_samples):
            sampled_depth = np.random.choice(_depths, _blocks, replace=True)

            # The other arguments are sampled for the depth of each block.
            n_layers = sampled_depth.sum()
            sampled_widths = np.random.choice(_widths, size=n_layers, replace=True)
            sampled_ksizes = np.random.choice(_ksizes, size=n_layers, replace=True)
            sampled_resolution = np.random.choice(_resolutions)

            # Append the sampled architecture
            samples.append({
                'resolution': int(sampled_resolution),
                'depths': sampled_depth.tolist(),
                'ksizes': sampled_ksizes.tolist(),
                'widths': sampled_widths.tolist(),
            })
        return samples
