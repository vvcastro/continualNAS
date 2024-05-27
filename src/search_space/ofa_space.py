import numpy as np


class OFASearchSpace:
    """
    Defines the OFA net search space.
    Some standard descriptors:
    @params
    - `num_blocks`: max number of units in the CNN sequence.
    - `block_depths`: number of layers on each block.
    - `block_widths`: expansion rate for the number of channels on each layer.
    - `block_ksizes`: kernel size of each layer.
    - `input_resolutions`: varies the resolution of the input image
    """

    def __init__(self, family: str):
        self.family = family
        match family:
            case "mobilenetv3":
                self.num_blocks = 5
                self.block_depths = [2, 3, 4]
                self.block_widths = [3, 4, 6]
                self.block_ksizes = [3, 5, 7]
                self.input_resolutions = list(range(128, 224, 4))
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
        Randombly sample an architecture from the maximal OFA.
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
        for _ in range(n_samples):
            sampled_depth = np.random.choice(_depths, _blocks, replace=True)
            sampled_resolution = np.random.choice(_resolutions)

            # The other arguments are sampled depth dependent
            n_layers = sum(sampled_depth)
            sampled_widths = np.random.choice(_widths, n_layers, replace=True)
            sampled_ksizes = np.random.choice(_ksizes, n_layers, replace=True)

            # Append the sampled architecture
            samples.append(
                {
                    "depths": sampled_depth.tolist(),
                    "ksizes": sampled_ksizes.tolist(),
                    "widths": sampled_widths.tolist(),
                    "resolution": int(sampled_resolution),
                }
            )
        return samples

    def encode(self, sample):
        """
        Performs the fix-length encoding to a integer string of a sample.
         - Made some changes for readibility of the encodings ( and to follow
         the paper example ).
        @params
        - `sample`: A dict of shape {
            'resolution': int,
            'depths': list(int),
            'ksizes': list(int),
            'widths': list(int)
        }

        ## Returns:
        A list with "6-blocks" (5 + input).
        Each block is represented by 9 sequential elements:

        - `block-depth (x1) | padded( ksizes ) (x4) | padded(exp_rates) (x4)`

        The overall structure is:
        `block1 | block2 | block3 | block4 | block5 | input_rest`
        """
        encoding = []

        # Pad the structures to ensure same length
        ksizes = self.zero_padding(sample["ksizes"], sample["depths"])
        widths = self.zero_padding(sample["widths"], sample["depths"])

        max_depth = max(self.block_depths)
        for i in range(self.num_blocks):
            encoding += [sample["depths"][i]]
            encoding += ksizes[i * max_depth : (i + 1) * max_depth]
            encoding += widths[i * max_depth : (i + 1) * max_depth]

        encoding += [self.input_resolutions.index(sample["resolution"])]
        return encoding

    def decode(self, sample):
        """
        Performs the opposite operation from encoding. Decodes the sample
        from a representative list of string values.

        ## Params:
        - `sample`: Is assumed to be in the form:

        `[...block1, ...block2, ..., ...block5, resolution]`
        """

        # Computes the overall size of each block
        max_depth = max(self.block_depths)
        encoded_block_size = 2 * max_depth + 1

        # Reconstruct by taking each element in order
        depths, ksizes, widths = [], [], []
        for i in range(0, len(sample) - 1, encoded_block_size):
            n_layers = sample[i]
            depths.append(n_layers)

            ksizes.extend(sample[i + 1 : i + 1 + n_layers])
            widths.extend(sample[i + (1 + max_depth) : i + (1 + max_depth) + n_layers])

        resolution = self.input_resolutions[sample[-1]]
        return {
            "depths": depths,
            "ksizes": ksizes,
            "widths": widths,
            "resolution": resolution,
        }

    def initialise(self, n_samples):
        data = [self._get_min_sample(), self._get_max_sample()]
        data.extend(self.sample(n_samples=(n_samples - 2)))
        return data

    def _get_min_sample(self):
        """
        Return the smallest possible architecture given the
        initialisation family.
        """
        return self.sample(
            n_samples=1,
            depths=[min(self.block_depths)],
            widths=[min(self.block_widths)],
            ksizes=[min(self.block_ksizes)],
            resolutions=[min(self.input_resolutions)],
        )[0]

    def _get_max_sample(self):
        """
        Return the smallest possible architecture given the
        initialisation family.
        """
        return self.sample(
            n_samples=1,
            depths=[max(self.block_depths)],
            widths=[max(self.block_widths)],
            ksizes=[max(self.block_ksizes)],
            resolutions=[max(self.input_resolutions)],
        )[0]

    def zero_padding(self, values: list, depths: list):
        """
        Pads the given values to the max depth available for the OFA model.
        ## Params
        - `values`: A flattened list with the values for each of the layers.
        - `depths`: A list with the depth of each block.

        ### Example:
        ```python
        depths = [2, 3, 1]
        values = [3, 5, 3, 5, 7, 3]
        expected = [3, 5, 0, 0, 3, 5, 7, 0, 3, 0, 0, 0]
        ```
        """
        padded_values, position = [], 0
        for d in depths:
            for _ in range(d):
                padded_values.append(values[position])
                position += 1
            padded_values += [0] * (max(self.block_depths) - d)
        return padded_values
