import numpy as np


class OFASearchSpaceWithExpansions:
    """
    Defines the OFA net search space. The enconded representation of an
    architecture presents an initial vector with the settings for the model, plus
    a few extra bits for the expansion direction.

    ## Arguments
    These refer to the attributes for the "Maximal Possible Model" under this
    search space.

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
        Randombly sample an architecture from the constraint OFA.

        The maximal model we can sample is constraint, so the sampled
        model + the scaling direction always lies inside the OFASearchSpace.

        The direction is sampled in a way that tells how much we are moving
        in the space of `block_X`.

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
            sampled_resolution = np.random.choice(_resolutions)

            # Sample the non-contraint parth of the model depth
            free_depths = np.random.choice(_depths, _blocks - 1, replace=True)
            free_widths = np.random.choice(_widths, sum(free_depths), replace=True)
            free_ksizes = np.random.choice(_ksizes, sum(free_depths), replace=True)

            # Sample the last block ensuring it can be expanded.
            cdepth = np.random.choice(_depths[:-1], 1, replace=True)
            cwidths = np.random.choice(_widths[:-1], sum(cdepth), replace=True)
            cksizes = np.random.choice(_ksizes[:-1], sum(cdepth), replace=True)

            # Concatenate to add the final sampled model
            sampled_depths = np.concatenate([free_depths, cdepth])
            sampled_widths = np.concatenate([free_widths, cwidths])
            sampled_ksizes = np.concatenate([free_ksizes, cksizes])

            # Add the scaling direction (depths, widths and ksizes)
            direction = np.random.choice([0, 1], size=3, replace=True)

            # Append the sampled architecture
            samples.append(
                {
                    "depths": sampled_depths.tolist(),
                    "ksizes": sampled_ksizes.tolist(),
                    "widths": sampled_widths.tolist(),
                    "resolution": int(sampled_resolution),
                    "direction": direction.tolist(),
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
        encoding += sample["direction"]
        return encoding

    def decode(self, sample):
        """
        Performs the opposite operation from encoding. Decodes the sample
        from a representative list of string values.

        ## Params:
        - `sample`: Is assumed to be in the form:

        `[...block1, ...block2, ..., ...block5, resolution]`
        """
        NON_DATA_BITS = 4

        # Computes the overall size of each block
        max_depth = max(self.block_depths)
        encoded_block_size = 2 * max_depth + 1

        # Reconstruct by taking each element in order
        depths, ksizes, widths = [], [], []
        for i in range(0, len(sample) - NON_DATA_BITS, encoded_block_size):
            n_layers = sample[i]
            depths.append(n_layers)

            ksizes.extend(sample[i + 1 : i + 1 + n_layers])
            widths.extend(sample[i + (1 + max_depth) : i + (1 + max_depth) + n_layers])

        resolution = self.input_resolutions[sample[-NON_DATA_BITS]]
        direction = sample[-(NON_DATA_BITS - 1) :]
        return {
            "depths": depths,
            "ksizes": ksizes,
            "widths": widths,
            "resolution": resolution,
            "direction": direction,
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

    def apply_scaling(self, base_sample, direction):
        """
        Applies the scaling direction to the base model.
        Note: Only from the architectural side (weights are handled later).
        """
        scaled_depths, scaled_ksizes, scaled_widths = [], [], []

        # Order of the encoded direction
        ddir, wdir, kdir = direction

        _pos = 0
        for depth in base_sample["depths"]:
            base_ksizes = base_sample["ksizes"][_pos : _pos + depth]
            base_widths = base_sample["widths"][_pos : _pos + depth]

            # Get in the indexing space for the scaling
            depth_idx = self.block_depths.index(depth)
            ksizes_idxs = [self.block_ksizes.index(k) for k in base_ksizes]
            widths_idxs = [self.block_widths.index(w) for w in base_widths]

            # Add the scaling in the indexing space
            scaled_depth_idx = min(depth_idx + ddir, len(self.block_depths) - 1)
            scaled_ksizes_idxs = [
                min(k + kdir, len(self.block_ksizes) - 1) for k in ksizes_idxs
            ]
            scaled_widhts_idxs = [
                min(w + wdir, len(self.block_widths) - 1) for w in widths_idxs
            ]

            # Compute the final scaled values
            _scaled_depth = self.block_depths[scaled_depth_idx]
            _scaled_ksizes = [self.block_ksizes[k] for k in scaled_ksizes_idxs]
            _scaled_widths = [self.block_widths[w] for w in scaled_widhts_idxs]

            # Add the new parameters if needed
            new_layers = _scaled_depth - depth
            _scaled_ksizes.extend([self.block_ksizes[0]] * new_layers)
            _scaled_widths.extend([self.block_widths[0]] * new_layers)

            # Store new values
            scaled_depths.append(_scaled_depth)
            scaled_ksizes.extend(_scaled_ksizes)
            scaled_widths.extend(_scaled_widths)

            _pos += depth

        # Convert back to values
        return {
            "depths": scaled_depths,
            "ksizes": scaled_ksizes,
            "widths": scaled_widths,
            "resolution": base_sample["resolution"],
        }
