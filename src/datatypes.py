class NormalizedDict(dict):
    """
    A dictionary subclass that normalizes its keys by converting them to lowercase and replacing spaces with underscores.
    This allows for case-insensitive and space-insensitive key lookups and assignments.

    Attributes:
        _normalized_dict (dict): A dictionary that stores normalized keys and their corresponding values.

    Methods:
        _normalize_key(key): Returns a normalized version of the key, where all characters are lowercase and spaces are replaced with underscores.
        __setitem__(key, value): Sets a key-value pair, normalizing the key before storing it.
        __getitem__(key): Retrieves the value associated with the normalized version of the key.
        __contains__(key): Checks if a normalized version of the key exists in the dictionary.
        get(key, default=None): Retrieves the value for the normalized key or returns the default value if the key is not found.
        update(*args, **kwargs): Updates the dictionary with another dictionary or with keyword arguments, using normalized keys.

    Example:
        palettes = NormalizedDict({
            "ONEDARK_CLASSIC_PALETTE": ONEDARK_CLASSIC_PALETTE,
            "onedark vivid palette": ONEDARK_VIVID_PALETTE,
            "GOLDER_sunset": GOLDER_SUNSET,
            "SUNSET SKYLINE": SUNSET_SKYLINE,
        })

        # Accessing the values using different cases or spacing will still work:
        palettes["ONEDARK_CLASSIC_PALETTE"]  # Returns ONEDARK_CLASSIC_PALETTE
        palettes["onedark classic palette"]  # Returns ONEDARK_CLASSIC_PALETTE
        palettes["golder sunset"]            # Returns GOLDER_SUNSET
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._normalized_dict = {}
        self.update(dict(*args, **kwargs))

    def _normalize_key(self, key):
        return key.lower().replace(" ", "_")

    def __setitem__(self, key, value):
        normalized_key = self._normalize_key(key)
        self._normalized_dict[normalized_key] = value
        super().__setitem__(key, value)

    def __getitem__(self, key):
        normalized_key = self._normalize_key(key)
        return self._normalized_dict[normalized_key]

    def __contains__(self, key):
        return self._normalize_key(key) in self._normalized_dict

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update expected at most 1 arguments, got %d" % len(args)
                )
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]
