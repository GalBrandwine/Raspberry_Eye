class FlagsHolder:
    """A simple flags object to be shares between with app_context. """

    def __init__(self):
        self.smart_toggle_global = False
        self.tom_toggle = False
        self.mufasa_toggle = False
        self.release_camera = False