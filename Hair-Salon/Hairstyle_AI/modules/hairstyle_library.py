class HairstyleLibrary:
    def __init__(self):
        self.styles = []
        self.index = 0

    def set_styles(self, styles):
        if styles != self.styles:
            self.styles = styles
            self.index = 0  # reset when recommendation changes

    def current(self):
        if not self.styles:
            return None
        return self.styles[self.index]

    def next(self):
        if not self.styles:
            return None
        self.index = (self.index + 1) % len(self.styles)
        return self.current()

    def previous(self):
        if not self.styles:
            return None
        self.index = (self.index - 1) % len(self.styles)
        return self.current()
