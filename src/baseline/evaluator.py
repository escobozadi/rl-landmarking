from .data_loader import DataLoader


class Evaluate(object):
    def __init__(self, arguments, label_ids):
        self.data = DataLoader(arguments.files, landmarks=len(label_ids))

