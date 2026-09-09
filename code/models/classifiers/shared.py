from torch import nn


class SharedClassifier(nn.Module):
    """The same object accepts encoder Z or subsequent 128-wide A/S fusion."""
    def __init__(self, num_classes):
        super().__init__()
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        self.num_classes = num_classes
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                    nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, representation):
        if representation.ndim != 2 or representation.shape[1] != 128:
            raise ValueError('Classifier requires [N, 128] representation')
        return self.layers(representation)
