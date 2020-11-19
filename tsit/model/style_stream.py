from tensorflow.keras import Model
from tsit.model.misc import CSResBlk


class StyleStream(Model):
    """Model code for Content Stream

    Args:
        Nothing
    """
    def __init__(self, k):
        """Initialize StyleStream with k parameters

        Args:
            k (list): list of channel no.
        """
        super(StyleStream, self).__init__()
        self.csresblk = []
        for i in range(7):
            self.csresblk.append(CSResBlk(k[i], k[i+1]))
    def call(self, x):
        """Run StyleStream

        Args:
            x (tensor): Input

        Returns:
            tensor: Output
        """
        content_feature = []
        for i in range(7):
            x = self.csresblk[i](x)
            if i != 0:
                content_feature.append(x)
        return content_feature

if __name__ == "__main__":
    cs = StyleStream([64, 128, 256, 512, 1024, 1024, 1024, 1024])
