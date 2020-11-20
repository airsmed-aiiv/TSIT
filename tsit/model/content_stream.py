from tsit.model.misc import CSResBlk
from tensorflow.keras import Model


class ContentStream(Model):
    """Model

    Args:
        Nothing
    """
    def __init__(self, k):
        super(ContentStream, self).__init__()
        self.csresblk = []
        for i in range(7):
            self.csresblk.append(CSResBlk(k[i], k[i+1]))
    def call(self, x):
        content_feature = []
        for i in range(7):
            x = self.csresblk[i](x)
            if i != 0:
                content_feature.append(x)
        return content_feature

if __name__ == "__main__":
    k = [64, 128, 256, 512, 1024, 1024, 1024, 1024]
    cs = ContentStream(k)
    