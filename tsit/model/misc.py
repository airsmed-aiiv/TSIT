from tensorflow.keras import Model


class CSResBlk(Model):
    def __init__(self):
        super(CSResBlk, self).__init__()
        
    def call(self, x):
        pass
    

class FADE(Model):
    def __init__(self):
        super(FADE, self).__init__()

    def call(self, x, feature):
        pass


class FADEResBlk(Model):
    def __init__(self):
        super(FADEResBlk, self).__init__()

    def call(self, x, feature):
        pass
