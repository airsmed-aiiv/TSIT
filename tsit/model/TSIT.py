from tsit.model.generator import Generator
from tsit.model.style_stream import StyleStream
import tensorflow as tf
from tensorflow.keras import Model
from tsit.model.content_stream import ContentStream
from tsit.model.style_stream import StyleStream
from tsit.model.generator import Generator


class TSIT(Model):
    def __init__(self, k):
        self.cs = ContentStream(k)
        self.ss = StyleStream(k)
        self.g = Generator(k)
        
    def call(self, x):
        cs = self.cs(x)
        ss = self.ss(x)
        return self.g(cs, ss)
    