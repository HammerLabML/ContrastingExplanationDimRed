from abc import ABC, abstractmethod


class DimRed():
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    @abstractmethod
    def fit(self, X):
        raise NotImplementedError()
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
