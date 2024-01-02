import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class DataAnalysis():  
    def __init__(self):
        self.iris = load_iris()
        self.df = pd.DataFrame(
            data= np.c_[self.iris["data"], self.iris["target"]],
            columns= self.iris["feature_names"] + ["target"]
        )
        self.df.rename(
            columns={"sepal length (cm)": "sepal_length",
                     "sepal width (cm)": "sepal_width",
                     "petal length (cm)": "petal_length",
                     "petal width (cm)": "petal_width",
            },
            inplace=True
        )
    
    def __str__(self) -> str:
        return self.iris.DESCR
    
    def dataset_info(self) -> pd.DataFrame:
        return self.df.info()
    
    def dataset_description(self) -> pd.DataFrame:
        return self.df.describe()
    
    def class_sizes(self) -> pd.Series:
        return self.df["target"].value_counts()
    
    def