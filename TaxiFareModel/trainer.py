# imports
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import *
from TaxiFareModel.utils import *
from TaxiFareModel.encoders import *

set_config(display='diagram')
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def split_holdout(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline

    def run(self, X_train, y_train):
        """set and train the pipeline"""
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # set X and y
    X, y = returnXY(clean_data(get_data(nrows=10_000), test=False))
    my_trainer = Trainer(X,y)
    # hold out
    X_train, X_test, y_train, y_test = my_trainer.split_holdout()
    print(X_test)
    # set pipeline
    my_trainer.set_pipeline()
    print(my_trainer)
    # train
    my_trainer.run(X_train, y_train)
    # evaluate
    my_trainer.evaluate(X_test, y_test)
