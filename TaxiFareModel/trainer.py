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
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.experiment_name = 'Batman_revenge'

    def split_holdout(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def set_pipeline(self, option):
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

        if option == 'LinearRegression':
            self.pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('linear_model', LinearRegression())
            ])
        elif option == 'RandomForestRegressor':
            self.pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('random_forest', RandomForestRegressor())
            ])
        elif option == 'DecisionTreeRegressor':
            self.pipeline = Pipeline([('preproc', preproc_pipe),
                                      ('decision_tree', DecisionTreeRegressor())
                                      ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    list_ = [
        'DecisionTreeRegressor', 'LinearRegression', 'RandomForestRegressor'
    ]
    for option in list_:
        # set X and y
        X, y = returnXY(clean_data(get_data(nrows=10_000), test=False))
        my_trainer = Trainer(X,y)
        # hold out
        my_trainer.split_holdout()
        # set pipeline
        my_trainer.set_pipeline(option)
        # train
        my_trainer.run()
        # evaluate
        rmse = my_trainer.evaluate()
        #logging
        my_trainer.mlflow_log_param('model', option)
        my_trainer.mlflow_log_metric('rmse', rmse)
        experiment_id = my_trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
        my_trainer.save_model()
