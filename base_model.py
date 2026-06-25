from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Base class for bike-demand hackathon submissions.

    Competitors submit a folder:

        submissions/<team_name>/
            model.py
            train.py
            weights.joblib   # or another artifact file, if evaluator is configured for it

    model.py must define a class named Model that subclasses BaseModel.

    The evaluator will do:

        model = Model()
        model.load("submissions/<team_name>/weights.joblib")
        predictions = model.predict(hidden_test_targets_df)

    Important:
        hidden_test_targets_df is NOT ride-level data.
        It is station-hour target data.

        Each row means:
            predict demand for this city + station + hour.

        The dataframe will contain columns such as:
            id
            city
            start_station_id
            hour_ts or target_hour_start
            date
            weekday
            hour
            weather columns
            station metadata columns

        It will NOT contain:
            demand
            started_at
            ended_at
            end_station_id
            usage_time_minutes
            distance_meters
            user_type

    predict() must return:
        a 1D array-like object of length len(test_df)

    Values may be floats. The evaluator clips predictions to non-negative values.
    """

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load trained weights/artifacts from disk."""
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict station-hour demand for each row in test_df.

        Args:
            test_df:
                Hidden station-hour target dataframe.

        Returns:
            1D array-like of predicted demand values.
        """
        pass
