from flex.utils.data import BaseDataPreprocessor

from flex.utils.config.base_config import Configuration
from flex.utils.data.base_loaders import BaseDataLoader
from flex.utils.learners.base_learner import BaseLearner
from flex.utils.models.base_model import BaseModel


class Application:
    def __init__(self,
                 loader: BaseDataLoader,
                 preprocessor: BaseDataPreprocessor,
                 model: BaseModel,
                 config: Configuration):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model
        self.config = config

    def run(self):
        # Load data
        raw_data = self.loader.load_data()

        # Preprocess data
        data = self.preprocessor.preprocess_data(raw_data)

        # Load model
        self.model.load()

        # Predict
        result = self.model.predict(data)

        return result




class Runner:
    def __init__(self,
                 loader: BaseDataLoader,
                 preprocessor: BaseDataPreprocessor,
                 model: BaseModel,
                 learner: BaseLearner,
                 config: Configuration):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model
        self.learner = learner
        self.config = config

    def run(self):
        # Load data
        raw_data = self.loader.load_data()

        # Preprocess data
        data = self.preprocessor.preprocess_data(raw_data)

        # TODO: Add to learner train, test split
        train_data = data
        test_data = data

        # Build model
        self.model.build()

        # Train
        self.learner.train(train_data=train_data, model=model)

        # Test
        self.learner.test(test_data=test_data, model=model)

        # Predict
        #self.model.predict()

        # Load performance
        self.config.to_csv(csv_file='../../runs/performance.csv')

class Git:
    def __init__(self, remote=None, local_path=None, branch=None):
        pass

    def clone(self, remote=None, remote_branch=None, local_branch=None):
        pass

    def add(self, ):
    def commit(self, msg, branch=None):
        pass

    def push(self, remote=None, branch=None, ffwd=True):
        pass

    def is_branch_exists(self):
        pass

    def create_branch(self, name):
        pass

    def tag(self, branch, tag):
        import git
        git.Repo.crea
        pass