from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """The abstract ML/DL base class
    for the sake of writing structured code

    This is for sub-model. The sub model will be used after the first stage.
    """

    def __init__(self, *args, **kwargs):
        # self.model_weight
        # self.model
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """take an input
        then preprocess it for the format that model can use in inference process

        return preprocessed data
        
        It should be defined in the same format
        for example, All image models should take an image in numpy array `(H, W, 3-color RGB)`
        """
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """take the inference output from the model
        then process it to usable/human-readible format

        It should be usable/easy to use for the other models that use this data as thier input
        """
        pass

    @abstractmethod
    def __call__(self, *args):
        """run inference via `model_var(data)`

        Doc string should be overridden in the child's `__call__` method
        Accepted input types should be cleary specified
        """
        pass

    def __repr__(self) -> str:
        return 'Model Name: ref URL'

    # alias: another way to run inference
    # detect = predict = __call__
