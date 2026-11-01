import numpy as np

from src.fedreclab.federated.utils import RoundStatistics


class BaseClientSelection:
    def __init__(self, num_clients, number_selected_clients, selection_value_type='auto', seed=None):
        self.num_clients = num_clients
        self.seed = seed
        if selection_value_type is None or selection_value_type == 'auto':
            if type(number_selected_clients) is float:
                selection_value_type = 'frac'
            elif type(number_selected_clients) is int:
                selection_value_type = 'num'
            else:
                raise ValueError("Invalid selection method")

        if selection_value_type == 'frac':
            if not 0 < number_selected_clients <= 1:
                raise ValueError(
                    f"selection_value_type='frac' expects a fraction in (0, 1], "
                    f"got {number_selected_clients}. Use selection_value_type='num' "
                    f"to select a fixed client count."
                )
            self.number_selected_clients = int(num_clients * number_selected_clients)
        elif selection_value_type == 'num':
            self.number_selected_clients = number_selected_clients
        else:
            raise ValueError("Invalid selection method")

        if (
            self.number_selected_clients < 1
            or self.number_selected_clients > self.num_clients
        ):
            raise ValueError(
                f"Cannot select {self.number_selected_clients} clients from a "
                f"population of {self.num_clients} without replacement."
            )

    def select(self, current_round=0):
        raise NotImplementedError("Subclasses must implement this method")

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        pass


class FullClientSelection(BaseClientSelection):
    def __init__(self, num_clients):
        super().__init__(
            num_clients=num_clients,
            number_selected_clients=num_clients,
            selection_value_type='num'
        )

    def select(self, current_round=0):
        return list(range(self.num_clients))


class RandomClientSelection(BaseClientSelection):
    def __init__(self, num_clients, number_selected_clients, selection_value_type='auto', seed=None):
        super().__init__(num_clients, number_selected_clients, selection_value_type, seed=seed)

    def select(self, current_round=0):
        if self.seed is None:
            ids = np.random.choice(self.num_clients, size=self.number_selected_clients, replace=False)
        else:
            # TODO: how current_round is folded into the seed should be overridable
            # (e.g. [seed, round] vs a stored advancing Generator).
            rng = np.random.default_rng([self.seed, current_round])
            ids = rng.choice(self.num_clients, size=self.number_selected_clients, replace=False)
        return [int(i) for i in np.atleast_1d(ids)]
