import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from src.fedreclab.federated.utils import RoundStatistics


class BaseClientSelection:
    def __init__(self, num_clients, number_selected_clients, selection_value_type='auto'):
        self.num_clients = num_clients
        if selection_value_type is None or selection_value_type == 'auto':
            if type(number_selected_clients) is float:
                selection_value_type = 'frac'
            elif type(number_selected_clients) is int:
                selection_value_type = 'num'
            else:
                raise ValueError("Invalid selection method")

        if selection_value_type == 'frac':
            self.number_selected_clients = int(num_clients * number_selected_clients)
        elif selection_value_type == 'num':
            self.number_selected_clients = number_selected_clients
        else:
            raise ValueError("Invalid selection method")

    def select(self):
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

    def select(self):
        return list(range(self.num_clients))


class RandomClientSelection(BaseClientSelection):
    def __init__(self, num_clients, number_selected_clients, selection_value_type='frac'):
        super().__init__(num_clients, number_selected_clients, selection_value_type)

    def select(self):
        return random.sample(range(self.num_clients), self.number_selected_clients)
