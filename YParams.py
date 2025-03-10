import json
import logging
import pprint

from ruamel.yaml import YAML


class ParamsBase:
    """Convenience wrapper around a dictionary

    Allows referring to dictionary items as attributes, and tracking which
    attributes are modified.
    """

    def __init__(self):
        self._original_attrs = None
        self.params = {}
        self._original_attrs = list(self.__dict__)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def to_dict(self):
        new_attrs = {
            key: val for key, val in vars(self).items()
            if key not in self._original_attrs
        }
        return {**self.params, **new_attrs}

    @staticmethod
    def from_json(path: str) -> "ParamsBase":
        with open(path) as f:
            c = json.load(f)
        params = ParamsBase()
        params.update_params(c)
        return params

    def update_params(self, config):
        for key, val in config.items():
            if val == 'None':
                val = None

            if type(val) == dict:
                child = ParamsBase()
                child.update_params(val)
                val = child

            self.params[key] = val
            self.__setattr__(key, val)


class YParams(ParamsBase):
    def __init__(self, yaml_filename, config_name, print_params=False):
        """Open parameters stored with ``config_name`` in the yaml file ``yaml_filename``"""
        super().__init__()
        self._yaml_filename = yaml_filename
        self._config_name = config_name

        with open(yaml_filename) as _file:
            d = YAML().load(_file)[config_name]

        self.update_params(d)

        if print_params:
            print("------------------ Configuration ------------------")
            for k, v in d.items():
                print(k, end='=')
                pprint.pprint(v)
            print("---------------------------------------------------")

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.to_dict().items():
            logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------------")