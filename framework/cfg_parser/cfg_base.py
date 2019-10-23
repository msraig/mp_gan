import yaml
import json
import copy
import inspect


#base class of dict recursive

class DictRecursive(object):
    def __init__(self):
        pass

    def load(self, kargs: dict, shared_dict=None):
        """
        Launch args of class from a dict. All subclass of DictRecursive will call this function automatically. Supported
            types includes int, float, list, str and DictRecursive

        Args:
            kargs: a dict saved the pairs of name/value of attributions
            shared_dict: a shared item used by all other items
        """
        if shared_dict is None:
            shared_dict = {}
        for cls_arg_name in self.__dict__.keys():
            arg_value = None
            if kargs is not None:
                arg_value = kargs[cls_arg_name] if cls_arg_name in kargs.keys() else None
            if shared_dict is not None:
                arg_value = shared_dict[cls_arg_name] if cls_arg_name in shared_dict.keys() else arg_value
            cls_arg = self.__dict__[cls_arg_name]
            self.__dict__[cls_arg_name] = self.parse_single_arg(cls_arg, arg_value, shared_dict)

    def save(self):
        save_dict = {}
        for cls_arg_name in self.__dict__.keys():
            save_dict[cls_arg_name] = self.inverse_single_arg(self.__dict__[cls_arg_name])
        return save_dict

    def load_from_yaml(self, yaml_path, shared_scope='', additional_shared_dict = {}):
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            cfg_cxt = yaml.load(fp.read(), Loader=yaml.FullLoader)
            _shared_dict = cfg_cxt[shared_scope] if shared_scope in cfg_cxt.keys() else dict()
            for _key in additional_shared_dict:
                _shared_dict[_key] = additional_shared_dict[_key]
            self.load(cfg_cxt, _shared_dict)

    def load_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as fp:
            self.load(json.load(fp))

    def save_to_json(self, json_path):
        with open(json_path, 'w') as fp:
            json.dump(self.save(), fp)

    @staticmethod
    def inverse_single_arg(arg_value):
        if issubclass(type(arg_value), DictRecursive):
            return arg_value.save()
        elif isinstance(arg_value, list):
            list_arg_value = list()
            for a_v in arg_value:
                list_arg_value.append(__class__.inverse_single_arg(a_v))
            return list_arg_value
        else:
            return arg_value

    @staticmethod
    def parse_single_arg(cls_arg, arg_value, shared_dict=None):
        if isinstance(cls_arg, int):
            cls_arg_value = int(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, str):
            cls_arg_value = str(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, float):
            cls_arg_value = float(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, list):
            if arg_value is not None:
                cls_arg_value = list()
                for a_v in arg_value:
                    cls_arg_value.append(__class__.parse_single_arg(cls_arg[0], a_v, shared_dict))
            else:
                cls_arg_value = cls_arg
        elif isinstance(cls_arg, dict):
            if arg_value is not None:
                cls_arg_value = dict()
                for a_v in arg_value:
                    cls_arg_value[a_v] = arg_value[a_v]
            else:
                cls_arg_value = cls_arg
        elif issubclass(type(cls_arg), DictRecursive):
            cls_arg_value = type(cls_arg)()
            cls_arg_value.load(arg_value, shared_dict)
        else:
            raise NotImplementedError
        return cls_arg_value

    def match_function_args(self, external_dict, target_func):
        args_dict = copy.deepcopy(external_dict)
        for func_key in inspect.signature(target_func).parameters.keys():
            if func_key not in self.__dict__.keys():
                continue
            if func_key in args_dict.keys():
                continue
            args_dict[func_key] = self.__dict__[func_key]
        return args_dict

class BasicConfigurator(DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str('default')
        self.gpu_list = list([0])


class BasicDataLoaderConfigurator(BasicConfigurator):
    def __init__(self):
        super().__init__()
        self.data_folder = str()
        self.data_name = str()

        self.batch_size = int(0)
        self.random_shuffle = int(0)
        self.random_augment = int(0)

        self.width = int(1)
        self.height = int(1)
        self.depth = int(1)
        self.channel = int(1)

class BasicRunnerConfigurator(BasicConfigurator):
    def __init__(self):
        super().__init__()

        self.log_dir = str()
        self.output_dir = str()
        self.previous_model = str()

        self.max_iter = int(0)
        self.display_step = int(0)
        self.checkpoint_step = int(0)

        self.auto_restart = int(0)
        self.max_num_checkpoint = int(20)
        self.debug_tag = int(0)

#base class for typical networks
class BasicConvNetworkStructureConfigurator(BasicConfigurator):
    def __init__(self):
        super().__init__()

        self.norm = str('bn')
        self.act = str('relu')