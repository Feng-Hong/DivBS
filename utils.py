import string
from datetime import datetime
import secrets
import os
import csv
import wandb
def random_str(num):
    salt = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(num))

    return salt

def get_date():

    now = datetime.now()
    return str(now.strftime("20%y_%h_%d_%H_%M_%S"))

def re_nest_configs(config_dict):
    flattened_params = [key for key in config_dict.keys() if '.' in key]
    for param in flattened_params:
        value = config_dict._items.pop(param)
        # value = config_dict[param]
        # del config_dict[param] 
        param_levels = param.split('.')
        parent = config_dict._items
        for level in param_levels:
            if isinstance(parent[level], dict):
                parent = parent[level]
            else:
                parent[level] = value

    if 'sweep_config' in config_dict.keys():
        config_dict._items.pop("sweep_config")

class custom_logger():
    def __init__(self, output_path, name='log'):
        os.makedirs(output_path, exist_ok=True)
        now = datetime.now()
        logger_name = str(now.strftime("20%y_%h_%d_")) + name + ".txt"
        self.logger_path = os.path.join(output_path, logger_name)
        self.csv_path = os.path.join(output_path, logger_name.replace('.txt', '.csv'))
        # init logger file
        f = open(self.logger_path, "w+")
        f.write(self.get_local_time() + 'Start Logging \n')
        f.close()
        # init csv
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([self.get_local_time(), ])

    def get_local_time(self):
        now = datetime.now()
        return str(now.strftime("%y_%h_%d %H:%M:%S : "))

    def info(self, log_str):
        print(str(log_str))
        with open(self.logger_path, "a") as f:
            f.write(self.get_local_time() + str(log_str) + '\n')

    def raise_error(self, error):
        prototype = '************* Error: {} *************'.format(str(error))
        self.info(prototype)
        raise ValueError(str(error))

    def info_iter(self, epoch, batch, total_batch, info_dict, print_iter):
        if batch % print_iter != 0:
            pass
        else:
            acc_log = 'Epoch {:5d}, Batch {:6d}/{},'.format(epoch, batch, total_batch)
            for key, val in info_dict.items():
                acc_log += ' {}: {:9.3f},'.format(str(key), float(val))
            self.info(acc_log)

    def write_results(self, result_list):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

    def wandb_init(self, config , project, name):
        wandb.init(project=project, name=name, config=config)
        

    def wandb_log(self, log_dict):
        wandb.log(log_dict)

    def wandb_finish(self):
        wandb.finish()

