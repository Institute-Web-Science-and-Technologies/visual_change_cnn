# Copyright (C) 2020 Daniel Vossen
# see COPYING for further details

from colorama import Fore, Style


def get_colored_bool_string(b):
    return (Fore.GREEN if b else Fore.RED) + str(b) + Style.RESET_ALL


def get_yellow_string(s):
    return Fore.YELLOW + s + Style.RESET_ALL


def print_settings(settings, users, users_to_use, webpages, webpages_to_use, training=True):
    print(Fore.MAGENTA + 'SETTINGS:' + Style.RESET_ALL)
    print('Root Layer Mode: ' + get_colored_bool_string(settings['root']))
    print('Masked Mode: ' + get_colored_bool_string(settings['masked']))
    print('Scrolling Mode: ' + get_colored_bool_string(settings['scrolling']))
    print('Minimum Overlap Mode: ' + get_colored_bool_string(settings['overlap']))
    if training:
        print('CUDA: ' + get_colored_bool_string(settings['has_cuda']))
        print('Balanced Mode: ' + get_colored_bool_string(settings['balanced']))
    print('Users: ' + get_yellow_string(str(users) + (' (default)' if not users_to_use else '')))
    print('Webpages: ' + get_yellow_string(str(webpages) + (' (default)' if not webpages_to_use else '')))
    print('Model Name: ' + get_yellow_string(settings['model_name']) + '\n')


def print_help():
    print(Fore.MAGENTA + 'USAGE:' + Style.RESET_ALL)
    print('python3 /path/to/script.py /path/to/data model_name <modes>\n')
    print('Possible modes:')
    print('cuda, root, masked, scrolling, balanced, overlap, batch_size=X, epochs=Y, split=PERCENTAGE, seed=Z, <webpage_name>, <user_name>\n')
    print('Example usage:')
    print('python3 ./script.py ./data cuda masked epochs=3 batch_size=5 amazon kia p1 p2')


def print_help_notice():
    print('Add ' + Fore.YELLOW + 'help' + Style.RESET_ALL + ' as parameter to see all options available to you.\n')


def print_training_start(obs1, obs2, batch_size, settings):
    obs_per_epoch = len(obs1) + len(obs2)
    if settings['balanced']:
        obs_per_epoch = 2 * min(len(obs1), len(obs2))
    print('Number of observations per epoch: ' + get_yellow_string(str(obs_per_epoch)))
    percentage = round(len(obs1) * 100 / (len(obs1) + len(obs2)), 2)
    if settings['balanced']:
        percentage = 50.00
    print('Labeled with visual change: ' + get_yellow_string(str(percentage) + '%\n'))
    print('Training started with batch size: ' + get_yellow_string(str(batch_size)))