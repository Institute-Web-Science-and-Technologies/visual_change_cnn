# Copyright (C) 2020 Daniel Vossen
# see COPYING for further details

from pathlib import Path
import cv2
import sys
import random
import image_processing as imp
import dict_creator
import cnn
import printer
import names
from colorama import Fore, Style
from functools import reduce

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


# splits list l in 2 by filter parameter p
def partition(l, p):
    return reduce(lambda x, y: x[0].append(y) or x if p(y) else x[1].append(y) or x, l,  ([], []))


# get relevant .webm file
def get_video_file(user, webpage, settings):
    video_file_name = webpage + '.webm'
    video_file = settings['data_folder'] / 'Dataset_visual_change' / user / video_file_name
    return str(video_file)


# prepare shuffled obs for next epoch, rotating through lists for balanced mode
def prepare_next_observations(os1, os2, settings):
    if settings['balanced']:
        next_obs = os1[:min(len(os1), len(os2))] + os2[:min(len(os1), len(os2))]
        os1 = os1[min(len(os1), len(os2)):] + os1[:min(len(os1), len(os2))]
        os2 = os2[min(len(os1), len(os2)):] + os2[:min(len(os1), len(os2))]
    else:
        next_obs = os1 + os2
    random.shuffle(next_obs)
    return os1, os2, next_obs


# where the magic happens
def train(arguments):
    # print help information
    if 'help' in arguments:
        printer.print_help()
        sys.exit()
    else:
        printer.print_help_notice()

    # create settings from arguments
    settings = {'data_folder': Path(arguments[1]),
                'masked': ('masked' in arguments),
                'scrolling': ('scrolling' in arguments),
                'has_cuda': ('cuda' in arguments),
                'root': ('root' in arguments),
                'balanced': ('balanced' in arguments),
                'overlap': ('overlap' in arguments),
                'model_name': arguments[2]}

    # filter users and webpages to use
    users = names.get_all_users()
    users_to_use = list(set(users) & set(arguments))
    if users_to_use:
        users = users_to_use
    else:
        users = users[1:]

    webpages = names.get_all_webpages()
    webpages_to_use = list(set(webpages) & set(arguments))
    if webpages_to_use:
        webpages = webpages_to_use

    # print settings
    printer.print_settings(settings, users, users_to_use, webpages, webpages_to_use)

    # create dict from CSV files
    csv_dict = dict_creator.create_dict(users, webpages, settings)

    # CNN
    net = cnn.Net()
    criterion = nn.BCEWithLogitsLoss()
    if 'load' in arguments:
        state_file = settings['model_name'] + '.pth'
        net.load_state_dict(torch.load(settings['data_folder'] / state_file))
        print('Model loaded successfully')

    if 'loadautosave' in arguments:
        state_file = 'autosave.pth'
        net.load_state_dict(torch.load(settings['data_folder'] / state_file))
        print('Model loaded successfully')

    if settings['has_cuda']:
        device = torch.device("cuda")
        net = net.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    i = 0
    vids = {}
    for user in users:
        vids[user] = {}
        for webpage in webpages:
            vids[user][webpage] = cv2.VideoCapture(get_video_file(user, webpage, settings))

    torch.cuda.empty_cache()

    # check if the observations are getting ORed or are separate by layers and their masks
    if settings['masked'] or settings['scrolling'] or settings['root'] or settings['overlap']:
        all_obs = [ob for user in users for webpage in webpages for ob in csv_dict[user][webpage]['features_meta']]
    else:
        all_obs = [ob for user in users for webpage in webpages for ob in csv_dict[user][webpage]['labeled_obs']]

    if settings['overlap']:
        all_obs = list(filter(lambda o: (int(o['overlap_height']) > 32) and (int(o['overlap_width']) > 32), all_obs))

    # split obs into training data and testing data
    # random_seed is used so test_cnn.py will have the same list to work with
    split_percentage = int(next((s for s in arguments if 'split' in s), 'split=0').split('=')[-1])
    if split_percentage:
        if split_percentage > 99 or split_percentage < 1:
            print('Percentage has to be between 0 and 100, splitting aborted')
        else:
            print('Splitting data:')
            print(str(len(all_obs)) + ' observations total')
            random_seed = next((s for s in arguments if 'seed' in s), 'seed=' + settings['model_name']).split('=')[-1]
            random.Random(random_seed).shuffle(all_obs)
            all_obs = all_obs[:int(len(all_obs) * split_percentage / 100)]
            print(str(len(all_obs)) + ' observations kept')

    # training parameters
    batch_size = int(next((s for s in arguments if 'batch_size' in s), 'batch_size=10').split('=')[-1])
    epochs = int(next((s for s in arguments if 'epochs' in s), 'epochs=5').split('=')[-1])
    running_loss = 0.0
    inputs = []
    targets = []

    # split obs between labels
    obs1, obs2 = partition(all_obs, lambda x: x.get('label') > 0)
    random.shuffle(obs1)
    random.shuffle(obs2)

    # start training
    printer.print_training_start(obs1, obs2, batch_size, settings)
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch ' + Fore.YELLOW + str(epoch + 1) + '/' + str(epochs) + Style.RESET_ALL + ' started')
        obs1, obs2, next_obs = prepare_next_observations(obs1, obs2, settings)
        for ob in next_obs:
            input_frame = imp.get_merged_frame_pair(vids[ob['user']][ob['webpage']], ob, settings)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            input_frame = torch.from_numpy(input_frame.transpose((2, 0, 1)))
            inputs.append(input_frame)
            targets.append([min(1, ob['label'])])

            if len(inputs) >= batch_size:

                inputs = torch.stack(inputs, dim=0)
                # print(inputs.shape)
                # inputs = torch.from_numpy(inputs)
                targets = torch.Tensor(targets)

                if settings['has_cuda']:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # forward + backward + optimize
                output = net(inputs)
                loss = criterion(output, targets)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clean up GPU memory
                del inputs
                del targets
                torch.cuda.empty_cache()

                # current loss
                running_loss += loss.item() * batch_size
                inputs = []
                targets = []

            # print statistics
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
            i += 1
        autosave_path = settings['data_folder'] / 'autosave.pth'
        torch.save(net.state_dict(), autosave_path)
        print('Autosave')
        if ('saveall' in arguments) and (epoch < (epochs - 1)):
            file_name = settings['model_name'] + '_afterEpoch' + str(epoch + 1) + '.pth'
            PATH = settings['data_folder'] / file_name
            torch.save(net.state_dict(), PATH)

    print('Finished Training')

    file_name = settings['model_name'] + '.pth'
    PATH = settings['data_folder'] / file_name
    torch.save(net.state_dict(), PATH)

    print('Saved as ' + file_name)


# when called as main
if __name__ == '__main__':
    train(sys.argv)
