from pathlib import Path
import cv2
import sys
import image_processing as imp
import dict_creator
import cnn
import printer
import names
import json
import random

import torch
from sklearn.metrics import classification_report


# get relevant .webm file
def get_video_file(user, webpage, settings):
    video_file_name = webpage + '.webm'
    video_file = settings['data_folder'] / 'Dataset_visual_change' / user / video_file_name
    return str(video_file)


def test(arguments):
    # create settings from arguments
    settings = {'data_folder': Path(arguments[1]),
                'model_name': arguments[2],
                'masked': ('masked' in arguments),
                'scrolling': ('scrolling' in arguments),
                'root': ('root' in arguments),
                'overlap': ('overlap' in arguments)}

    # filter users and webpages to use
    users = names.get_all_users()
    users_to_use = list(set(users) & set(arguments))
    if users_to_use:
        users = users_to_use
    else:
        users = users[:1]

    webpages = names.get_all_webpages()
    webpages_to_use = list(set(webpages) & set(arguments))
    if webpages_to_use:
        webpages = webpages_to_use

    print('Loading Model')
    net = cnn.Net()
    state_file = settings['model_name'] + '.pth'
    net.load_state_dict(torch.load(settings['data_folder'] / state_file))
    print('Model loaded successfully')

    # print settings
    printer.print_settings(settings, users, users_to_use, webpages, webpages_to_use, training=False)

    # create dict from CSV files
    csv_dict = dict_creator.create_dict(users, webpages, settings)

    # check if the observations are getting ORed or are separate by layers and their masks
    if settings['masked'] or settings['scrolling'] or settings['root']:
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
            all_obs = all_obs[int(len(all_obs) * (split_percentage) / 100):]
            print(str(len(all_obs)) + ' observations kept')

    vids = {}
    for user in users:
        vids[user] = {}
        for webpage in webpages:
            vids[user][webpage] = cv2.VideoCapture(get_video_file(user, webpage, settings))

    predictions = []
    net.eval()
    labeled = 0
    print('Observations to test: ' + str(len(all_obs)))
    print('Testing started!')
    i = 0
    with torch.no_grad():
        for ob in all_obs:

            # prepare input
            input_frame = imp.get_merged_frame_pair(vids[ob['user']][ob['webpage']], ob, settings)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            input_frame = torch.from_numpy(input_frame.transpose((2, 0, 1)))
            inputs = [input_frame]
            target = min(1, ob['label'])
            labeled += target

            # get prediction
            inputs = torch.stack(inputs, dim=0)
            outputs = net(inputs)
            outputs = torch.sigmoid(outputs)
            output_labels = torch.round(outputs)
            prediction = output_labels.numpy().squeeze().tolist()
            not_rounded = outputs.numpy().squeeze().tolist()

            # fill prediction list
            predictions.append({
                'target': target,
                'prediction': prediction,
                'not_rounded': not_rounded,
                'webpage': ob['webpage'],
                'user': ob['user']
            })

            # print statistics
            i += 1
            if i % 200 == 0:
                print('Observations tested so far: ' + str(i))

    reports = {}
    target_names = ['Visually same', 'Visually different']
    print('REPORT\n')
    print('OVERALL')
    reports['overall'] = classification_report([p['target'] for p in predictions],
                                               [p['prediction'] for p in predictions],
                                               target_names=target_names,
                                               output_dict=True)
    print(classification_report([p['target'] for p in predictions],
                                [p['prediction'] for p in predictions],
                                target_names=target_names))

    print('USERS')
    for u in users:
        print('User ' + u)
        reports[u] = classification_report([p['target'] for p in predictions if p['user'] == u],
                                           [p['prediction'] for p in predictions if p['user'] == u],
                                           target_names=target_names,
                                           output_dict=True)
        print(classification_report([p['target'] for p in predictions if p['user'] == u],
                                    [p['prediction'] for p in predictions if p['user'] == u],
                                    target_names=target_names))
    print('WEBPAGES')
    for wp in webpages:
        print('Webpage ' + wp)
        reports[wp] = classification_report([p['target'] for p in predictions if p['webpage'] == wp],
                                            [p['prediction'] for p in predictions if p['webpage'] == wp],
                                            target_names=target_names,
                                            output_dict=True)
        print(classification_report([p['target'] for p in predictions if p['webpage'] == wp],
                                    [p['prediction'] for p in predictions if p['webpage'] == wp],
                                    target_names=target_names))

    print('NOT ROUNDED OVERALL')
    sorted_preds = sorted(predictions, key=lambda p: p['not_rounded'])
    print('Split at ' + str(sorted_preds[(len(all_obs) - labeled)]['not_rounded']))
    for pred in sorted_preds[:(len(all_obs) - labeled)]:
        pred['prediction'] = 0
    for pred in sorted_preds[(len(all_obs) - labeled):]:
        pred['prediction'] = 1
    reports['not_rounded_overall'] = classification_report([p['target'] for p in sorted_preds],
                                               [p['prediction'] for p in sorted_preds],
                                               target_names=target_names,
                                               output_dict=True)
    print(classification_report([p['target'] for p in sorted_preds],
                                [p['prediction'] for p in sorted_preds],
                                target_names=target_names))

    report_file_name = settings['model_name'] + '_report.json'
    report_file = open(settings['data_folder'] / report_file_name, 'w')
    json.dump(reports, report_file)
    report_file.close()
    print('Report saved as ' + report_file_name)


# when called as main
if __name__ == '__main__':
    test(sys.argv)
