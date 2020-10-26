import sys
import csv


def create_dict(users, webpages, settings):
    print('Creating dictionary from CSV files')
    csv_dict = {}

    for user in users:
        csv_dict[user] = {}
        for webpage in webpages:
            csv_dict[user][webpage] = {}
            # get all relevant .csv files and save into one dict
            for name in ['features_meta', 'labels-l1']:
                data_name = webpage + '_' + name + '.csv'
                csv_file = settings['data_folder'] / 'Dataset_visual_change' / user / data_name
                csv_dict[user][webpage][name] = []
                try:
                    with open(csv_file, newline='') as file:
                        if name == 'features' or name == 'features_meta':
                            csv_reader = csv.DictReader(file)
                            for row in csv_reader:
                                csv_dict[user][webpage][name].append(dict(row))
                        else:
                            csv_reader = csv.reader(file)
                            for row in csv_reader:
                                csv_dict[user][webpage][name].append(row)
                except FileNotFoundError:
                    print('No file found. Add the path of the folder containing Dataset_visual_change as first parameter.')
                    sys.exit()

            # add user and website as parameters for easier use later on
            for ob in csv_dict[user][webpage]['features_meta']:
                ob['user'] = user
                ob['webpage'] = webpage
                ob['label'] = int(float(csv_dict[user][webpage]['labels-l1'][int(ob['observation_id'])][0]))

            if settings['root']:
                csv_dict[user][webpage]['features_meta'] = list(filter(lambda ob: ob.get('layer_type') == 'root',
                                                                       csv_dict[user][webpage]['features_meta']))

            # combine different layer types into one labeled observation for unmasked OR between obs
            csv_dict[user][webpage]['labeled_obs'] = obs = []
            for observation in csv_dict[user][webpage]['features_meta']:
                labeled_ob = next((o for o in obs if o['prev_video_frame'] == int(observation['prev_video_frame'])), None)
                if labeled_ob:
                    labeled_ob['label'] += int(
                        float(csv_dict[user][webpage]['labels-l1'][int(observation['observation_id'])][0]))
                else:
                    obs.append({
                        'user': user,
                        'prev_video_frame': int(observation['prev_video_frame']),
                        'cur_video_frame': int(observation['cur_video_frame']),
                        'label': int(float(csv_dict[user][webpage]['labels-l1'][int(observation['observation_id'])][0])),
                        'webpage': webpage,
                        'overlap_width': int(observation['overlap_width']),
                        'overlap_height': int(observation['overlap_height'])
                    })

    print('Dict created')
    return csv_dict
