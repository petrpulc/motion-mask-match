#!/usr/bin/env python3
import os
import pickle


class Mask:
    """
    Mask-RCNN detection object.

    :ivar frame_mask_id ID of mask in respective frame (can be used to align partial sequences)
    :ivar score_dist Distribution of MS COCO class probabilities
    """

    def __init__(self, frame_mask_id, score_dist):
        self.frame_mask_id = frame_mask_id
        self.score_dist = score_dist


class Point:
    """
    Motion tracking point object.



    """

    def __init__(self, x, y, frame_id):
        self.x = x
        self.y = y
        self.frame_id = frame_id
        self.masks = []

    def add_mask(self, mask):
        self.masks.append(mask)


def load_tracks_csv(path):
    sequence_id = 0
    frames = 0
    sequence_dict = {}

    with open(path) as fh:
        for line in fh:
            data = line.split(',')[:-1]
            sequence_dict[sequence_id] = []

            # get frame count once, initialize frame_dict
            if frames == 0:
                frames = int(len(data) / 2)
                frame_dict = {k: [] for k in range(frames)}

            for frame_id in range(frames):
                if data[2 * frame_id] == '':
                    continue
                point = Point(round(float(data[2 * frame_id])), round(float(data[2 * frame_id + 1])), frame_id)
                sequence_dict[sequence_id].append(point)
                frame_dict[frame_id].append(point)

            sequence_id += 1
    return frame_dict, sequence_dict


def assign_masks(frames, path):
    for frame_id, frame_points in frames.items():
        data = pickle.load(open(os.path.join(path, 'mrcnn/{:06d}.jpg.pickle'.format(frame_id + 1)), 'rb'))
        for mask_id in range(len(data['class_ids'])):
            mask_obj = Mask(mask_id, data['score_dist'][mask_id])
            mask = data['masks'][:, :, mask_id]
            for point in frame_points:
                try:
                    if mask[point.y, point.x]:
                        point.add_mask(mask_obj)
                except IndexError:
                    pass


def process_folder(path):
    frames, sequences = load_tracks_csv(os.path.join(path, 'tracks.csv'))
    assign_masks(frames, path)

    output_sequences = []
    for sequence in sequences:
        output_sequence = {}
        for point in sequences[sequence]:
            output_sequence[point.frame_id] = [{'id': mask.frame_mask_id, 'score_dist': mask.score_dist} for mask in
                                               point.masks]
        output_sequences.append(output_sequence)

    pickle.dump(output_sequences, open(os.path.join(path, 'tracks_w_masks.pickle'), 'wb'))


if __name__ == '__main__':
    import sys
    process_folder(sys.argv[1])
