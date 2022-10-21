import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET


DIOR_CLASSES = ['airplane', 'airport', 'baseballfield', 'basketballcourt',
                'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
                'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
                'tenniscourt', 'trainstation', 'vehicle', 'windmill'
                ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 20], help="Range of seeds"
    )
    return parser.parse_args()


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in DIOR_CLASSES}
    data_file = ["data/dior/ImageSets/Main/train.txt"]
    for sets in data_file:
        file = open(sets, 'r')
        fileids = file.read().split('\n')
        fileids = [x for x in fileids if x != '']
        file.close()
        data.extend(fileids)
    for fileid in data:
        anno_file = os.path.join("data/dior/", "Annotations/Horizontal_Bounding_Boxes", fileid + ".xml")
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 10]
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in data_per_cat.keys():
            c_data = []
            for j, shot in enumerate(shots):
                diff_shot = shots[j] - shots[j - 1] if j != 0 else 1
                shots_c = random.sample(data_per_cat[c], diff_shot)
                num_objs = 0
                for s in shots_c:
                    if s not in c_data:
                        tree = ET.parse(s)
                        file = tree.find("filename").text
                        name = file.strip(".jpg")  # "JPEGImages/{}".format(file)
                        c_data.append(name)
                        for obj in tree.findall("object"):
                            if obj.find("name").text == c:
                                num_objs += 1
                        if num_objs >= diff_shot:
                            break
                result[c][shot] = copy.deepcopy(c_data)
        save_path = "data/few_shot_ann/dior/diorsplit/seed{}".format(i)
        os.makedirs(save_path, exist_ok=True)
        for c in result.keys():
            for shot in result[c].keys():
                filename = "box_{}shot_{}_train.txt".format(shot, c)
                with open(os.path.join(save_path, filename), "w") as fp:
                    fp.write("\n".join(result[c][shot]) + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)
