# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from plot_2d_from_json import plot_community
from tqdm import tqdm
import json
import os


if __name__ == '__main__':
    os.makedirs('image_data', exist_ok=True)
    canvas_size = 1000

    with open('ReCo_json.json', encoding='utf-8') as f:
        data = json.load(f)

        for community in tqdm(data):
            _id = community['_id']

            plot_community(community, canvas_size, printing_title=False,
                           building_color='black', boundary_color='red', hide_spines=True)

            plt.savefig('image_data/'+str(_id)+'.jpg', dpi=150)
            plt.clf()
