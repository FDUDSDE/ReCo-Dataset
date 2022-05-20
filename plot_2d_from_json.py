# -*- coding: utf-8 -*-
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import json


# find centroid of a polygon
def centroid_coords(poly):
    centroid = poly.centroid
    x, y = centroid.coords[0]
    return x, y


# parallel moving all coords to the center of the graph
def move_coords(coords, diff_xy):
    new_coords = []
    for j in coords:
        x, y = j
        new_coords.append([x - diff_xy[0], y - diff_xy[1]])
    return new_coords


# function of plotting
def plot_community(community, canvas_size, only_boundary=False, only_buildings=False,
                   hide_axis=False, hide_spines=False, printing_title=True,
                   building_color='cornflowerblue', boundary_color='red'):
    canvas_poly = Polygon([(0, 0), (canvas_size, 0), (canvas_size, canvas_size), (0, canvas_size)])
    canvas_cen_x, canvas_cen_y = centroid_coords(canvas_poly)

    # if coords transfer is needed, read data from raw data, else read data from _rp
    boundary_coords = community['boundary']
    buildings = community['buildings']

    _id = community['_id']

    # construct boundary polygon
    boundary_poly = Polygon(boundary_coords)

    boundary_cen_x, boundary_cen_y = centroid_coords(boundary_poly)
    diff_xy = [boundary_cen_x-canvas_cen_x, boundary_cen_y-canvas_cen_y]

    # plot buildings by 'filling', if needed, transfer the wgs84 coords to webmercator coords
    for i in range(len(buildings)):
        building_coords = buildings[i]['coords']
        # move buildings to the center
        building_coords = move_coords(building_coords, diff_xy)
        # polygons for buildings
        buildings_poly = Polygon(building_coords)
        if not only_boundary:
            plt.fill(*buildings_poly.exterior.xy, color=building_color)
        continue

    # move boundary to the center and plot it
    boundary_coords = move_coords(boundary_coords, diff_xy)
    # polygon for boundary
    boundary_poly = Polygon(boundary_coords)
    if not only_buildings:
        plt.plot(*boundary_poly.exterior.xy, color=boundary_color)

    # set axis to 'equal'
    plt.axis('equal')

    # constrain x, y values
    plt.xlim((0, canvas_size))
    plt.ylim((0, canvas_size))

    # hide axis spines
    if hide_spines:
        hide_axis = True
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

    # hide x, y axis
    if hide_axis:
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

    if printing_title:
        print(_id)


if __name__ == '__main__':
    _id_to_draw = '61ef8a8b32b5d4672152cf77'
    canvas_size = 800

    plotting = {}
    with open('ReCo_json.json', encoding='utf-8') as f:
        data = json.load(f)

        for example in data:
            if example['_id'] == _id_to_draw:
                plotting = example

    plot_community(plotting, canvas_size, building_color='black', boundary_color='black')
    plt.show()
