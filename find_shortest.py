from math import cos, sin, radians, sqrt

import numpy as np


def plotLineLow(x0, y0, x1, y1):
    line_points = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    for x in range(x0, x1):
        line_points.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return line_points


def plotLineHigh(x0, y0, x1, y1):
    line_points = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    for y in range(y0, y1):
        line_points.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return line_points


def plot_line(x0, y0, x1, y1):
    # print("From:(",x0,",",y0,") to (",x1,",",y1,")")
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)


def get_furthest_pos_point(start_point, line_points, bin_array):
    dist_list = np.zeros((len(line_points)), )
    point_list = np.zeros((len(line_points), 2), )
    for point in range(0, len(line_points)):
        if bin_array[line_points[point][0], line_points[point][1]] == 0:
            dist = sqrt((line_points[point][0] - start_point[0]) ** 2 + (line_points[point][1] - start_point[1]) ** 2)
            dist_list[point] = dist
            point_list[point][0] = line_points[point][0]
            point_list[point][1] = line_points[point][1]
    return dist_list, point_list


def find_shortest_line(start_point, bin_array, start_angle, end_angle, stepping):
    #print("Find shortest:", start_point[0], ",", start_point[1])
    ranged_line = (0, 50)
    # print("Rotational range: ",ranged_line)
    min_dist = 1000000
    min_angle = -1
    min_from = (-1, -1)
    min_to = (-1, -1)
    start_angle_cor = int(round(start_angle * (1 / stepping)))
    end_angle_cor = int(round(end_angle * (1 / stepping)))
    for angle in range(start_angle_cor, end_angle_cor):
        stepping_angle = angle * stepping
        rotation_point = (round((ranged_line[0] * cos(radians(stepping_angle)) - ranged_line[1] * sin(radians(stepping_angle)))), round((ranged_line[0] * sin(radians(stepping_angle)) + ranged_line[1] * cos(radians(stepping_angle)))))
        # print(angle,"° :",rotation_point)
        line_points_pos = plot_line(start_point[0], start_point[1], start_point[0] + rotation_point[0], start_point[1] + rotation_point[1])
        line_points_neg = plot_line(start_point[0], start_point[1], start_point[0] - rotation_point[0], start_point[1] - rotation_point[1])
        if len(line_points_neg) == len(line_points_pos):
            # print("Match")
            neg_dist, neg_point_list = get_furthest_pos_point(start_point, line_points_neg, bin_array)
            pos_dist, pos_point_list = get_furthest_pos_point(start_point, line_points_pos, bin_array)
            if len(pos_point_list) > 0 and len(neg_point_list) > 0:
                # print("NDPL:",neg_point_list)
                # print("ND:",neg_dist)
                # print("PDPL:",pos_point_list)
                # print("PD:",pos_dist)
                if np.mean(neg_dist) != 0 and np.mean(pos_dist) != 0:
                    nd_non_null_min = np.amin(neg_dist[neg_dist > 0])
                    nd_max_point = neg_point_list[np.where(neg_dist == nd_non_null_min)[0][0]]
                    # print("NDMP:",nd_max_point)
                    pd_non_null_min = np.amin(pos_dist[pos_dist > 0])
                    pd_max_point = pos_point_list[np.where(pos_dist == pd_non_null_min)[0][0]]
                    # print("PDMP:", pd_max_point)
                    dist = sqrt((nd_max_point[0] - pd_max_point[0]) ** 2 + (nd_max_point[1] - pd_max_point[1]) ** 2)
                    # print(dist)
                    if min_dist > dist and dist != 0.0:
                        min_from = (int(round(nd_max_point[0])), int(round(nd_max_point[1])))
                        min_to = (int(round(pd_max_point[0])), int(round(pd_max_point[1])))
                        min_dist = dist
                        min_angle = stepping_angle
                        if min_angle < 0:
                            min_angle %= 360
                        #print("NEW: Min found dist:", min_dist, "with angle: ", min_angle, ", from:(", min_from[0], ",", min_from[1], ") to (", min_to[0], ",", min_to[1], ")")
                    # print(np.amax(dist_sum[bin_array == 1]))
                    # print(np.where(dist_sum == np.amax(dist_sum)))
    #print("FS done:", start_angle, ",", end_angle, ",", stepping)
    #print("FINAL: Min found dist:", min_dist, "with angle: ", min_angle, ", from:(", min_from[0], ",", min_from[1], ") to (", min_to[0], ",", min_to[1], ")")
    return min_dist, min_from, min_to, min_angle
