# /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy as sp
import scipy.spatial
from . import general
import warnings
import itertools as itt


# Get the centroid of a collection of points
#   *points:
#       the points used to calculate the centroid. all should be sequences
#       of equal length containing numerics.
def centroid(*points):
    points = np.array(points)
    return sum(points) / len(points)


# Estimate the geometric median of a collection of points
#   *points:
#       the points used to calculate the geometric median. all should be
#       sequences of equal length containing numerics.
#   iter=100:
#       the maximum number of iterations to perform. int.
#   epsilon=None:
#       if the distance between two consecutive estimates is less than
#       epsilon * the average width of the convex hull of points, no further
#       estimates are made and the last estimate is returned. numeric or None.
def weiszfeld(*points, iter=100, epsilon=None):
    points = np.array(points)
    y = centroid(*points)
    # Calculate epsilon
    if epsilon is not None:
        hull = sp.spatial.ConvexHull(points)
        vertices = hull.points[hull.vertices]
        epsilon = epsilon * sum(abs(vertices - y)) / len(vertices)
    for i in range(iter):
        old_y = y
        # New estimate
        y = t_tilde(y, points)
        # Stop if distance between consecutive estimates < epsilon
        if epsilon is not None:
            if sum((old_y - y) ** 2) ** 0.5 < epsilon:
                break
    return y


def modified_weiszfeld(*points, iter=100, epsilon=None):
    points = np.array(points)
    y = centroid(*points)
    # Calculate epsilon
    if epsilon is not None:
        hull = sp.spatial.ConvexHull(points)
        vertices = hull.points[hull.vertices]
        epsilon = epsilon * sum(abs(vertices - y)) / len(vertices)
    for i in range(iter):
        old_y = y
        # New estimate
        eta = 1 if y in points else 0
        y = max(0, (1 - eta / r_tilde(y, points))) * t_tilde(y, points) + \
            min(1, eta / r_tilde(y, points)) * y
        # Stop if distance between consecutive estimates < epsilon
        if epsilon is not None:
            if sum((old_y - y) ** 2) ** 0.5 < epsilon:
                break
    return y


def r_tilde(y, points):
    r = np.zeros(2)
    for i in points:
        if not np.array_equal(i, y):
            r += (i - y) / sum((i - y) ** 2) ** 0.5
    return sum(r ** 2) ** 0.5


def t_tilde(y, points):
    # Creator function for numerator of Weiszfeld's algorithm
    def num(x):
        return lambda j, *p: p[j] / sum((p[j] - x) ** 2) ** 0.5

    # Creator function for denominator of Weiszfeld's algorithm
    def denom(x):
        return lambda j, *p: 1 / sum((p[j] - x) ** 2) ** 0.5

    if y in points:
        return y
    else:
        return general.summation(0, len(points) - 1, num(y), *points) \
               / general.summation(0, len(points) - 1, denom(y), *points)


def spherical_weiszfeld(*points, center=(0, 0, 0), radius=1,
                        iter=100, epsilon=None):
    if len(points) == 1:
        return points[0]
    return_type = 'cartesian'
    if len(points[0]) == 2:
        points = [geographical_to_cartesian(lat, lng, radius)
                  for lat, lng in points]
        return_type = 'geographical'
    elif len(points[0]) != 3:
        raise ValueError('points must be latitude/longitude or xyz')
    pole = same_hemisphere(*points, radius=radius, return_type='central')
    rotated_points = rotate_3d_points_to((0, 0, 0), pole, (0, 0, 1), *points)
    projected_points = project_azimuthal((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                         *rotated_points, radius=radius)
    rotated_median = modified_weiszfeld(*projected_points,
                                        iter=iter, epsilon=epsilon)
    median_z = math.sqrt(radius ** 2 - sum(np.array(rotated_median) ** 2))
    rotated_median = tuple(rotated_median) + (median_z,)
    unrotated_median = rotate_3d_points_to((0, 0, 0), (0, 0, 1),
                                           pole, rotated_median)[0]
    x, y, z = unrotated_median
    return (cartesian_to_geographical(x, y, z, center)
            if return_type == 'geographical' else (x, y, z))


def great_circle_dist(lat1, lng1, lat2, lng2, radius, degrees=False):
    if degrees:
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def great_circle_fraction(lat1, lng1, lat2, lng2, fraction, degrees=False):
    if degrees:
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dist = great_circle_dist(lat1, lng1, lat2, lng2, 1, degrees)
    A = math.sin((1 - fraction) * dist) / math.sin(dist)
    B = math.sin(fraction * dist) / math.sin(dist)
    x = A * math.cos(lat1) * math.cos(lng1) + B * math.cos(lat2) * math.cos(
        lng2)
    y = A * math.cos(lat1) * math.sin(lng1) + B * math.cos(lat2) * math.sin(
        lng2)
    z = A * math.sin(lat1) + B * math.sin(lat2)
    lat = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    lng = math.atan2(y, x)
    return ((math.degrees(lat), math.degrees(lng)) if degrees
            else (lat, lng))


# Rotate points around a center point
#   center:
#       The point to rotate the points around. 2-element sequence (x, y).
#   angle:
#       the angle to rotate the points, in radians. float.
#   *points:
#       the points to rotate. 2 element sequences (x, y).
def rotate_2d_points(center, theta, *points):
    cx, cy = center
    rotated = []
    for px, py in points:
        px, py = px - cx, py - cy
        rotated.append(
            (px * math.cos(2 * theta) - py * math.sin(2 * theta) + cx,
             px * math.sin(2 * theta) + py * math.cos(2 * theta) + cy))
    return rotated


def rotate_3d_points(center, axis, theta, *points):
    cx, cy, cz = center
    u, v, w = np.array(axis) / np.linalg.norm(axis)
    rotated = []
    matrix = np.array([[u ** 2 + (v ** 2 + w ** 2) * math.cos(theta),
                        u * v * (1 - math.cos(theta)) - w * math.sin(theta),
                        u * w * (1 - math.cos(theta)) + v * math.sin(theta),
                        (cx * (v ** 2 + w ** 2) - u * (cy * v + cz * w)) * (
                            1 - math.cos(theta)) + (
                        cy * w - cz * v) * math.sin(theta)],
                       [u * v * (1 - math.cos(theta)) + w * math.sin(theta),
                        v ** 2 + (u ** 2 + w ** 2) * math.cos(theta),
                        v * w * (1 - math.cos(theta)) - u * math.sin(theta),
                        (cy * (u ** 2 + w ** 2) - v * (cx * u + cz * w)) * (
                            1 - math.cos(theta)) + (
                        cz * u - cx * w) * math.sin(theta)],
                       [u * w * (1 - math.cos(theta)) - v * math.sin(theta),
                        v * w * (1 - math.cos(theta)) + u * math.sin(theta),
                        w ** 2 + (u ** 2 + v ** 2) * math.cos(theta),
                        (cz * (u ** 2 + v ** 2) - w * (cx * u + cy * v)) * (
                            1 - math.cos(theta)) + (
                        cx * v - cy * u) * math.sin(theta)],
                       [0,
                        0,
                        0,
                        1]])
    for px, py, pz in points:
        rotated.append(np.dot([px, py, pz, 1], matrix)[:-1])
    return rotated


def rotate_3d_points_to(center, start_point, end_point, *points):
    start_vector = np.array(start_point) - center
    end_vector = np.array(end_point) - center
    if np.array_equal(start_vector, end_vector):
        return points
    axis = (np.cross(start_vector, end_vector) /
            sum(np.cross(start_vector, end_vector) ** 2) ** 0.5)
    theta = -angle(start_vector, [0, 0, 0], end_vector)
    return rotate_3d_points(center, axis, theta, *points)


def project_azimuthal(center, pole, x_axis, *points, radius=1):
    if len(points[0]) == 2:
        points = [geographical_to_cartesian(*p, radius=radius) for p in points]
    elif len(points[0]) != 3:
        raise ValueError('points must be latitude/longitude or xyz')
    if not all([pole[i] == (0, 0, 0)[i] for i in range(3)]):
        points = rotate_3d_points_to(center, pole, (0, 0, 1), *points)
    points = [p[:2] for p in points]
    theta = -angle(rotate_3d_points_to(center, pole, (0, 0, 1), x_axis)[0],
                   (0, 0, 0), (1, 0, 0))
    return rotate_2d_points(center[:2], theta, *points)


# Get the bounding box of a set of points.
#   points:
#       the points to find a bounding box around. tuple or list of 2-element
#       sequences (x,y)
#   aligned = True:
#       if True, return a bounding box aligned to the x and y axes. If False,
#       return a bounding box oriented to the points.
# Returns a list of tuples containing the corners of the bounding box.
def bounding_box(points, aligned=True):
    points = np.array(points)
    # Get an aligned bounding box
    if aligned:
        mins = tuple(np.min(points, axis=0))
        maxes = tuple(np.max(points, axis=0))
        # Find (max, min), (min, max), (min, min), and (max, max)
        product = tuple(itt.product(*zip(mins, maxes)))
        arr = np.array(product)
        box_hull = sp.spatial.ConvexHull(arr)
        # Find the length of the sides via pythagoras
        top = box_hull.points[box_hull.vertices][0:2]
        side = box_hull.points[box_hull.vertices][1:3]
        width, height = (sum((top[0] - top[1]) ** 2) ** 0.5,
                         sum((side[0] - side[1]) ** 2) ** 0.5)
        # Aligned to axes, so rotation is always 0
        theta = 0
    else:
        # Rotating calipers only supports 2D
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError('points does not contain only 2D points.')
        hull = sp.spatial.ConvexHull(points)
        vertices = hull.points[hull.vertices]
        edges = (
            [(vertices[i], vertices[i - 1]) for i in range(len(vertices))])
        best_area = float('inf')
        best_bbox = None
        best_angle = None
        for edge in edges:
            # Get the slope
            rise, run = edge[0] - edge[1]
            # math.atan handles divide by zero on its own
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                # slope to angle
                theta = math.atan(rise / run)
            # Rotate the points so that the edge is aligned with the y axis
            rotated = rotate_2d_points((0, 0), theta, *points)
            # Find the aligned bbox of the rotated points
            rotated_bbox = bounding_box(rotated)
            # Find the width, height, and area of the rotated bbox
            side, top = (np.array(rotated_bbox[0:2]),
                         np.array(rotated_bbox[1:3]))
            w, h = (sum((top[0] - top[1]) ** 2) ** 0.5,
                    sum((side[0] - side[1]) ** 2) ** 0.5)
            new_area = h * w
            # Pick the minimum area bbox
            if new_area < best_area:
                best_bbox = rotate_2d_points((0, 0), -theta, *rotated_bbox)
                best_area = new_area
                best_angle = theta
                bh, bw = h, w
        # Use the aligned bbox if the rotated bbox is no better
        unrotated_bbox = bounding_box(points)
        if unrotated_bbox.height * unrotated_bbox.width > best_area:
            box_hull = sp.spatial.ConvexHull(np.array(best_bbox))
            box_angle = best_angle
            height, width = bh, bw
        else:
            return bounding_box(points)
    box_angle = None
    box_vertices = [tuple(x) for x in box_hull.points[box_hull.vertices]]
    # Make a Bbox object to hold the information
    return BBox(points, box_vertices, box_angle, width, height)


# Get the angle of a line.
#   p1, p2:
#       points to find the angle of a line between. 2-element sequences (x, y).
# Returns the angle from the ray from (0, 0) to (0, inf) in radians.
def line_angle(p1, p2):
    points = np.array((p1, p2))
    i, j = points[1] - points[0]
    return math.atan(i / j)


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    u, v = a - b, c - b
    cos_theta = np.dot(u, v) / np.dot(sum(u ** 2) ** 0.5, sum(v ** 2) ** 0.5)
    return math.acos(cos_theta)


def geographical_to_cartesian(lat, lng, radius=1):
    lat, lng = math.radians(lat), math.radians(lng)
    return (radius * math.cos(lat) * math.cos(lng),
            radius * math.cos(lat) * math.sin(lng),
            radius * math.sin(lat))


def cartesian_to_geographical(x, y, z, center=(0, 0, 0)):
    coords = np.array((x, y, z)) - center
    r = sum(coords ** 2) ** 0.5
    x, y, z = coords / r
    lat = math.asin(z)
    lng = math.atan2(y, x)
    return math.degrees(lat), math.degrees(lng)


def same_hemisphere(*points, radius=1, return_type='exist'):
    EPSILON = 1.0e-10
    poles = []
    if len(points[0]) == 2:
        points = [geographical_to_cartesian(*p, radius=radius) for p in points]
    elif len(points[0]) != 3:
        raise ValueError('points must be latitude/longitude or xyz')
    points = np.array(points)
    for i in points:
        for j in points:
            if np.allclose(i, j, atol=EPSILON):
                continue
            cross = np.cross(i, j)
            pole = cross / np.linalg.norm(cross)
            good_poles = {'+': cross, '-': -cross}
            for k in points:
                dot = np.dot(pole, k)
                if dot + EPSILON < 0 and '+' in good_poles:
                    del good_poles['+']
                if -dot + EPSILON < 0 and '-' in good_poles:
                    del good_poles['-']
            poles += list(good_poles.values())
    if return_type == 'poles':
        return set(general.recursive_seq_change(poles))
    elif return_type == 'central':
        if len(poles) > 0:
            if len(points) <= 2:
                p = np.mean(np.array(points), axis=0)
            else:
                p = np.mean(np.array(poles), axis=0)
            return p / np.linalg.norm(p)
        else:
            return None
    else:
        return len(poles) > 0


class BBox():
    def __init__(self, points, corners, rotation, width, height):
        self.points = np.array(points)
        self.corners = sp.spatial.ConvexHull(corners)
        self.rotation = rotation
        self.width = width
        self.height = height
        self.__setattr__ = self.no_mute

    def no_mute(self, unused_value):
        raise TypeError("'BBox' object does not support item assignment")

    def __getitem__(self, *index):
        try:
            return self.corners.points[self.corners.vertices[index]]
        except IndexError:
            raise IndexError('bbox corner index out of range (0-3)')
