"""
Facilitates the *svg_to_polylines()* function for reading an SVG file (with <path> elements)
and converting it into a list of polylines.
Uses svgpathtools to parse paths and approximate into straight line segments.

:author: Ingo Breßler, May 14, 2026
:license: MIT
"""

import re
import xml.etree.ElementTree as ET
from itertools import groupby
from pathlib import Path

import numpy as np
import svgpathtools
from svgpathtools import parse_path


def sample_path_to_lines(path: svgpathtools.Path, flatness: float = 0.5, max_depth: int = 18):
    """
    Convert an SVG path to a series of line segments based on the specified flatness and
    recursive depth limits.

    This function takes an SVG path, evaluates each of its segments, and approximates
    them as a series of line segments by recursively subdividing until the segment is
    flat enough or the maximum recursion depth is reached. This process converts curves
    into linear approximations for further processing, rendering, or analysis.

    :param path: An object representing an SVG path, typically containing a sequence of
        segments that can be evaluated and manipulated through appropriate methods.
    :type path: svgpathtools.Path
    :param flatness: Determines the allowable deviation of intermediate points from a
        straight line. Segments with deviations greater than this value are further
        subdivided. Default is 0.5.
    :type flatness: float
    :param max_depth: The maximum recursion depth used during segment subdivision. A
        higher value enables finer approximations but may increase computation time.
        Default is 18.
    :type max_depth: int
    :return: A list of sampled points representing the linear approximation of the SVG
        path. Each point is a tuple containing the x and y coordinates (float) of the
        sampled location.
    :rtype: list of tuple(float, float)
    """

    def point_line_distance(p, a, b):
        """
        Calculate the perpendicular distance from a point to a line segment.

        This function computes the shortest distance from a given point to a
        line segment defined by two endpoints. The calculation considers
        special cases where the line segment degenerates into a single point
        or where the point projects beyond the line segment endpoints. The
        result is always non-negative.

        :param p: The complex number representing the point whose distance
            from the line segment is to be calculated.
        :param a: The complex number representing the starting endpoint of
            the line segment.
        :param b: The complex number representing the ending endpoint of
            the line segment.

        :return: The shortest distance from the point `p` to the line
            segment defined by `a` and `b`.
        :rtype: float
        """
        p = np.array([p.real, p.imag], dtype=float)
        a = np.array([a.real, a.imag], dtype=float)
        b = np.array([b.real, b.imag], dtype=float)

        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        if denom == 0.0:
            return np.linalg.norm(ap)

        t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def segment_is_flat_enough(seg):
        """
        Determines whether a given segment is flat enough based on its deviation from a
        straight line connecting its endpoints.

        This function calculates the maximum perpendicular distance of intermediate
        control points of the segment from the straight line defined by its start and
        end points. If this maximum distance is less than or equal to the threshold of
        flatness, the segment is deemed to be flat enough.

        :param seg: A segment object that provides access to its points via the
            `point` method. The `point` method accepts a parameter ranging from 0.0 to
            1.0, where 0.0 corresponds to the start of the segment and 1.0 corresponds
            to the end of the segment.
        :return: A boolean indicating whether the segment is flat enough.
        :rtype: bool
        """
        p0 = seg.point(0.0)
        p1 = seg.point(1.0)
        m1 = seg.point(1 / 3)
        m2 = seg.point(2 / 3)
        return (
            max(
                point_line_distance(m1, p0, p1),
                point_line_distance(m2, p0, p1),
            )
            <= flatness
        )

    def subdivide_segment(seg, depth, out_points):
        """
        Recursively subdivides a segment into smaller segments until a maximum depth is
        reached or the segment is flat enough. Subdivision ensures precision in curve
        rendering by sampling more points along the segment.

        :param seg: The segment to be processed. It must have `point` and `split`
            methods. The `point` method retrieves a point on the segment for a
            given parameter (float), and the `split` method divides the segment into
            two smaller segments at a given parameter.
        :type seg: object
        :param depth: The current depth of the recursive subdivision. This parameter
            tracks the recursion level with respect to the maximum depth.
        :type depth: int
        :param out_points: A list to which the sampled points will be appended. Points
            from the segment subdivision will be added in sequential order.
        :type out_points: list of tuple(float, float)
        :return: None
        """
        p0 = seg.point(0.0)
        p1 = seg.point(1.0)

        if depth >= max_depth or segment_is_flat_enough(seg):
            if not out_points or out_points[-1] != (p0.real, p0.imag):
                out_points.append((p0.real, p0.imag))
            out_points.append((p1.real, p1.imag))
            return

        left, right = seg.split(0.5)
        subdivide_segment(left, depth + 1, out_points)
        if out_points and out_points[-1] == (right.point(0.0).real, right.point(0.0).imag):
            out_points.pop()
        subdivide_segment(right, depth + 1, out_points)

    points = []
    for seg in path:
        subdivide_segment(seg, 0, points)

    return points


def transform_point(x, y, matrix):
    """
    Transforms a given point using a 2D linear transformation matrix.

    This function applies a 2D linear transformation to the given point `(x, y)`
    based on the provided transformation matrix. The transformation matrix is
    expected to be a sequence of six numbers representing the coefficients in
    row-major order: [a, b, c, d, e, f].

    :param x: The x-coordinate of the point to transform.
    :param y: The y-coordinate of the point to transform.
    :param matrix: A sequence of six numbers representing the transformation matrix
        in the order: [a, b, c, d, e, f].
    :return: A tuple containing the transformed x and y coordinates.
    :rtype: tuple
    """
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def apply_transform_to_points(points, matrix):
    """
    Apply a transformation matrix to a list of points.

    This function takes a list of 2D points and applies a transformation matrix to each
    point. It utilizes an internal helper function to perform the point transformation
    and returns a new list containing the transformed points.

    :param points: A list of tuples, where each tuple represents a point in 2D space
        as (x, y).
    :param matrix: A 2D transformation matrix used to apply the transformation to
        the points. Expected to be in the form of a 2x2 or 3x3 matrix depending on
        the specific transform.
    :return: A list of transformed points, represented as tuples in 2D space. The
        transformed points correspond to the input `points` after the application
        of the given `matrix`.
    :rtype: list[tuple[float, float]]
    """
    transformed = []
    for x, y in points:
        transformed.append(transform_point(x, y, matrix))
    return transformed


def multiply_matrices(m1, m2):
    """
    Multiply two 2D transformation matrices represented in an affine form.

    This function takes two transformation matrices, each represented as a tuple
    of six numbers. These numbers follow the order of the affine transformation:
    (a, b, c, d, e, f), where the components map to the matrix structure:

        | a  c  e |
        | b  d  f |
        | 0  0  1 |

    The multiplication follows matrix multiplication rules to calculate the
    resulting transformation.

    :param m1: The first transformation matrix as a tuple (a, b, c, d, e, f).
        Represents the transformation matrix to be multiplied from the left.
    :param m2: The second transformation matrix as a tuple (a, b, c, d, e, f).
        Represents the transformation matrix to be multiplied from the right.
    :return: The resulting transformation matrix as a tuple (a, b, c, d, e, f),
        resulting from the multiplication of m1 and m2.
    :rtype: tuple
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def parse_transform(transform):
    """
    Parses a string representation of a transform and computes the resulting transformation matrix.
    The input string may include various transformations such as `translate`, `scale`, and `matrix`
    operations, which will be combined into a single 2D affine transformation matrix.

    :param transform: A string representing transformation operations in the format of
        `translate(x, y)`, `scale(sx, sy)`, or `matrix(a, b, c, d, e, f)`. The operations,
        if valid, will be accumulated in the given sequence. If the string is empty or None,
        the identity matrix is returned.

    :return: A tuple representing the resultant 2D affine transformation matrix in the form
        `(a, b, c, d, e, f)`.
    """
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    if not transform:
        return matrix

    for name, args_str in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", transform):
        values = [float(v) for v in re.split(r"[,\s]+", args_str.strip()) if v]

        if name == "translate":
            tx = values[0]
            ty = values[1] if len(values) > 1 else 0.0
            step = (1.0, 0.0, 0.0, 1.0, tx, ty)

        elif name == "scale":
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            step = (sx, 0.0, 0.0, sy, 0.0, 0.0)

        elif name == "matrix":
            if len(values) != 6:
                raise ValueError(f"Invalid matrix transform: {transform}")
            step = tuple(values)

        else:
            raise ValueError(f"Unsupported transform '{name}' in: {transform}")

        matrix = multiply_matrices(matrix, step)

    return matrix


def collect_transformed_line_paths(svg_file, output_scale=1.0):
    """
    Collects and processes line paths from an SVG file, applies transformations, and scales
    them according to the specified output scale. The resulting line paths are transformed
    to adhere to a uniform design scale while maintaining proportionality based on the
    dimensions of the SVG file.

    :param svg_file: The path to the SVG file to process.
    :type svg_file: str
    :param output_scale: A scaling factor to apply to the extracted paths and computed
                         bounds. Defaults to 1.0.
    :type output_scale: float
    :return: A tuple containing:
              - A list of transformed and scaled line paths. Each path is represented
                as a list of coordinate tuples (x, y).
              - Bounds of the SVG content as a tuple (min_x, min_y, max_x, max_y)
                scaled according to the output scale, or `None` if no paths are present.
    :rtype: tuple[list[list[tuple[int, int]]], tuple[float, float, float, float] or None]
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    all_line_paths = []
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    designWidth = float(root.attrib["width"].strip("m"))
    designHeight = float(root.attrib["height"].strip("m"))
    designScaleX = output_scale
    designScaleY = output_scale

    for g in root.findall(".//svg:g", ns):
        group_matrix = parse_transform(g.get("transform"))

        for path_el in g.findall("./svg:path", ns):
            d = path_el.get("d")
            if not d:
                continue

            path = parse_path(d)
            if len(path) == 0:
                continue

            sampled_points = sample_path_to_lines(path)
            line_points = apply_transform_to_points(
                sampled_points,
                group_matrix,
            )

            if len(line_points) < 2:
                continue

            for x, y in line_points:
                if min_x is None or x < min_x:
                    min_x = x
                if min_y is None or y < min_y:
                    min_y = y
                if max_x is None or x > max_x:
                    max_x = x
                if max_y is None or y > max_y:
                    max_y = y

            # print(f"{(max_x - min_x)=:.1f}, {(max_y - min_y)=:.1f}")
            # print(f"{(designWidth)=:.1f}, {(designHeight)=:.1f}")
            designScaleX = output_scale * float(designWidth) / (max_x - min_x)
            designScaleY = output_scale * float(designHeight) / (max_y - min_y)

            scaledPoints = [((x * designScaleX), (y * designScaleY)) for x, y in line_points]
            all_line_paths.append(scaledPoints)

    bounds = (
        None
        if min_x is None
        else (
            min_x * designScaleX,
            min_y * designScaleY,
            max_x * designScaleX,
            max_y * designScaleY,
        )
    )
    return all_line_paths, bounds


def svg_to_polylines(
    svg_file: Path,
    scale: float = 1.0,
    z: float | None = None,
    upper_left: tuple[float, float] | None = None,
):
    """
    Converts an SVG file into a list of polylines, considering scaling and optional z-coordinates.

    The function reads an SVG file, processes its geometry to extract line paths,
    and transforms them into polylines. If applicable, it scales the polylines and assigns
    the optional z-coordinate to each point. If `upper_left` is specified, all paths are
    moved such that their common bounding box matches that corner coordinate.
    A summary of geometry size and extents is printed if bounds are available.

    :param svg_file: The path to the SVG file to be converted.
    :type svg_file: Path
    :param scale: The scaling factor applied to the geometry. Defaults to 1.0.
    :type scale: float
    :param z: The optional z-coordinate to assign to each point in the polylines. Defaults to None.
    :type z: float or None
    :param upper_left: The optional (x, y) coordinate for the most upper-left corner of the
                       common bounding box. Defaults to None.
    :type upper_left: tuple[float, float] or None
    :return: A list of numpy arrays where each array represents a polyline, containing
             either 2D or 3D points depending on the input parameters.
    :rtype: list[numpy.ndarray]
    """
    line_paths, bounds = collect_transformed_line_paths(str(svg_file), output_scale=scale)

    # get an array from the paths to recalc bounds and offset
    arr = np.array([pt for line_points in line_paths for pt, _ in groupby(line_points)])
    if not len(arr):
        print("No drawable paths found.")
        return [], None

    p_min = arr.min(axis=0)
    p_max = arr.max(axis=0)
    size = p_max - p_min
    offset_x = upper_left[0] - p_min[0] if upper_left is not None else 0.0
    offset_y = upper_left[1] - (size[1] - p_min[1]) if upper_left is not None else 0.0
    print(f"Geometry size: {size[0]:.1f} x {size[1]:.1f}")
    print(f"Geometry min/max: ({p_min[0]:.1f}, {p_min[1]:.1f}), ({p_max[0]:.1f}, {p_max[1]:.1f})")

    polylines = []
    for line_points in line_paths:
        # Flip y-axis to match ROS convention
        points = [(pt[0], size[1] - pt[1]) for pt, _ in groupby(line_points)]
        if upper_left is not None:
            points = [(pt[0] + offset_x, pt[1] + offset_y) for pt in points]
        if z is not None:
            points = [(pt[0], pt[1], z) for pt in points]
        polylines.append(np.array(points))

    bounds = np.hstack(
        (
            np.stack(tuple(path.min(axis=0) for path in polylines)).min(axis=0),
            np.stack(tuple(path.max(axis=0) for path in polylines)).max(axis=0),
        )
    )

    return polylines, bounds
