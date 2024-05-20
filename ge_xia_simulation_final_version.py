import math
import numpy as np
import matplotlib.pyplot as plt
from geometry import Point2D, Segment2D, Triangle2D, Arc2D, Circle2D
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from heapq import heappop, heappush


#
# returns a random list of points given the amount
#
def scatter_points(num_points):
    points_list = []
    for i in range(num_points):
        x, y = np.random.rand(2)
        point_id = i
        point = Point2D(x, y, point_id)
        points_list.append(point)
    return points_list


# important example
def points_example():
    return [
        Point2D(4, 5, 0),
        Point2D(5, 5, 1),
        Point2D(5.01, 4.94, 2),
        Point2D(4.01, 5.02, 3),
        Point2D(4.07, 5.14, 4),
    ]


#
# returns all undirected edges in the delaunay triangulation
#
def delaunay_edges_triangles(point2d_list):
    # Extract (x, y) coordinates from Point2D instances
    points = [(point.x, point.y) for point in point2d_list]

    tri = Delaunay(points)
    segments = []
    triangles = []

    for i, simplex in enumerate(tri.simplices):
        # For each simplex (triangle), add its three edges to the set
        simplex0 = get_point_by_id(points=point2d_list, id=simplex[0])
        simplex1 = get_point_by_id(points=point2d_list, id=simplex[1])
        simplex2 = get_point_by_id(points=point2d_list, id=simplex[2])

        triangle = Triangle2D(simplex0, simplex1, simplex2, i)
        triangles.append(triangle)

        segment0 = Segment2D(simplex0, simplex1, i * 3)
        segment1 = Segment2D(simplex1, simplex2, i * 3 + 1)
        segment2 = Segment2D(simplex2, simplex0, i * 3 + 2)

        segments.extend([segment0, segment1, segment2])

    # Eliminate duplicates by converting the list to a set
    unique_segments = list(set(segments))

    return unique_segments, triangles


#
# plot the delaunay triangulation with edges and nodes
#
def show_delaunay(delaunay_nodes, delaunay_edges, circles, arcs, intersection_segments):
    fig, ax = plt.subplots()

    # Plot nodes
    for point in delaunay_nodes:
        ax.plot(point.x, point.y, "o", markersize=8)
        ax.annotate(
            str(point.id),
            (point.x, point.y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    # Plot edges
    for edge in delaunay_edges:
        node1 = edge.point1
        node2 = edge.point2
        color = "orange" if edge in intersection_segments else "black"
        line = Line2D(
            [node1.x, node2.x],
            [node1.y, node2.y],
            linestyle="-",
            linewidth=2,
            color=color,
        )
        ax.add_line(line)

    # Plot circles
    for circle in circles:
        center = circle.center
        radius = circle.radius
        circle_plot = plt.Circle((center.x, center.y), radius, color="b", fill=False)
        ax.plot(center.x, center.y, "o", markersize=8)
        ax.annotate(
            "M"
            + str(circle.triangle.point1.id)
            + str(circle.triangle.point2.id)
            + str(circle.triangle.point3.id),
            (center.x, center.y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
        ax.add_patch(circle_plot)

    # Plot arcs
    for arc in arcs:
        arc_plot = Arc(
            (arc.middle_point.x, arc.middle_point.y),
            2 * arc.radius(),
            2 * arc.radius(),
            theta1=math.degrees(arc.theta1()),
            theta2=math.degrees(arc.theta2()),
            linewidth=2,
            color="orange",
        )
        ax.add_patch(arc_plot)

    ax.set_aspect("equal", adjustable="datalim")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Delaunay Triangulation")
    plt.show()


#
# Returns all possible segments uv from a set of points
#
def segments(points, del_edges):
    segment_list = []
    count = 0
    for i, point0 in enumerate(points):
        for j, point1 in enumerate(points):
            new_segment = Segment2D(point0, point1, count)
            if i < j and new_segment not in del_edges:
                # segment_list.append(new_segment)
                return [new_segment]
                count += 1

    return []
    # return segment_list


#
# Returns the first non trivial edge found
#
def get_current_segment(points, del_edges):
    for i, point0 in enumerate(points):
        for j, point1 in enumerate(points):
            new_segment = Segment2D(point0, point1)
            if i < j and new_segment not in del_edges:
                return new_segment
    return None


#
# Returns the smallest possible circle with point1 and point2 on its edge
#
def get_smallest_circle(point1, point2):
    center_x = (point1.x + point2.x) / 2
    center_y = (point1.y + point2.y) / 2
    center = Point2D(center_x, center_y)
    radius = center.distance_to(point1)
    return Circle2D(center, radius)


#
# Returns the common edge of two triangles. Will be None if no common edge is found.
#
def get_adjacent_triangle_segment(triangle_1, triangle_2):
    result = [
        segment
        for segment in triangle_1.get_segments()
        if segment in triangle_2.get_segments()
    ]
    if len(result) != 1:
        return None

    return result[0]


#
# Returns a point with the certain id.
#
def get_point_by_id(points, id):
    for point in points:
        if point.id == id:
            return point


#
# Returns the triangle of the three points witth ids point_id_1, point_id_2, point_id_3.
#
def get_triangle_by_point_ids(triangles, point_id_1, point_id_2, point_id_3):
    for tri in triangles:
        ids = {tri.point1.id, tri.point2.id, tri.point3.id}
        if ids == {point_id_1, point_id_2, point_id_3}:
            return tri
    return None


#
# Returns the orthogonal projection of a point on a segment
#
def point_on_segment_projection(point, segment):
    x = np.array([point.x, point.y])
    u = np.array([segment.point1.x, segment.point1.y])
    v = np.array([segment.point2.x, segment.point2.y])

    n = v - u
    n /= np.linalg.norm(n, 2)

    projection_point = u + np.dot(x - u, n) * n
    return Point2D(projection_point[0], projection_point[1])


#
# Returns sorted list of circles by projection of the middle point and its distance to point 1 of the current_segment.
#
def sort_circles_by_projection(circles, segment):
    return sorted(
        circles,
        key=lambda circle: segment.point1.distance_to(
            point_on_segment_projection(circle.center, segment)
        ),
    )


#
# Returns all triangles which contain a certain segment.
#
def get_triangle_with_segment(triangles, segment):
    for tri in triangles:
        if segment in tri.get_segments():
            return tri


#
# Returns the orientation of a point to a segment. 0 = the point lies on the segment. 1 = point lies on one side of the segment. -1 = point lies on the other side of the line.
#
def orientation(segment, query_point):
    det = (segment.point2.x - segment.point1.x) * (query_point.y - segment.point1.y) - (
        query_point.x - segment.point1.x
    ) * (segment.point2.y - segment.point1.y)

    return np.sign(det)


#
# Returns the shortest path per Dijkstra algorithm.
#
def shortest_path(start, end, nodes, edges):
    # Initialize distances dictionary with infinity for all nodes except start
    distances = {node: float("inf") for node in nodes}
    distances[start] = 0

    # Priority queue to store nodes with their current distances
    priority_queue = [(0, start)]

    # Previous nodes in the shortest path
    previous_nodes = {node: None for node in nodes}

    while priority_queue:
        current_distance, current_node = heappop(priority_queue)

        # Check if the current node is the destination
        if current_node == end:
            path = []
            while previous_nodes[current_node] is not None:
                path.append(Segment2D(previous_nodes[current_node], current_node))
                current_node = previous_nodes[current_node]
            # Reverse the path to get the correct order
            return path[::-1]

        for edge in edges:
            if current_node in edge.points():
                neighbor = (
                    edge.points()[0]
                    if current_node == edge.points()[1]
                    else edge.points()[1]
                )
                distance = (
                    distances[current_node] + edge.length()
                )  # Corrected distance calculation
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heappush(priority_queue, (distance, neighbor))

    # If no path is found
    return None


#
# Returns all upper arcs of the G. Xia path.
#
def arcs_A0_AN(sorted_circles, outer_segments, current_segment, triangles):
    arcs = []
    current_point = current_segment.point1

    for i in range(len(sorted_circles) - 1):
        tri = sorted_circles[i].triangle
        tri_next = sorted_circles[i + 1].triangle
        for point in tri_next.get_points():
            next_segment = Segment2D(current_point, point)

            if (
                next_segment in outer_segments
                and orientation(current_segment, point) >= 0
            ):
                arcs.append(
                    Arc2D(
                        start_point=point,
                        middle_point=get_triangle_with_segment(triangles, next_segment)
                        .outer_circle()
                        .center,
                        end_point=current_point,
                    )
                )
                current_point = point
                break
        # check if last point is terminal point
    if current_point != current_segment.point2:
        arcs.append(
            Arc2D(
                current_segment.point2,
                sorted_circles[len(sorted_circles) - 1].center,
                current_point,
            )
        )
    return arcs


#
# Returns all lower arcs of the G. Xia path.
#
def arcs_B0_BN(sorted_circles, outer_segments, current_segment, triangles):
    arcs = []
    current_point = current_segment.point1

    for i in range(len(sorted_circles) - 1):
        tri = sorted_circles[i].triangle
        tri_next = sorted_circles[i + 1].triangle
        for point in tri_next.get_points():
            next_segment = Segment2D(current_point, point)

            if (
                next_segment in outer_segments
                and orientation(current_segment, point) <= 0
            ):
                arcs.append(
                    Arc2D(
                        start_point=current_point,
                        middle_point=get_triangle_with_segment(triangles, next_segment)
                        .outer_circle()
                        .center,
                        end_point=point,
                    )
                )
                current_point = point
                break
    # check if last point is terminal point
    if current_point != current_segment.point2:
        arcs.append(
            Arc2D(
                current_point,
                sorted_circles[len(sorted_circles) - 1].center,
                current_segment.point2,
            )
        )
    return arcs


#
# Returns all triangles which are intersected by the line uv (i.e. current_segment).
#
def get_intersection_triangles(del_triangles, current_segment):
    intersecting_triangles = []
    for tri in del_triangles:
        if tri.intersects(current_segment):
            intersecting_triangles.append(tri)
    return intersecting_triangles


#
# Returns all circles of the triangles which are intersected by the line uv (i.e. current_segment).
#
def get_circles_from_triangles(intersecting_triangles):
    circles = []
    for tri in intersecting_triangles:
        circles.append(tri.outer_circle())
    return circles


#
# Returns all segments which are intersected by the line uv (i.e. current_segment).
#
def get_intersection_segments(sorted_circles):
    intersections_segments = []
    for i in range(len(sorted_circles) - 1):
        intersections_segments.append(
            get_adjacent_triangle_segment(
                sorted_circles[i].triangle, sorted_circles[i + 1].triangle
            )
        )
    return intersections_segments


#
# Returns all segments which are not intersected by the line uv (i.e. current_segment).
#
def get_outer_segments(intersecting_triangles, intersections_segments):
    outer_segments = []
    for tri in intersecting_triangles:
        for seg in tri.get_segments():
            if seg not in intersections_segments:
                outer_segments.append(seg)
    return outer_segments


#
# Returns the length of a path.
#
def path_length(path):
    length = 0
    for edge in path:
        length += edge.length()
    return length


#
# Print all necessary stuff.
#
def print_all(
    points,
    current_segment,
    sorted_circles,
    intersections_segments,
    arcs_A_i,
    arcs_B_i,
    path_gx,
    spanning_ratio_gx,
    alternate_path=None,
    alternate_path_spanning_ratio=None,
):
    print("====================================")
    print("POINTS: ", points)
    print("CURRENT SEGMENT: ", current_segment)
    print("SORTED CIRCLES: ", sorted_circles)
    print("INTERSECTION SEGMENTS: ", intersections_segments)
    print("ARCS A0-AN: ", arcs_A_i)
    print("ARCS B0-BN: ", arcs_B_i)
    print("PATH GE_XIA: ", path_gx)
    print("SPANNING RATIO GX: ", spanning_ratio_gx)
    print("PATH ALTERNATE: ", alternate_path)
    print("SPANNING RATIO ALTERNATE: ", alternate_path_spanning_ratio)
    print("====================================")


#
# Returns the shortest path, together with its spanning ratio.
#
def shortest_path_construction(del_edges, points, current_segment):
    # actual shortest path
    path_shortest = shortest_path(
        current_segment.point1,
        current_segment.point2,
        points,
        del_edges,
    )

    spanning_ratio_shortest = path_length(path_shortest) / current_segment.length()
    return path_shortest, spanning_ratio_shortest


#
# Returns the G. Xia path together with all points, edges and arcs.
#
def ge_xia_path_construction(del_triangles, current_segment):
    if current_segment is None:
        return None
    # current_segment = Segment2D(
    #     get_point_by_id(points, 4), get_point_by_id(points, 0), id=0
    # )
    intersecting_triangles = get_intersection_triangles(del_triangles, current_segment)

    # all circles around the intersection triangles
    circles = get_circles_from_triangles(intersecting_triangles)

    # sorted circles by the projection onto the connecting line by distance to one terminal point
    sorted_circles = sort_circles_by_projection(circles, current_segment)

    # a_ib_i
    intersections_segments = get_intersection_segments(sorted_circles)

    for segment in intersections_segments:
        if segment is None:
            return None

    # segments which are not present in two triangles
    outer_segments = get_outer_segments(intersecting_triangles, intersections_segments)

    # A_0...A_N
    arcs_A_i = arcs_A0_AN(
        sorted_circles=sorted_circles,
        outer_segments=outer_segments,
        current_segment=current_segment,
        triangles=intersecting_triangles,
    )

    # B_0...#B_N
    arcs_B_i = arcs_B0_BN(
        sorted_circles=sorted_circles,
        outer_segments=outer_segments,
        current_segment=current_segment,
        triangles=intersecting_triangles,
    )

    # all arcs
    arcs = arcs_A_i + arcs_B_i

    # allowed edges are arcs and intersection segments
    gx_graph_edges = arcs + intersections_segments
    gx_dual_graph_nodes = set(
        point for edge in gx_graph_edges for point in edge.points()
    )

    # ge xia path
    path_gx = shortest_path(
        current_segment.point1,
        current_segment.point2,
        gx_dual_graph_nodes,
        gx_graph_edges,
    )
    spanning_ratio_gx = path_length(path_gx) / current_segment.length()

    return (
        intersecting_triangles,
        circles,
        sorted_circles,
        intersections_segments,
        outer_segments,
        arcs_A_i,
        arcs_B_i,
        arcs,
        gx_graph_edges,
        gx_dual_graph_nodes,
        path_gx,
        spanning_ratio_gx,
    )


#
# Returns an invalid edge in the path (edge is too long). None is returned if no too long edge is found.
#
def get_invalid_edge(path, current_segment):
    for seg in path:
        if seg.length() > current_segment.length():
            return seg
    return None


#
# Returns 0 if the long edge has a common node with the current_segment.
# Returns 1 if the long edge has one point inside the smallest circle and the other one outside and if the edge intersects the current_segment.
# Returns 2 if the long edge has one point inside the smallest circle and the other one outside and if the edge does not intersects the current_segment.
# Returns 3 otherwise.
#
def determine_edge_position_case(edge, current_segment):
    terminal_point1 = current_segment.point1
    terminal_point2 = current_segment.point2
    smallest_circle = get_smallest_circle(terminal_point1, terminal_point2)

    if (
        terminal_point1 == edge.point1
        or terminal_point1 == edge.point2
        or terminal_point2 == edge.point1
        or terminal_point2 == edge.point2
    ):

        return 0
    elif (
        smallest_circle.contains_point(edge.point1)
        and not smallest_circle.contains_point(edge.point2)
        or smallest_circle.contains_point(edge.point2)
        and not smallest_circle.contains_point(edge.point1)
    ):
        if edge.intersects(current_segment):
            return 1
        else:
            return 2
    else:
        return 3


#
# Returns whether a short way exists and its spanning ratio
#
def has_short_way(path_gx, spanning_ratio_gx, del_edges, current_segment, points):

    alternate_path = path_gx
    alternate_path_spanning_ratio = spanning_ratio_gx
    del_edges_copy = del_edges.copy()

    for edge in del_edges_copy:
        if edge.length() > current_segment.length():
            del_edges.remove(edge)
    (
        alternate_path,
        alternate_path_spanning_ratio,
    ) = shortest_path_construction(del_edges, points, current_segment)
    if alternate_path_spanning_ratio > spanning_ratio_gx:
        return (False, alternate_path_spanning_ratio)
    else:
        return (True, alternate_path_spanning_ratio)


if __name__ == "__main__":
    N = 10000000
    node_amount = [5]
    counts_edge_position = []
    counts_contains_larger_edge = []
    counts_no_short_way = []
    invalid_construction = 0
    max_spanning_ratio = 0
    max_alternate_way_spanning_ratio = 0

    for nodes in node_amount:
        count_edge_position = [0, 0, 0, 0]
        count_contains_larger_edge = 0
        count_no_short_way = 0

        for i in range(N):
            valid_construction = False

            while not valid_construction:
                points = scatter_points(nodes)
                del_edges, del_triangles = delaunay_edges_triangles(points)
                current_segment = get_current_segment(points, del_edges)

                result = ge_xia_path_construction(del_triangles, current_segment)
                if result is not None:
                    valid_construction = True
                    (
                        intersecting_triangles,
                        circles,
                        sorted_circles,
                        intersections_segments,
                        outer_segments,
                        arcs_A_i,
                        arcs_B_i,
                        arcs,
                        gx_graph_edges,
                        gx_graph_nodes,
                        path_gx,
                        spanning_ratio_gx,
                    ) = result

                    max_spanning_ratio = max(max_spanning_ratio, spanning_ratio_gx)
                    invalid_edge = get_invalid_edge(path_gx, current_segment)

                    if invalid_edge != None:

                        count_contains_larger_edge += 1

                        case_number = determine_edge_position_case(
                            invalid_edge, current_segment
                        )
                        count_edge_position[case_number] += 1

                        is_short_way, alternate_way_spanning_ratio = has_short_way(
                            path_gx,
                            spanning_ratio_gx,
                            del_edges,
                            current_segment,
                            points,
                        )

                        if not is_short_way:
                            count_no_short_way += 1
                            max_alternate_way_spanning_ratio = max(
                                max_alternate_way_spanning_ratio,
                                alternate_way_spanning_ratio,
                            )

                        print_all(
                            points,
                            current_segment,
                            sorted_circles,
                            intersections_segments,
                            arcs_A_i,
                            arcs_B_i,
                            path_gx,
                            spanning_ratio_gx,
                        )
                        print("Finished: ", nodes, i / N)
                        if case_number != 1:
                            show_delaunay(
                                points, del_edges, circles, arcs, intersections_segments
                            )
                else:
                    invalid_construction += 1

        counts_edge_position.append(count_edge_position)
        counts_contains_larger_edge.append(count_contains_larger_edge)
        counts_no_short_way.append(count_no_short_way)

    print("counts_edge_position: ", counts_edge_position)
    print("counts_contains_larger_edge: ", counts_contains_larger_edge)
    print("counts_no_short_way: ", counts_no_short_way)
    print("invalid_constructions: ", invalid_construction)
    print("max_spanning_ratio: ", max_spanning_ratio)
    print("max_alternate_way_spanning_ratio: ", max_alternate_way_spanning_ratio)
