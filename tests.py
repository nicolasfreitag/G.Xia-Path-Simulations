import unittest
import math
import numpy as np
from geometry import Point2D, Segment2D, Triangle2D, Circle2D, Arc2D
from ge_xia_simulation_final_version import (
    point_on_segment_projection,
    get_point_by_id,
    determine_edge_position_case,
    get_adjacent_triangle_segment,
    orientation,
    shortest_path,
    get_current_segment,
    delaunay_edges_triangles,
    ge_xia_path_construction,
    get_invalid_edge,
    has_short_way,
)


def points_example():
    return [
        Point2D(4, 5, 0),
        Point2D(5, 5, 1),
        Point2D(5.01, 4.94, 2),
        Point2D(4.01, 5.02, 3),
        Point2D(4.07, 5.14, 4),
    ]


def points_example_2():
    return [
        Point2D(-7.4, 3.01, 0),
        Point2D(-7.02, -0.83, 1),
        Point2D(0.74, 5.25, 2),
        Point2D(1.38, -0.89, 3),
        Point2D(-4.88, -2.81, 4),
        Point2D(2.24, 2.75, 5),
        Point2D(-2.26, 1.65, 6),
    ]


def points_example_3():
    return [
        Point2D(4, 5, 0),
        Point2D(5, 5, 1),
        Point2D(5.01, 4.94, 2),
        Point2D(4.01, 5.02, 3),
    ]


class TestSegmentIntersection(unittest.TestCase):
    def test_segment_intersection(self):
        points = points_example()
        segment_1 = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 1), id=0
        )
        segment_2 = Segment2D(
            get_point_by_id(points, 4), get_point_by_id(points, 2), id=0
        )
        segment_3 = Segment2D(
            get_point_by_id(points, 3), get_point_by_id(points, 2), id=0
        )
        segment_4 = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 2), id=0
        )
        segment_5 = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 3), id=0
        )
        segment_6 = Segment2D(
            get_point_by_id(points, 4), get_point_by_id(points, 1), id=0
        )
        segment_7 = Segment2D(
            get_point_by_id(points, 4), get_point_by_id(points, 3), id=0
        )
        # Case 1: Intersection
        self.assertTrue(segment_1.intersects(segment_2))
        self.assertTrue(segment_1.intersects(segment_3))

        # Case 2: Terminal point intersection
        self.assertFalse(segment_1.intersects(segment_4))
        self.assertFalse(segment_1.intersects(segment_5))

        # Case 3: No intersection
        self.assertFalse(segment_6.intersects(segment_3))
        self.assertFalse(segment_4.intersects(segment_7))


class TestAdjacentTriangleSegment(unittest.TestCase):
    def test_adjacent_triangle_segment(self):
        points = points_example()
        triangle_1 = Triangle2D(
            get_point_by_id(points, 0),
            get_point_by_id(points, 2),
            get_point_by_id(points, 3),
            id=0,
        )
        triangle_2 = Triangle2D(
            get_point_by_id(points, 2),
            get_point_by_id(points, 3),
            get_point_by_id(points, 4),
            id=1,
        )
        triangle_3 = Triangle2D(
            get_point_by_id(points, 2),
            get_point_by_id(points, 1),
            get_point_by_id(points, 4),
            id=2,
        )

        # Case 1: Triangles have an adjacent segment
        adjacent_segment = get_adjacent_triangle_segment(triangle_1, triangle_2)
        self.assertIsNotNone(adjacent_segment)
        self.assertIn(adjacent_segment, triangle_1.get_segments())
        self.assertIn(adjacent_segment, triangle_2.get_segments())

        # Case 2: Triangles do not have an adjacent segment
        adjacent_segment = get_adjacent_triangle_segment(triangle_1, triangle_3)
        self.assertIsNone(adjacent_segment)


class TestOrientation(unittest.TestCase):
    def test_orientation(self):
        segment = Segment2D(Point2D(0, 0), Point2D(1, 1))
        query_point1 = Point2D(1, 0)
        query_point2 = Point2D(0.5, 0.5)
        query_point3 = Point2D(0, 1)
        self.assertEqual(orientation(segment, query_point1), -1)
        self.assertEqual(orientation(segment, query_point2), 0)
        self.assertEqual(orientation(segment, query_point3), 1)


class TestArcLength(unittest.TestCase):
    def test_arc_length(self):
        point1 = Point2D(0, 0)
        point2 = Point2D(1, 1)
        point3 = Point2D(0, 1)
        arc1 = Arc2D(point1, point3, point2)
        arc2 = Arc2D(point2, point3, point1)
        self.assertEqual(arc1.length(), (2 * np.pi) / 4)
        self.assertEqual(arc2.length(), (2 * np.pi * 3) / 4)


class TestEdgePosition(unittest.TestCase):
    def test_determine_edge_position_case(self):
        points = [
            Point2D(-1, 0, 0),
            Point2D(1, 0, 1),
            Point2D(-1, 1, 2),
            Point2D(1, -1, 3),
            Point2D(0, 0.5, 4),
            Point2D(0, -0.5, 5),
        ]
        current_segment = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 1)
        )
        segment_1 = Segment2D(get_point_by_id(points, 2), get_point_by_id(points, 3))
        segment_2 = Segment2D(get_point_by_id(points, 0), get_point_by_id(points, 3))
        segment_3 = Segment2D(get_point_by_id(points, 4), get_point_by_id(points, 3))
        segment_4 = Segment2D(get_point_by_id(points, 5), get_point_by_id(points, 3))

        self.assertEquals(determine_edge_position_case(segment_1, current_segment), 3)
        self.assertEquals(determine_edge_position_case(segment_2, current_segment), 0)
        self.assertEquals(determine_edge_position_case(segment_3, current_segment), 1)
        self.assertEquals(determine_edge_position_case(segment_4, current_segment), 2)


class TestPointProjection(unittest.TestCase):
    def test_point_projection(self):
        points = [
            Point2D(-1, 0, 0),
            Point2D(1, 0, 1),
            Point2D(-1, 1, 2),
            Point2D(1, -1, 3),
            Point2D(1, 1, 4),
        ]
        segment_1 = Segment2D(get_point_by_id(points, 0), get_point_by_id(points, 1))
        segment_2 = Segment2D(get_point_by_id(points, 2), get_point_by_id(points, 3))

        self.assertAlmostEqual(
            point_on_segment_projection(get_point_by_id(points, 4), segment_2).x, 0
        )
        self.assertAlmostEqual(
            point_on_segment_projection(get_point_by_id(points, 4), segment_2).y, 0
        )
        self.assertAlmostEqual(
            point_on_segment_projection(get_point_by_id(points, 4), segment_1).x, 1
        )
        self.assertAlmostEqual(
            point_on_segment_projection(get_point_by_id(points, 4), segment_1).y, 0
        )


class TestShortestPath(unittest.TestCase):
    def test_shortest_path(self):
        points = points_example()
        del_edges, del_triangles = delaunay_edges_triangles(points)
        current_segment = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 1)
        )
        path = shortest_path(
            current_segment.point1,
            current_segment.point2,
            points,
            del_edges,
        )

        self.assertEqual(len(path), 2)
        self.assertTrue(
            path[0], Segment2D(get_point_by_id(points, 0), get_point_by_id(points, 2))
        )
        self.assertTrue(
            path[1], Segment2D(get_point_by_id(points, 2), get_point_by_id(points, 1))
        )


class TestGeXiaPath(unittest.TestCase):
    def test_ge_xia_path(self):
        points = points_example()
        del_edges, del_triangles = delaunay_edges_triangles(points)
        current_segment = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 1)
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx,
            _,
        ) = ge_xia_path_construction(del_triangles, current_segment)

        self.assertEqual(len(path_gx), 3)
        self.assertTrue(
            path_gx[0],
            Segment2D(get_point_by_id(points, 0), get_point_by_id(points, 3)),
        )
        self.assertTrue(
            path_gx[1],
            Segment2D(get_point_by_id(points, 3), get_point_by_id(points, 2)),
        )
        self.assertTrue(
            path_gx[2],
            Segment2D(get_point_by_id(points, 2), get_point_by_id(points, 1)),
        )


class TestInvalidEdge(unittest.TestCase):
    def test_invalid_edge(self):
        points = points_example()
        points_2 = points_example_2()

        del_edges, del_triangles = delaunay_edges_triangles(points)
        del_edges_2, del_triangles_2 = delaunay_edges_triangles(points_2)

        current_segment = Segment2D(
            get_point_by_id(points, 0), get_point_by_id(points, 1)
        )
        current_segment_2 = Segment2D(
            get_point_by_id(points_2, 0), get_point_by_id(points_2, 3)
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx,
            _,
        ) = ge_xia_path_construction(del_triangles, current_segment)

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx_2,
            _,
        ) = ge_xia_path_construction(del_triangles_2, current_segment_2)

        self.assertEqual(
            get_invalid_edge(path_gx, current_segment),
            Segment2D(get_point_by_id(points, 3), get_point_by_id(points, 2)),
        )
        print(path_gx_2)
        self.assertIsNone(get_invalid_edge(path_gx_2, current_segment_2))


class TestCurrentSegment(unittest.TestCase):
    def test_current_segment(self):
        points = points_example()
        points_2 = points_example_2()

        del_edges, del_triangles = delaunay_edges_triangles(points)
        del_edges_2, del_triangles_2 = delaunay_edges_triangles(points_2)

        current_segment = get_current_segment(points, del_edges)
        current_segment_2 = get_current_segment(points_2, del_edges_2)

        self.assertEqual(
            current_segment,
            Segment2D(get_point_by_id(points, 0), get_point_by_id(points, 1)),
        )

        self.assertEqual(
            current_segment_2,
            Segment2D(get_point_by_id(points_2, 0), get_point_by_id(points_2, 3)),
        )


class TestHasShortWay(unittest.TestCase):
    def test_has_short_way(self):
        points = points_example()
        points_2 = points_example_2()
        points_3 = points_example_3()

        del_edges, del_triangles = delaunay_edges_triangles(points)
        del_edges_2, del_triangles_2 = delaunay_edges_triangles(points_2)
        del_edges_3, del_triangles_3 = delaunay_edges_triangles(points_3)

        current_segment = get_current_segment(points, del_edges)
        current_segment_2 = get_current_segment(points_2, del_edges_2)
        current_segment_3 = get_current_segment(points_3, del_edges_3)

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx,
            spanning_ratio_gx,
        ) = ge_xia_path_construction(del_triangles, current_segment)

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx_2,
            spanning_ratio_gx_2,
        ) = ge_xia_path_construction(del_triangles_2, current_segment_2)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            path_gx_3,
            spanning_ratio_gx_3,
        ) = ge_xia_path_construction(del_triangles_3, current_segment_3)

        self.assertFalse(
            has_short_way(
                path_gx, spanning_ratio_gx, del_edges, current_segment, points
            )[0]
        )

        self.assertTrue(
            has_short_way(
                path_gx_2, spanning_ratio_gx_2, del_edges_2, current_segment_2, points_2
            )[0]
        )

        self.assertTrue(
            has_short_way(
                path_gx_3, spanning_ratio_gx_3, del_edges_3, current_segment_3, points_3
            )[0]
        )


if __name__ == "__main__":
    unittest.main()
