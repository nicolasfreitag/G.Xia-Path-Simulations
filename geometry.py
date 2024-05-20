import math
import numpy as np


class Point2D:
    def __init__(self, x, y, id=None):
        self.x = float(x)
        self.y = float(y)
        self.id = id if id is not None else None

    def __repr__(self):
        return f"Point2D({self.x}, {self.y}, {self.id})"

    def distance_to(self, other_point):
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def is_almost(self, other_point):
        return round(self.x, 10) == round(other_point.x, 10) and round(
            self.y, 10
        ) == round(other_point.y, 10)

    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, Point2D):
            raise ValueError("Comparison with non-Point2D object")
        return (self.id) < (other.id)

    def __hash__(self):
        return hash((self.x, self.y, self.id))


class Segment2D:
    def __init__(self, point1, point2, id=None):
        if not isinstance(point1, Point2D) or not isinstance(point2, Point2D):
            raise ValueError("Both arguments must be instances of Point2D")

        self.id = int(id) if id is not None else None
        if point1.x > point2.x:
            self.point1, self.point2 = point2, point1
        else:
            self.point1, self.point2 = point1, point2

    def length(self):
        return math.sqrt(
            (self.point1.x - self.point2.x) ** 2 + (self.point1.y - self.point2.y) ** 2
        )

    def points(self):
        return [self.point1, self.point2]

    def intersects(self, other_segment):
        # Calculate determinants
        det = (self.point2.x - self.point1.x) * (
            other_segment.point1.y - other_segment.point2.y
        ) - (self.point2.y - self.point1.y) * (
            other_segment.point1.x - other_segment.point2.x
        )

        # Check if the lines are parallel
        if det == 0:
            return None  # Lines are parallel, no intersection

        # Calculate parameters s and t
        det_s = (other_segment.point1.x - self.point1.x) * (
            other_segment.point1.y - other_segment.point2.y
        ) - (other_segment.point1.y - self.point1.y) * (
            other_segment.point1.x - other_segment.point2.x
        )
        det_t = (self.point2.x - self.point1.x) * (
            other_segment.point1.y - self.point1.y
        ) - (self.point2.y - self.point1.y) * (other_segment.point1.x - self.point1.x)

        # Calculate s and t
        s = det_s / det
        t = det_t / det

        # Check if the intersection point is within the line segments
        if 0 <= s <= 1 and 0 <= t <= 1:
            # Calculate intersection point
            x_intersection = self.point1.x + s * (self.point2.x - self.point1.x)
            y_intersection = self.point1.y + s * (self.point2.y - self.point1.y)
            intersection_point = Point2D(x_intersection, y_intersection)

            # print("INTERSECTION_POINT: ", intersection_point)
            return not intersection_point.is_almost(
                other_segment.point1
            ) and not intersection_point.is_almost(other_segment.point2)

        else:
            return False

    #
    # Check if three points are in counterclockwise order
    # See: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    #
    # def ccw(self, point1, point2, point3):
    #     return (point3.y - point1.y) * (point2.x - point1.x) > (point2.y - point1.y) * (
    #         point3.x - point1.x
    #     )

    # def intersects(self, other_segment):
    #     line1 = shapely.LineString(
    #         [(self.point1.x, self.point1.y), (self.point2.x, self.point2.y)]
    #     )
    #     line2 = shapely.LineString(
    #         [
    #             (other_segment.point1.x, other_segment.point1.y),
    #             (other_segment.point2.x, other_segment.point2.y),
    #         ]
    #     )
    #     print("SEGMENT INTERSECTION: ", line1.intersection(line2))
    #     return line1.intersection(line2) != None

    # def intersects(self, other_segment):
    #     if self.__eq__(other_segment):
    #         return True

    #     if self.ccw(
    #         self.point1, other_segment.point1, other_segment.point2
    #     ) != self.ccw(
    #         self.point2, other_segment.point1, other_segment.point2
    #     ) and self.ccw(
    #         self.point1, self.point2, other_segment.point1
    #     ) != self.ccw(
    #         self.point1, self.point2, other_segment.point2
    #     ):
    #         return self.intersection_is_non_terminal_point(other_segment)

    # def intersection_is_non_terminal_point(self, other_segment):
    #     x1, y1 = self.point1.x, self.point1.y
    #     x2, y2 = self.point2.x, self.point2.y
    #     x3, y3 = other_segment.point1.x, other_segment.point1.y
    #     x4, y4 = other_segment.point2.x, other_segment.point2.y

    #     # Calculate the intersection point using cross products
    #     det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    #     if det == 0:
    #         # Segments are parallel or coincident
    #         return True

    #     intersection_x = (
    #         (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    #     ) / det
    #     intersection_y = (
    #         (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    #     ) / det
    #     intersection_point = Point2D(intersection_x, intersection_y, -1)

    #     # Check if intersection point is terminal point
    #     if (
    #         round(intersection_point.x, 7) == round(x3, 7)
    #         and round(intersection_point.y, 7) == round(y3, 7)
    #         or round(intersection_point.x, 7) == round(x4, 7)
    #         and round(intersection_point.y, 7) == round(y4, 7)
    #     ):
    #         return False
    #     # print(
    #     #     "INTERSECTION_POINT: ",
    #     #     round(intersection_point.x, 7),
    #     #     round(intersection_point.y, 7),
    #     #     "WITH TERMINAL POINTS",
    #     #     round(x3, 7),
    #     #     round(y3, 7),
    #     #     round(x4, 7),
    #     #     round(y4, 7),
    #     # )
    #     return True

    def __repr__(self):
        return f"Segment2D({self.point1.id}, {self.point2.id}, {self.length()})"

    def __eq__(self, other):
        if not isinstance(other, Segment2D):
            return False
        return (self.point1 == other.point1 and self.point2 == other.point2) or (
            self.point1 == other.point2 and self.point2 == other.point1
        )

    def __hash__(self):
        sorted_points = tuple(sorted([self.point1, self.point2]))
        return hash((sorted_points, self.id))


class Triangle2D:
    def __init__(self, point1, point2, point3, id):
        if (
            not isinstance(point1, Point2D)
            or not isinstance(point2, Point2D)
            or not isinstance(point3, Point2D)
        ):
            raise ValueError("All arguments must be instances of Point2D")

        self.triangle_id = int(id)
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

    def get_segments(self):
        return [
            Segment2D(self.point1, self.point2),
            Segment2D(self.point2, self.point3),
            Segment2D(self.point1, self.point3),
        ]

    def get_points(self):
        return [self.point1, self.point2, self.point3]

    def intersects(self, segment):
        segment1 = Segment2D(self.point1, self.point2, id=0)
        segment2 = Segment2D(self.point1, self.point3, id=1)
        segment3 = Segment2D(self.point2, self.point3, id=2)
        # print(self)
        return (
            segment1.intersects(segment)
            or segment2.intersects(segment)
            or segment3.intersects(segment)
        )

    def outer_circle(self):
        # Calculate the center of the circumscribed circle
        a = self.point2.x - self.point1.x
        b = self.point2.y - self.point1.y
        c = self.point3.x - self.point1.x
        d = self.point3.y - self.point1.y

        e = a * (self.point1.x + self.point2.x) + b * (self.point1.y + self.point2.y)
        f = c * (self.point1.x + self.point3.x) + d * (self.point1.y + self.point3.y)

        g = 2 * (
            a * (self.point3.y - self.point2.y) - b * (self.point3.x - self.point2.x)
        )

        if g == 0:
            # Points are collinear, no circumscribed circle
            return None

        center_x = (d * e - b * f) / g
        center_y = (a * f - c * e) / g

        # Calculate the radius of the circumscribed circle
        radius = math.sqrt(
            (center_x - self.point1.x) ** 2 + (center_y - self.point1.y) ** 2
        )

        return Circle2D(
            Point2D(
                center_x,
                center_y,
                "M" + str(self.point1.id) + str(self.point2.id) + str(self.point3.id),
            ),
            radius,
            self,
        )

    def __repr__(self):
        return f"Triangle2D({self.point1.id}, {self.point2.id}, {self.point3.id})"


class Circle2D:
    def __init__(self, center, radius, triangle=None):
        if not isinstance(center, Point2D):
            raise ValueError("Center must be an instance of Point2D")

        self.center = center
        self.radius = float(radius)

        if triangle is not None and not isinstance(triangle, Triangle2D):
            raise ValueError("Triangle must be an instance of Triangle2D")

        self.triangle = triangle

    # def intersection(self, other_circle):
    #     d = math.sqrt(
    #         (other_circle.center.x - self.center.x) ** 2
    #         + (other_circle.center.y - self.center.y) ** 2
    #     )

    #     if d > self.radius + other_circle.radius or d < abs(
    #         self.radius - other_circle.radius
    #     ):
    #         return None

    #     a = (self.radius**2 - other_circle.radius**2 + d**2) / (2 * d)
    #     h = math.sqrt(self.radius**2 - a**2)
    #     x = self.center.x + a * (other_circle.center.x - self.center.x) / d
    #     y = self.center.y + a * (other_circle.center.y - self.center.y) / d
    #     r = h / d

    #     if d == self.radius + other_circle.radius or d == abs(
    #         self.radius - other_circle.radius
    #     ):
    #         return Point2D(x, y)

    #     return [
    #         Point2D(
    #             x - r * (other_circle.center.y - self.center.y) / d,
    #             y + r * (other_circle.center.x - self.center.x) / d,
    #         ),
    #         Point2D(
    #             x + r * (other_circle.center.y - self.center.y) / d,
    #             y - r * (other_circle.center.x - self.center.x) / d,
    #         ),
    #     ]

    def contains_point(self, point):
        if not isinstance(point, Point2D):
            raise ValueError("Point must be an instance of Point2D")

        distance = math.sqrt(
            (point.x - self.center.x) ** 2 + (point.y - self.center.y) ** 2
        )
        return distance <= self.radius

    # def intersects_segment(self, segment):
    #     # Check if any endpoint of the segment is inside the circle
    #     if self.contains_point(segment.point1) or self.contains_point(segment.point2):
    #         return True

    #     # Check if the segment intersects with the circle
    #     closest_point = self.closest_point(segment)
    #     distance = math.sqrt(
    #         (closest_point.x - self.center.x) ** 2
    #         + (closest_point.y - self.center.y) ** 2
    #     )
    #     return distance <= self.radius

    # def closest_point(self, segment):
    #     # Find the closest point on the segment to the circle's center
    #     seg_vector = (
    #         segment.point2.x - segment.point1.x,
    #         segment.point2.y - segment.point1.y,
    #     )
    #     center_vector = (
    #         self.center.x - segment.point1.x,
    #         self.center.y - segment.point1.y,
    #     )

    #     seg_length = math.sqrt(seg_vector[0] ** 2 + seg_vector[1] ** 2)

    #     # Project center_vector onto seg_vector
    #     projection = (
    #         center_vector[0] * seg_vector[0] + center_vector[1] * seg_vector[1]
    #     ) / seg_length**2

    #     # Clamp the projection to the segment
    #     t = max(0, min(1, projection))

    #     closest_point = Point2D(
    #         segment.point1.x + t * seg_vector[0], segment.point1.y + t * seg_vector[1]
    #     )

    #     return closest_point

    def __repr__(self):
        return f"Circle2D({self.center}, radius={self.radius})"


class Arc2D:
    def __init__(self, start_point, middle_point, end_point, id=None):
        if (
            not isinstance(start_point, Point2D)
            or not isinstance(middle_point, Point2D)
            or not isinstance(end_point, Point2D)
        ):
            raise ValueError("All arguments must be instances of Point2D")

        self.id = int(id) if id is not None else None
        self.start_point = start_point
        self.middle_point = middle_point
        self.end_point = end_point

    def length(self):
        angle_rad = self.angle()
        return (
            angle_rad * self.radius()
            if angle_rad >= 0
            else (2 * np.pi + angle_rad) * self.radius()
        )

    def points(self):
        return [self.start_point, self.end_point]

    def angle(self):
        # Calculate the angle given three points using the dot product and cross product
        vector1 = (
            self.start_point.x - self.middle_point.x,
            self.start_point.y - self.middle_point.y,
        )
        vector2 = (
            self.end_point.x - self.middle_point.x,
            self.end_point.y - self.middle_point.y,
        )

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        angle_rad = math.atan2(cross_product, dot_product)
        return angle_rad

    def theta1(self):
        vector1 = (1, 0)
        vector2 = (
            self.start_point.x - self.middle_point.x,
            self.start_point.y - self.middle_point.y,
        )

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        angle_rad = math.atan2(cross_product, dot_product)
        return angle_rad

    def theta2(self):
        vector1 = (1, 0)
        vector2 = (
            self.end_point.x - self.middle_point.x,
            self.end_point.y - self.middle_point.y,
        )

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        angle_rad = math.atan2(cross_product, dot_product)
        return angle_rad

    def radius(self):
        return math.sqrt(
            (self.start_point.x - self.middle_point.x) ** 2
            + (self.start_point.y - self.middle_point.y) ** 2
        )

    def __repr__(self):
        return f"Arc2D({self.start_point.id}, {self.middle_point.id}, {self.end_point.id}, {self.length()})"

    def __hash__(self):
        return hash((self.point1, self.point2, self.id))
