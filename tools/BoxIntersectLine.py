class Point:
    def __init__(self, point_x, point_y, point_z):
        self.coord = [point_x, point_y, point_z]



class LineSegment:
    def __init__(self, point_start, point_end):
        origin = []
        direction = []
        for index in range(3):
            origin.append(point_start.coord[index])
            direction.append(point_end.coord[index] - point_start.coord[index])

        self.origin = origin
        self.direction = direction


    def get_point(self, coefficient):
        point_coord = []
        for index in range(3):
            point_coord.append(self.origin[index] + coefficient * self.direction[index])
        return Point(point_coord[0], point_coord[1], point_coord[2])



class Box:
    def __init__(self, point_a, point_b):
        self.pA = point_a
        self.pB = point_b


    def get_intersect_point(self, line_segment):

        for index, direction in enumerate(line_segment.direction):
            if direction == 0:
                box_max = max(self.pA.coord[index], self.pB.coord[index])
                box_min = min(self.pA.coord[index], self.pB.coord[index])
                if line_segment.origin[index] > box_max or line_segment.origin[index] < box_min:
                    return None, None


        t0, t1 = 0., 1.
        for index in range(3):
            if line_segment.direction[index] != 0.:
                inv_dir = 1. / line_segment.direction[index]
                t_near = (self.pA.coord[index] - line_segment.origin[index]) * inv_dir
                t_far = (self.pB.coord[index] - line_segment.origin[index]) * inv_dir
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t_near, t0)
                t1 = min(t_far, t1)
                if t0 > t1:
                    return None, None
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return intersection_point_near, intersection_point_far


    def new_get_intersect_point(self, line_segment):

        for index, direction in enumerate(line_segment.direction):
            if direction == 0:
                box_max = max(self.pA.coord[index], self.pB.coord[index])
                box_min = min(self.pA.coord[index], self.pB.coord[index])
                if line_segment.origin[index] > box_max or line_segment.origin[index] < box_min:
                    return None, None


        t0, t1 = 0., 1.
        for index in range(2):
            if line_segment.direction[index] != 0.:
                inv_dir = 1. / line_segment.direction[index]
                t_near = (self.pA.coord[index] - line_segment.origin[index]) * inv_dir
                t_far = (self.pB.coord[index] - line_segment.origin[index]) * inv_dir
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t_near, t0)
                t1 = min(t_far, t1)
                if t0 > t1:
                    return None, None
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return intersection_point_near, intersection_point_far


    def get_intersect_length(self, line_segment):
        near_point, far_point = self.get_intersect_point(line_segment)
        if near_point is None:
            return 0.
        length = 0.
        for index in range(3):
            length += (far_point.coord[index] - near_point.coord[index]) ** 2
        return length ** 0.5


    def new_get_intersect_length(self, line_segment):
        near_point, far_point = self.new_get_intersect_point(line_segment)
        if near_point is None:
            return 0.
        length = 0.
        for index in range(3):
            length += (far_point.coord[index] - near_point.coord[index]) ** 2
        return length ** 0.5


if __name__ == '__main__':
    box = Box(Point(-1, -1, -1), Point(1, 1, 1))
    line = LineSegment(Point(-2, -2, 0), Point(2, 2, 0))
    print(box.get_intersect_length(line))