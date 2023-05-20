import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import keyboard

SIZE = 729


def edge(v0, v1, p):
    a = (p[0] - v0[0], p[1] - v0[1])
    b = (v1[0] - v0[0], v1[1] - v0[1])
    return a[1] * b[0] - a[0] * b[1]


def crop(v1, v2):
    v1 = (v1[0] - v2[0], v1[1] - v2[1])
    v = [0, 0]
    v[0] = 0 if v1[0] < 0 else v1[0]
    v[1] = 0 if v1[1] < 0 else v1[1]
    return v


def div_segment_in_ratio(v0, v1, ratio):
    x = (v0[0] + ratio * v1[0]) / (1 + ratio)
    y = (v0[1] + ratio * v1[1]) / (1 + ratio)
    return x, y


def get_2_points_in_ratio(v0, v1, ratio):
    p0 = div_segment_in_ratio(v0, v1, ratio / (1 + ratio))
    p1 = div_segment_in_ratio(v0, v1, (1 + ratio) / ratio)
    return p0, p1


def get_intersection_of_2_segment(v0, v1, v2, v3):
    denominator = (v0[0] - v1[0]) * (v2[1] - v3[1]) - \
                  (v0[1] - v1[1]) * (v2[0] - v3[0])
    x = (v0[0] * v1[1] - v0[1] * v1[0]) * (v2[0] - v3[0]) - \
        (v0[0] - v1[0]) * (v2[0] * v3[1] - v2[1] * v3[0])
    y = (v0[0] * v1[1] - v0[1] * v1[0]) * (v2[1] - v3[1]) - \
        (v0[1] - v1[1]) * (v2[0] * v3[1] - v2[1] * v3[0])
    x /= denominator
    y /= denominator
    return x, y


class SierpinskiCemetery:
    view_port: tuple[tuple, tuple]
    init_polygon: Polygon
    steps: int
    ratio: float
    image: np.array

    def __init__(self, polygon, steps, ratio):
        self.view_port = ((0, 0), (1, 1))
        self.init_polygon = polygon
        self.steps = steps
        self.ratio = ratio
        self.image = np.ones((SIZE, SIZE, 3))
        self.make_fractal()

    def make_fractal(self):
        start_polygon = self.init_polygon
        x, y = start_polygon.exterior.xy
        for i in range(len(x)):
            x[i] *= SIZE
            y[i] *= SIZE
        poly = np.vstack((x, y)).T
        start_polygon = Polygon(poly)
        self.fill_polygon(start_polygon, color='black')
        self.generate_fractal(start_polygon, self.steps)

    def fill_polygon(self, polygon, color):
        self.fill_triangle(
            (polygon.exterior.xy[0][0], polygon.exterior.xy[1][0]),
            (polygon.exterior.xy[0][1], polygon.exterior.xy[1][1]),
            (polygon.exterior.xy[0][2], polygon.exterior.xy[1][2]),
            color
        )
        self.fill_triangle(
            (polygon.exterior.xy[0][0], polygon.exterior.xy[1][0]),
            (polygon.exterior.xy[0][2], polygon.exterior.xy[1][2]),
            (polygon.exterior.xy[0][3], polygon.exterior.xy[1][3]),
            color
        )

    def scale_triangle(self, v0, v1, v2):
        temp = (self.view_port[0][0] * SIZE * 1.0,
                self.view_port[0][1] * SIZE * 1.0)
        v0 = crop(v0,  temp)
        v1 = crop(v1,  temp)
        v2 = crop(v2,  temp)
        diff = (self.view_port[1][0] - self.view_port[0][0],
                self.view_port[1][1] - self.view_port[0][1])
        scale_coefficient_x = 1 / diff[0] if diff[0] > 0 else 0
        scale_coefficient_y = 1 / diff[1] if diff[1] > 0 else 0
        v0[0] *= scale_coefficient_x
        v0[1] *= scale_coefficient_y
        v1[0] *= scale_coefficient_x
        v1[1] *= scale_coefficient_y
        v2[0] *= scale_coefficient_x
        v2[1] *= scale_coefficient_y
        return v0, v1, v2

    def fill_triangle(self, v0, v1, v2, color):
        v0, v1, v2 = self.scale_triangle(v0, v1, v2)
        min_x = round(min(min(v0[0], v1[0]), v2[0]))
        min_y = round(min(min(v0[1], v1[1]), v2[1]))
        max_x = round(max(max(v0[0], v1[0]), v2[0]))
        max_y = round(max(max(v0[1], v1[1]), v2[1]))
        for x in range(min_x, min(max_x + 1, SIZE)):
            for y in range(min_y, min(max_y + 1, SIZE)):
                p = np.array([x, y])
                e10 = edge(v1, v0, p)
                e21 = edge(v2, v1, p)
                e02 = edge(v0, v2, p)
                if e10 >= 0 and e21 >= 0 and e02 >= 0:
                    import matplotlib
                    self.image[x, y] = matplotlib.colors.to_rgb(color)

    def generate_fractal(self, polygon, current_step):
        if current_step <= 0:
            return
        ab_points = get_2_points_in_ratio(
            (polygon.exterior.xy[0][0], polygon.exterior.xy[1][0]),
            (polygon.exterior.xy[0][1], polygon.exterior.xy[1][1]),
            self.ratio
        )
        bc_points = get_2_points_in_ratio(
            (polygon.exterior.xy[0][1], polygon.exterior.xy[1][1]),
            (polygon.exterior.xy[0][2], polygon.exterior.xy[1][2]),
            self.ratio
        )
        cd_points = get_2_points_in_ratio(
            (polygon.exterior.xy[0][2], polygon.exterior.xy[1][2]),
            (polygon.exterior.xy[0][3], polygon.exterior.xy[1][3]),
            self.ratio
        )
        da_points = get_2_points_in_ratio(
            (polygon.exterior.xy[0][3], polygon.exterior.xy[1][3]),
            (polygon.exterior.xy[0][0], polygon.exterior.xy[1][0]),
            self.ratio
        )
        v0 = get_intersection_of_2_segment(
            bc_points[0],
            da_points[1],
            ab_points[1],
            cd_points[0]
        )
        v1 = get_intersection_of_2_segment(
            bc_points[1],
            da_points[0],
            ab_points[1],
            cd_points[0]
        )
        v3 = get_intersection_of_2_segment(
            bc_points[0],
            da_points[1],
            ab_points[0],
            cd_points[1]
        )
        v2 = get_intersection_of_2_segment(
            bc_points[1],
            da_points[0],
            ab_points[0],
            cd_points[1]
        )
        self.fill_polygon(
            Polygon(
                np.array([ab_points[0], ab_points[1], v0, v3])
            ),
            color='gray'
        )
        self.fill_polygon(
            Polygon(
                np.array([bc_points[0], bc_points[1], da_points[0], da_points[1]])
            ),
            color='gray'
        )
        self.fill_polygon(
            Polygon(
                np.array([v2, v1, cd_points[0], cd_points[1]])
            ),
            color='gray'
        )
        self.generate_fractal(
            Polygon(np.array([
                ab_points[1],
                (polygon.exterior.xy[0][1], polygon.exterior.xy[1][1]),
                bc_points[0],
                v0
            ])),
            current_step - 1
        )
        self.generate_fractal(
            Polygon(np.array([
                bc_points[1],
                (polygon.exterior.xy[0][2], polygon.exterior.xy[1][2]),
                cd_points[0],
                v1
            ])),
            current_step - 1
        )
        self.generate_fractal(
            Polygon(np.array([
                da_points[0],
                v2,
                cd_points[1],
                (polygon.exterior.xy[0][3], polygon.exterior.xy[1][3])
            ])),
            current_step - 1
        )
        self.generate_fractal(
            Polygon(np.array([
                (polygon.exterior.xy[0][0], polygon.exterior.xy[1][0]),
                ab_points[0],
                v3,
                da_points[1]
            ])),
            current_step - 1
        )

    def render(self):
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.image)
        plt.show()


steps = 5
ratio = 1
poly_array = [[0.1, 0.1], [0.25, 0.75], [0.85, 0.65], [0.95, 0.05]]
poly = Polygon(poly_array)
sc = SierpinskiCemetery(poly, steps, ratio)
sc.render()
