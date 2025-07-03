import numpy as np
import matplotlib.pyplot as plt

class Polygon:
    def __init__(self, vertices):
        """
        Initialize a polygon.
        
        :param vertices: (N,2) array-like list of vertices.
        """
        self.vertices = np.array(vertices)
    
    def sdf(self, points):
        """
        Compute the signed distance for a set of 2D points with respect to this polygon.
        This is a vectorized implementation of the algorithm.
        
        :param points: (M,2) array of sample points.
        :return: (M,) array of signed distances.
        """
        A = self.vertices
        B = np.roll(self.vertices, -1, axis=0)
        AB = B - A
        AB_norm_sq = np.sum(AB ** 2, axis=1)
        
        diff = points[:, None, :] - A[None, :, :]
        dot = np.sum(diff * AB[None, :, :], axis=2)
        t = np.clip(dot / (AB_norm_sq[None, :] + 1e-12), 0, 1)
        projection = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        distances = np.linalg.norm(points[:, None, :] - projection, axis=2)
        min_distance = np.min(distances, axis=1)
        
        # Inside test via angle-sum method.
        v1 = A[None, :, :] - points[:, None, :]
        v2 = B[None, :, :] - points[:, None, :]
        dot_angles = np.sum(v1 * v2, axis=2)
        cross_angles = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        angles = np.arctan2(cross_angles, dot_angles)
        angle_sum = np.sum(angles, axis=1)
        inside = np.abs(angle_sum) > 1
        
        # Negative distance inside.
        sdf = np.where(inside, -min_distance, min_distance)
        return sdf

    def sample_points(self, num_points=10000, bbox_extension=(0.5, 0.5)):
        """
        Sample points around the polygon. First, a set of uniformly random points
        is drawn from an extended bounding box. Optionally, points can be sampled near
        the polygon's boundary with some noise added.
        
        :param num_points: Total number of points.
        :param bbox_extension: How much to extend the polygon's bounding box.
        :return: (points, sdf) where points is an (M,2) array and sdf is (M,).
        """
        
        perturb_variances = [0.05, 0.02, 0.005]
        
        min_corner = self.vertices.min(axis=0) - np.array(bbox_extension)
        max_corner = self.vertices.max(axis=0) + np.array(bbox_extension)
        
        # Uniform sampling.
        num_uniform = num_points // 5
        uniform_points = np.random.uniform(min_corner, max_corner, size=(num_uniform, 2))
        uniform_sdf = self.sdf(uniform_points)
        
        # Perturbed sampling around each vertex.
        perturbed_points = []
        for var in perturb_variances:
            num_repeat = num_points // self.vertices.shape[0]
            noise = np.random.normal(scale=np.sqrt(var), size=(self.vertices.shape[0] * num_repeat, 2))
            pts = np.repeat(self.vertices, num_repeat, axis=0) + noise
            perturbed_points.append(pts)
        
        if perturbed_points:
            perturbed_points = np.vstack(perturbed_points)
            perturbed_sdf = self.sdf(perturbed_points)
        else:
            perturbed_points = np.empty((0,2))
            perturbed_sdf = np.empty((0,))
        
        all_points = np.vstack((uniform_points, perturbed_points))
        all_sdf = np.hstack((uniform_sdf, perturbed_sdf))
        return all_points, all_sdf

class MultiplePolygons:
    def __init__(self, polygons):
        """
        Initialize with a list of Polygon objects or an iterable of vertex arrays.
        
        :param polygons: list of Polygon objects or list of vertices.
        """
        # If the elements are not Polygon instances, try to convert them.
        self.polygons = [p if isinstance(p, Polygon) else Polygon(p) for p in polygons]

    def sdf(self, points):
        """
        Compute the signed distance for a set of points as the minimum over all polygons.
        For each point, the overall SDF is the minimum of the SDFs computed for each polygon.
        
        :param points: (M,2) array of sample points.
        :return: (M,) array of signed distances.
        """
        # Compute SDF for each polygon.
        sdfs = np.array([poly.sdf(points) for poly in self.polygons])
        # Return the minimum SDF at each point.
        return np.min(sdfs, axis=0)

    def sample_points(self, num_points=10000, bbox_extension=(0.5, 0.5)):
        """
        Sample points around the union of all polygons. We first compute a bounding box 
        that encloses all polygons, then sample uniformly and add perturbed points near each polygon.
        
        :param num_points: Total number of points.
        :param bbox_extension: How much to extend the overall bounding box.
        :return: (points, sdf) where points is an (M,2) array and sdf is (M,).
        """
        # Get all vertices from all polygons.
        all_vertices = np.vstack([poly.vertices for poly in self.polygons])
        min_corner = all_vertices.min(axis=0) - np.array(bbox_extension)
        max_corner = all_vertices.max(axis=0) + np.array(bbox_extension)
        
        num_uniform = num_points // 5
        uniform_points = np.random.uniform(min_corner, max_corner, size=(num_uniform, 2))
        uniform_sdf = self.sdf(uniform_points)
        
        # Perturbed sampling around each polygon's vertices.
        perturbed_points = []
        for poly in self.polygons:
            for var in [0.05, 0.02, 0.005]:
                num_repeat = num_points // poly.vertices.shape[0]
                noise = np.random.normal(scale=np.sqrt(var), size=(poly.vertices.shape[0] * num_repeat, 2))
                pts = np.repeat(poly.vertices, num_repeat, axis=0) + noise
                perturbed_points.append(pts)
        
        if perturbed_points:
            perturbed_points = np.vstack(perturbed_points)
            perturbed_sdf = self.sdf(perturbed_points)
        else:
            perturbed_points = np.empty((0, 2))
            perturbed_sdf = np.empty((0,))
        
        all_points = np.vstack((uniform_points, perturbed_points))
        all_sdf = np.hstack((uniform_sdf, perturbed_sdf))
        return all_points, all_sdf