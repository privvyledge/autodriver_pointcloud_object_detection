import time

try:
    # API Reference link: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.cluster_dbscan
    import open3d as o3d
    import open3d.core as o3c
    OPEN3D_INSTALLED = True
except ImportError:
    print("OPEN3D is not installed")
    OPEN3D_INSTALLED = False

try:
    # API Reference link: https://hdbscan.readthedocs.io/en/latest/api.html
    import hdbscan
    HDBSCAN_INSTALLED = True
except ImportError:
    print("HDBSCAN is not installed")
    HDBSCAN_INSTALLED = False

try:
    # API Reference links:
    #   HDBScan - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#hdbscan
    #   DBScan - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    #   OPTICS - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
    from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
    SKLEARN_INSTALLED = True
except ImportError:
    print("HDBSCAN is not installed")
    SKLEARN_INSTALLED = False



class PointCloudClustering:
    def __init__(self, points=None, method='hdbscan', backend='sklearn',
                 eps=0.0, min_cluster_size=5, min_samples=5, max_cluster_size=0, n_jobs=None, approx_predict=False):
        """

        :param points:
        :param method: either dbscan, hdbscan, optics
        :param backend: either open3d, sklearn, hdbscan
        """
        self.points = points
        self.method = method
        self._backend_priority = ['open3d', 'sklearn', 'hdbscan']
        self.backend = self._validate_backend_installed(backend)

        self.eps = eps  # cluster_tolerance. For hdbscan, set to 0
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.max_cluster_size = max_cluster_size  # 0 for hdbscan, None for sklearn, -1 for Open3D (not implemented for now)
        self.n_jobs = n_jobs
        self.approx_predict = approx_predict

        self.clusterer, self.labels = self.initialize_clusterer(points=self.points)

    def _validate_backend_installed(self, backend):
        if backend not in self._backend_priority:
            raise ValueError(f"Invalid backend '{backend}'. Must be one of {self._backend_priority}")

        if backend.lower() == 'open3d':
            if not OPEN3D_INSTALLED:
                if SKLEARN_INSTALLED:
                    backend = "sklearn"
                elif HDBSCAN_INSTALLED:
                    backend = "hdbscan"

        elif backend.lower() == 'sklearn':
            if not SKLEARN_INSTALLED:
                if OPEN3D_INSTALLED:
                    backend = "open3d"
                elif HDBSCAN_INSTALLED:
                    backend = "hdbscan"
        else:
            # HDBSCAN backend
            if not HDBSCAN_INSTALLED:
                if OPEN3D_INSTALLED:
                    backend = "open3d"
                elif SKLEARN_INSTALLED:
                    backend = "sklearn"
                else:
                    raise ValueError("Neither Open3d, sklearn nor hdbscan installed.")
        return backend

    def initialize_clusterer(self, method='', backend=None, points=None):
        labels = None
        if not method:
            method = self.method
        if backend is None:
            backend = self.backend

        backend = self._validate_backend_installed(backend)

        if backend == 'open3d':
            if method.lower() != 'dbscan':
                raise ValueError('Open3D backend only supports DBSCAN.')
            clusterer = o3d.t.geometry.PointCloud()
            if points is not None:
                clusterer.point.positions = o3d.core.Tensor.from_numpy(points)
                labels = self._predict_open3d()
        elif backend == 'sklearn':
            if method.lower() == 'dbscan':
                clusterer = DBSCAN(
                    eps=self.eps,
                    min_samples=self.min_samples,
                    n_jobs=self.n_jobs
                )
            elif method.lower() == 'hdbscan':
                clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.eps,
                    max_cluster_size=self.max_cluster_size,
                    cluster_selection_method='eom',  # eom, leaf
                    n_jobs=self.n_jobs,
                )
            elif method.lower() == 'optics':
                clusterer = OPTICS(
                    min_samples=self.min_samples,
                    # max_eps=np.inf,  # np.inf
                    eps=self.eps,
                    min_cluster_size=self.min_cluster_size,
                    cluster_method='xi',  # xi, dbscan
                    n_jobs=self.n_jobs
                )
            else:
                raise NotImplementedError(f'Clustering method: {method} not implemented for backend: {backend}.')
        elif backend == 'hdbscan':
            if method.lower() not in ['dbscan', 'hdbscan']:
                raise ValueError('HDBSCAN backend only supports DBSCAN.')
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.eps,
                max_cluster_size=self.max_cluster_size,
                cluster_selection_method='eom',  # eom, leaf
                prediction_data=True,  # self.approx_predict
            )

        if backend in ['sklearn', 'hdbscan'] and points is not None:
            clusterer.fit(points)
            labels = clusterer.labels_
        return clusterer, labels

    def _predict_open3d(self, clusterer=None, points=None):
        if clusterer is None:
            clusterer = self.clusterer

        if points is not None:
            clusterer.point.positions = o3d.core.Tensor.from_numpy(points)
        # else set to self.points

        labels = clusterer.cluster_dbscan(
            eps=self.eps,
            min_points=self.min_cluster_size,
            print_progress=False
        )
        return labels

    def _predict_sklearn_and_hdbscan(self, clusterer=None, points=None, fit_predict=False, approx_predict=None):
        if approx_predict is None:
            approx_predict = self.approx_predict

        if clusterer is None:
            clusterer = self.clusterer

        if points is None:
            # or set points = self.points
            labels = clusterer.labels_
        else:
            if approx_predict and self.backend == 'hdbscan':
                labels = hdbscan.approximate_predict(clusterer, points)

            elif self.method == 'dbscan' and self.backend == 'hdbscan':
                labels = clusterer.dbscan_clustering(cut_distance=self.eps, min_cluster_size=self.min_cluster_size)

            else:
                if fit_predict:
                    labels = clusterer.fit_predict(points)
                else:
                    self.clusterer = clusterer.fit(points)
                    labels = self.clusterer.labels_
        return labels

    def predict(self, points=None, clusterer=None, fit_predict=False):
        if self.backend == 'open3d':
            labels = self._predict_open3d(clusterer=clusterer, points=points)
        else:
            labels = self._predict_sklearn_and_hdbscan(clusterer=clusterer, points=points, fit_predict=fit_predict)
        return labels


if __name__ == '__main__':
    # eps=0.0, min_cluster_size=5, min_samples=5, max_cluster_size=None, n_jobs=None, approx_predict=False
    clusterer_open3d = PointCloudClustering(method='dbscan', backend='open3d', eps=0.02, min_cluster_size=5)
    clusterer_sklearn_hdbscan = PointCloudClustering(method='hdbscan', backend='sklearn', max_cluster_size=None)
    clusterer_sklearn_dbscan = PointCloudClustering(method='dbscan', backend='sklearn', eps=0.5, min_cluster_size=5, max_cluster_size=None)
    clusterer_hdbscan_hdbscan = PointCloudClustering(method='hdbscan', backend='hdbscan', max_cluster_size=0)
    clusterer_hdbscan_dbscan = PointCloudClustering(method='dbscan', backend='hdbscan', max_cluster_size=0)

    # Test predictions
    import numpy as np
    points = np.random.random((1000, 3))
    labels_open3d = clusterer_open3d.predict(points)
    labels_sklearn_hdbscan = clusterer_sklearn_hdbscan.predict(points)
    labels_sklearn_dbscan = clusterer_sklearn_dbscan.predict(points)
    labels_hdbscan_hdbscan = clusterer_hdbscan_hdbscan.predict(points)
    labels_hdbscan_dbscan = clusterer_hdbscan_dbscan.predict(points)

    # Test invalid initialization
    clusterer_open3d = PointCloudClustering(method='hdbscan', backend='open3d')
    clusterer_sklearn_dbscan = PointCloudClustering(method='dfadf', backend='sklearn')
    clusterer_sklearn_dbscan = PointCloudClustering(method='dfadf', backend='dfadf')
    print("Done")