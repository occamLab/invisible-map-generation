Contained within this directory is the code that enabled expectation maximization, which was used to tune the optimization weights. It is archived because some expectation maximization code is dependent on the "dummy nodes" (nodes used to incur penalties for changing odometry nodes gravity axes), which have since been removed without the necessary changes having been made to accommodate their absence. Additionally, the expectation maximization functionality had not been in active use for the research for some time.

The files immediately within this directory are standalone. The methods that were removed from the `Graph` class are included below.

```python
def expectation_maximization_once(self) -> Dict[str, np.ndarray]:
    """Run one cycle of expectation maximization.

    It generates an unoptimized graph from current vertex estimates and edge measurements and importances, and
    optimizes the graph. Using the errors, it tunes the weights so that the variances maximize the likelihood of
    each error by type.
    """
    self.generate_unoptimized_graph()
    self.optimize_graph()
    self.update_vertices_estimates()
    self.generate_maximization_params()
    return self.tune_weights().to_dict()

def expectation_maximization(self, maxiter=10, tol=1) -> int:
    """Run many iterations of expectation maximization.

    Kwargs:
        maxiter (int): The maximum amount of iterations.
        tol (float): The maximum magnitude of the change in weight vectors that will signal the end of the cycle.

    Returns:
        Number of iterations ran
    """
    previous_weights = self._weights.to_dict()
    i = 0
    while i < maxiter:
        self.expectation_maximization_once()
        new_weights = self._weights.to_dict()
        for weight_type in new_weights:
            if np.linalg.norm(new_weights[weight_type] - previous_weights[weight_type]) < tol:
                return i
        previous_weights = new_weights
        i += 1
    return i

def generate_maximization_params(self) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the arrays to be processed by the maximization model.

    Sets the error field to an array of errors, as well as a 2-d array populated by 1-hot 18 element observation
    vectors indicating the type of transform_vector. The meaning of the position of the one in the observation
    vector corresponds to the layout of the weights vector.

    Returns:
        Errors and observations
    """
    errors = np.array([])
    observations = np.reshape([], [0, 18])
    optimized_edges = {edge.id(): edge for edge in list(
        self.optimized_graph.edges())}

    for uid in self.edges:
        edge = self.edges[uid]
        start_mode = self.vertices[edge.startuid].mode
        end_mode = self.vertices[edge.enduid].mode

        if end_mode != VertexType.WAYPOINT:
            errors = np.hstack(
                [errors, self._basis_matrices[uid].T.dot(
                    optimized_edges[uid].error())])

        if start_mode == VertexType.ODOMETRY:
            if end_mode == VertexType.ODOMETRY:
                observations = np.vstack([observations, np.eye(6, 18)])
            elif end_mode == VertexType.TAG:
                observations = np.vstack([observations, np.eye(6, 18, 6)])
            elif end_mode == VertexType.DUMMY:
                observations = np.vstack([observations, np.eye(6, 18, 12)])
            elif end_mode == VertexType.WAYPOINT:
                continue
            else:
                raise Exception("Unspecified handling for edge of start type {} and end type {}".format(start_mode,
                                                                                                        end_mode))
        else:
            raise Exception("Unspecified handling for edge of start type {} and end type {}".format(start_mode,
                                                                                                    end_mode))

    self.errors = errors
    self.observations = observations
    return errors, observations

def tune_weights(self):
    """Tune the weights to maximize the likelihood of the errors found between the unoptimized and optimized graphs.
    """
    results = maxweights(self.observations, self.errors, self._weights)
    self.maximization_success = results.success
    self._weights = map_processing.weights.Weights.legacy_from_array(results.x)
    self.maximization_results = results
    self.update_edge_information()
    return self._weights
```
