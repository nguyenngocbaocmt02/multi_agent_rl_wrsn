from scipy.spatial.distance import euclidean
import numpy as np

class BaseStation:
    def __init__(self, location):
        """
        The initialization for basestation
        :param location: the coordinate of a basestation
        """
        # controlling timeline
        self.env = None

        # include all components in our network
        self.net = None

        self.location = np.array(location)
        self.monitored_target = []
        self.direct_nodes = []

    def probe_neighbors(self):
        for node in self.net.listNodes:
            if euclidean(self.location, node.location) <= node.com_range:
                self.direct_nodes.append(node)

    def receive_package(self, package):
        return

    def operate(self, t=1):
        self.probe_neighbors()
        while True:
            yield self.env.timeout(t)