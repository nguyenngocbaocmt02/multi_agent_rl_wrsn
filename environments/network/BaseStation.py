import environments.Extension as Ex
from environments.network import Parameter


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

        self.location = location
        self.direct_nodes = []
        self.monitored_target = []

    def probe_neighbors(self):
        for node in self.net.listNodes:
            if Ex.euclideanDistance(self.location, node.location) <= Parameter.COM_RANGE:
                self.direct_nodes.append(node)

    def receive_package(self, package):
        return

    def operate(self, t=1):
        self.probe_neighbors()
        while True:
            yield self.env.timeout(t)

# def chargeMC(self, mc, t=0):
#    if distance.euclidean(mc.location, self.location) == 0:
#        mc.energy = mc.capacity
#    yield self.env.timeout(t)
