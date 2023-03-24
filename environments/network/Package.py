from environments.network import Parameter


class Package:
    def __init__(self, target_id, size):
        self.package_size = size
        self.target_id = target_id
