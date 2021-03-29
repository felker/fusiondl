try:
    from MDSplus import Connection
except ImportError:
    pass


class Machine(object):
    def __init__(self, name, server, fetch_data_fn, max_cores=8,
                 current_threshold=0):
        self.name = name
        self.server = server
        self.max_cores = max_cores
        self.fetch_data_fn = fetch_data_fn
        self.current_threshold = current_threshold

    def get_connection(self):
        return Connection(self.server)

    def __eq__(self, other):
        return self.name.__eq__(other.name)

    def __lt__(self, other):
        return self.name.__lt__(other.name)

    def __ne__(self, other):
        return self.name.__ne__(other.name)

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
