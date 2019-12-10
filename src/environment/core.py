import numpy as np

ENTITY_AGENT = 3
ENTITY_WL = 2
ENTITY_BOTH = 10


class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        self.id = None
        self.pos = None

class Workload(Entity):
    def __init__(self):
        super(Workload, self).__init__()

class Service(Entity):
    def __init__(self):
        super(Service, self).__init__()

