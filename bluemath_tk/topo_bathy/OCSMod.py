from ocsmesh.mesh import EuclideanMesh2D


class EuclideanMesh2D(EuclideanMesh2D):
    """
    A subclass of EuclideanMesh2D that adds some @property / @setter methods
    """

    @property
    def value(self):
        """Reference to the value property of the internal mesh object"""
        return self.msh_t.value

    @value.setter
    def value(self, value):
        self.msh_t.value = value

    @property
    def ndims(self):
        """Reference to the number of dimensions of the internal mesh object"""
        return self.msh_t.ndims

    @property
    def tria3(self):
        """Reference to TRI3 (triangular 3-node elements) of the internal mesh object"""
        return self.msh_t.tria3

    @tria3.setter
    def tria3(self, value):
        self.msh_t.tria3 = value

    @property
    def quad4(self):
        """Reference to QUAD4 (quadrilateral 4-node elements) of the internal mesh object"""
        return self.msh_t.quad4

    @quad4.setter
    def quad4(self, value):
        self.msh_t.quad4 = value

    @property
    def mshID(self):
        """Reference to the mesh ID of the internal mesh object"""
        return self.msh_t.mshID

    @property
    def hexa8(self):
        """Reference to HEXA8 (hexahedral 8-node elements) of the internal mesh object"""
        return self.msh_t.hexa8

    @hexa8.setter
    def hexa8(self, value):
        self.msh_t.hexa8 = value

    @property
    def vert2(self):
        """Reference to underlying mesh 2D vertices structure"""
        return self.msh_t.vert2

    @vert2.setter
    def vert2(self, value):
        self.msh_t.vert2 = value
