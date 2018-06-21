import numpy as np
import argparse
import os
import sys
import networkx as nx

class NotPlanarException(Exception):
    """
    Exception risen if ring not deemed planar.
    """
    pass

def get_plane_norm(points, planar_cutoff = 0.05):
    """
    Get normal vector of plane defined by 3 or more points.
    
    Method: normal vectors pointing towards each point from the first one defined, cross product
    is taken as individual norm from each possible pairs, unless they are close to antiparallel.
    If each component of the std dev of these norms is smaller then a cutoff, the plane is deemed
    planar and the mean normalized vector is returned.
    
    :param points (list): points defining the plane
    :param planar_cutoff: cutoff to decide on plarity
    :return (np.array): normal vector of plane
    :rises NotPlanarException
    """
    if len(points) < 3:
        print("Give at least 3 points to define a plane.")
        return None
    vectors = np.empty(shape=(len(points) - 1, len(points[0])), dtype=np.float_)
    for i in range(len(vectors)):
        vectors[i] = (points[i + 1] - points[0]) / np.linalg.norm(points[i + 1] - points[0])
    norms = np.empty(shape=(int(len(vectors) * (len(vectors) - 1) / 2), len(points[0])), dtype=np.float_)
    k = 0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if np.abs(np.dot(vectors[i], vectors[j])+1) < 0.01:
                norms[k] = None
            else:
                norms[k] = np.cross(vectors[i], vectors[j])
            if k != 0:
                if np.dot(norms[k], norms[0]) < 0:
                    norms[k] = -1 * norms[k]
            k += 1
    for v in np.nanstd(norms, axis=0):
        if v > [planar_cutoff]:
            raise NotPlanarException
    return np.nanmean(norms, axis=0) / np.linalg.norm(np.nanmean(norms, axis=0))


class Structure:
    """
        Object handling the data a structure loaded from an xyz file.

        Connectivity is based on proximity set by a threshold.

        Attributes:
            atoms (list): atom names
            coord (dict): atom coordinates of each index
            count (int): number of atoms
            connect (np.array): symmetric (adjacency) matrix of connectivity
            graph (nx.Graph): connectivity graph based on the connectivity cutoff
            molecules (list): separate molecules defined by atom indices
            Mols (list): Molecule objects consisting the structure
        """
    def __init__(self):
        """
        Object initiation.
        """
        self.atoms = []
        self.coord = {}
        self.count = 0
        self.connect = None
        self.graph = None
        self.molecules = []
        self.Mols = []
        self.trivalent = None
        self.neighbours = None
        self.planar = None
        self.planar_norm = None

    def read_xyz(self, filename):
        """
        Read structure from xyz file.
        
        :param filename (str): path to xyz file 
        :return: 
        """
        if not os.path.isfile(filename):
            print("Cannot open file {0}".format(filename))
            sys.exit(0)
        with open(filename, "r") as f:
            try:
                self.count = int(f.readline().strip())
            except:
                print("Invalid xyz file, cannot read number of atoms.")
            f.readline()
            for i in range(self.count):
                l = f.readline().split()
                self.atoms.append(l[0])
                self.coord[i] = np.array(l[1:4], dtype=np.float_)
        print("{0:d} atoms were read from {1}".format(self.count, filename))

    def get_connectivity(self, cutoff = 1.8):
        """
        Create connectivity/adjacency matrix and bond network graph.
        
        Bonding is distinguished from distance by a give threshold.
        Molecules are recognized as separated subgraphs of connectivity. 
               
        :param cutoff: bonding cutoff
        :return: 
        """
        if self.count == 0:
            print("There is no atom read to be processed.")
            return
        self.connect = np.zeros(shape=(self.count, self.count), dtype=np.bool)
        for i in range(self.count):
            for j in range(i+1, self.count):
                if np.linalg.norm(self.coord[i]-self.coord[j]) < cutoff:
                    self.connect[i, j] = True
                    self.connect[j, i] = True

        self.graph = nx.from_numpy_matrix(self.connect)
        for c in nx.connected_components(self.graph):
            self.molecules.append(c)
        print("With the connectivity criterion of {:f}, {:d} separate molecules were identified.".format(cutoff, len(self.molecules)))

    def get_trivalent(self, atoms = ['C', 'N']):
        """
        Deprecated
        :param atoms: 
        :return: 
        """
        if self.connect == None:
            self.get_connectivity()
        self.trivalent = []
        self.neighbours = {}
        for i in range(self.count):
            if self.atoms[i] in atoms and np.count_nonzero(self.connect[i]) == 3:
                self.trivalent.append(i)
                self.neighbours[i] = []
                for j in range(self.count):
                    if self.connect[i,j] == True:
                        self.neighbours[i].append(j)

    def get_planar(self, cutoff = 2):
        """
        Deprecated
        :param cutoff: 
        :return: 
        """
        if self.trivalent == None:
            self.get_trivalent()
        self.planar = []
        self.planar_norm = {}
        for i in self.trivalent:
            v1 = self.coord[self.neighbours[i][0]] - self.coord[i]
            v2 = self.coord[self.neighbours[i][1]] - self.coord[i]
            v3 = self.coord[self.neighbours[i][2]] - self.coord[i]
            a1 = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            a2 = np.degrees(np.arccos(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))))
            a3 = np.degrees(np.arccos(np.dot(v3, v2) / (np.linalg.norm(v3) * np.linalg.norm(v2))))
            if (360 - a1 - a2 - a3) < 2:
                self.planar.append(i)
                self.planar_norm[i] = get_plane_norm([self.coord[i], self.coord[self.neighbours[i][0]], self.coord[self.neighbours[i][1]], self.coord[self.neighbours[i][2]]])

    def process_mols(self, planar_cutoff = 0.05):
        """
        Create Molecule object for each identified molecule.
        
        Rings in molecules are automatically evaluated to find aromatic/planar ones. 
        :return: 
        """
        if self.connect == None:
            self.get_connectivity()
        for mol in self.molecules:
            self.Mols.append(Molecule(parent=self, atoms=mol))
            self.Mols[-1].get_aromatic_rings(planar_cutoff)

    def get_intermolecular_pi_pi(self, cutoff = 4.5):
        if len(self.Mols) < 2:
            print("There are not separate molecules to get intermolecular interaction. Please consider changing the cutoffs.")
            return
        for i in range(len(self.Mols)):
            for j in range(i+1, len(self.Mols)):
                for k in range(len(self.Mols[i].rings)):
                    for l in range(len(self.Mols[j].rings)):
                        if self.Mols[i].ar_rings[k] and self.Mols[j].ar_rings[l]:
                            d = self.Mols[i].centres[k] - self.Mols[j].centres[l]
                            if np.linalg.norm(d) < cutoff:
                                print(self.Mols[i].rings[k], self.Mols[j].rings[l])
                                print(np.linalg.norm(d))
                                x1 = np.dot(d, self.Mols[i].ar_ring_norms[k])
                                if x1 < 0:
                                    x1 = np.dot(d, -self.Mols[i].ar_ring_norms[k])
                                y1 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x1) ** 2)
                                x2 = np.dot(d, self.Mols[j].ar_ring_norms[l])
                                if x2 < 0:
                                    x2 = np.dot(d, -self.Mols[j].ar_ring_norms[l])
                                y2 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x2) ** 2)
                                print(x1, y1, x2, y2)
                                # print(self.Mols[i].centres[k], self.Mols[j].centres[l])
                                # print(self.Mols[i].ar_ring_norms[k], self.Mols[j].ar_ring_norms[l])
                                # print(np.linalg.norm(self.Mols[i].ar_ring_norms[k]), np.linalg.norm(self.Mols[j].ar_ring_norms[l]))


class Molecule(Structure):
    """
    Object handling the data for one molecule.
    
    Aromaticity is only based on the planarity of rings.
    
    Attributes:
        ind (list): atom indices from the structure
        atoms (dict): atom names of each index
        coord (dict): atom coordinates of each index
        count (int): number of atoms
        graph (nx.Graph): connectivity graph based on the parent structure's criteria
        rings (list): simple rings (cycles) in the molecule
        centres (list): coordinates of the centre of each rings
        ar_rings (list): indicator of aromaticity of rings
        ar_ring_norms (dict): normal vector of each aromatic ring's plane 
    """

    def __init__(self, parent, atoms):
        """
        Object initialization.
        
        :param parent (Structure): system containing this molecule
        :param atoms (set): indices of atoms in the molecule
        """
        self.ind = list(atoms)
        self.atoms = {}
        for i in range(len(parent.atoms)):
            if i in self.ind:
                self.atoms[i] = parent.atoms[i]
        self.coord = {}
        for key, value in parent.coord.items():
            if key in self.ind:
                self.coord[key] = value
        self.count = len(self.atoms)
        self.graph = parent.graph.subgraph(self.ind)
        if parent.trivalent != None:
            self.trivalent = atoms.intersection(set(parent.trivalent))
            self.neighbours = {}
            for key, value in parent.neighbours.items():
                if key in self.ind:
                    self.neighbours[key] = value
        if parent.planar != None:
            self.planar = atoms.intersection(set(parent.planar))
            self.planar_norm = {}
            for key, value in parent.planar_norm.items():
                if key in self.ind:
                    self.planar_norm[key] = value
        self.rings = nx.cycle_basis(self.graph)
        self.centres = np.zeros(shape=(len(self.rings), 3), dtype=np.float_)
        self.ar_rings = []
        self.ar_ring_norms = {}

    def get_aromatic_rings(self, planar_cutoff = 0.05):
        """
        Choose planar rings as aromatic moieties.
        
        Check get_plane_norm() for details.
        
        :param planar_cutoff (float): threshold distinguishing planar rings
        :return: void
        """
        for i in range(len(self.rings)):
            for a in self.rings[i]:
                self.centres[i] += self.coord[a]
            self.centres[i] = self.centres[i] / len(self.rings[i])
            points = [self.centres[i]]
            for a in self.rings[i]:
                points.append(self.coord[a])
            try:
                self.ar_ring_norms[i] = (get_plane_norm(points, planar_cutoff))
                self.ar_rings.append(True)
            except NotPlanarException:
                self.ar_rings.append(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz", help="input xyz file")
    parser.add_argument("-c", "--bond-cutoff", default="1.8",
                        help="Maximum distance bonds. Closer atoms will be considered bounded.")
    parser.add_argument("-i", "--intermolecular-cutoff", default="4.5",
                        help="Cutoff of intermolecular interaction examined between rings.")
    parser.add_argument("-p", "--planar-cutoff", default="0.05",
                        help="Cutoff for finding planar rings.")
    # flag for atoms which can be in a ring
    args = parser.parse_args()
    args.c = float(args.c)
    args.i = float(args.i)
    args.p = float(args.p)

    geom = Structure()
    geom.read_xyz(args.xyz)
    geom.get_connectivity(args.c)
    geom.process_mols(args.p)

    # output to be handled
    geom.get_intermolecular_pi_pi(args.i)

    print("end")




if __name__ == "__main__": main()