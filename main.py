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
    :param planar_cutoff: cutoff to decide on planarity
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
        if v > planar_cutoff:
            raise NotPlanarException
    return np.nanmean(norms, axis=0) / np.linalg.norm(np.nanmean(norms, axis=0))


def get_angle(v1, v2):
    """
    Calculates the angle in degrees between two vectors.

    :param v1 (np.array): vector #1
    :param v2 (np.array): vector #2
    :return (float): angle between v1 and v2 in degrees
    """
    return np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.nannorm(v2))))


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
        self.output = ""
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

    def write(self, string):
        if self.output == "":
            print("Output is not set for this structure.")
        else:
            with open(self.output, "a") as f:
                f.write(string+"\n")

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
        self.output = filename.split('.')[0] + "_pi-pi-py.dat"
        with open(self.output, "w") as f:
            f.write("{0:d} atoms were read from {1}\n".format(self.count, filename))

    def get_connectivity(self, cutoff=1.8):
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
        self.write("With the connectivity criterion of {:f}, {:d} separate molecules were identified."
                   .format(cutoff, len(self.molecules)))

    def get_trivalent(self, atoms=['C', 'N']):
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
                    if self.connect[i, j]:
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
            a1 = get_angle(v1, v2)
            a2 = get_angle(v1, v3)
            a3 = get_angle(v3, v2)
            if (360 - a1 - a2 - a3) < 2:
                self.planar.append(i)
                self.planar_norm[i] = get_plane_norm([self.coord[i], self.coord[self.neighbours[i][0]], self.coord[self.neighbours[i][1]], self.coord[self.neighbours[i][2]]])

    def process_mols(self, planar_cutoff = 0.05):
        """
        Create Molecule object for each identified molecule.
        
        Rings in molecules are automatically evaluated to find aromatic/planar ones. 
        :return: 
        """
        if len(self.molecules) == 0:
            self.get_connectivity()
        for mol in self.molecules:
            self.Mols.append(Molecule(parent=self, atoms=mol))
            self.Mols[-1].get_aromatic_rings(planar_cutoff)
            self.Mols[-1].get_ar_ring_angles()

    def get_intermolecular_pi_pi(self, cutoff = 4.5):
        if len(self.Mols) < 2:
            print("There are not separate molecules to get intermolecular interaction.")
            print("Please consider changing the cutoffs.")
            return
        self.write("Ring 1                     Ring 2                      d-cent d-plane1 d-plane2 shift1  shift2 angle")
        for i in range(len(self.Mols)):
            for j in range(i+1, len(self.Mols)):
                for k in range(len(self.Mols[i].rings)):
                    for l in range(len(self.Mols[j].rings)):
                        if self.Mols[i].ar_rings[k] and self.Mols[j].ar_rings[l]:
                            d = self.Mols[i].centres[k] - self.Mols[j].centres[l]
                            if np.linalg.norm(d) < cutoff:
                                x1 = np.dot(d, self.Mols[i].ar_ring_norms[k])
                                if x1 < 0:
                                    x1 = np.dot(d, -self.Mols[i].ar_ring_norms[k])
                                y1 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x1) ** 2)
                                x2 = np.dot(d, self.Mols[j].ar_ring_norms[l])
                                if x2 < 0:
                                    x2 = np.dot(d, -self.Mols[j].ar_ring_norms[l])
                                y2 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x2) ** 2)
                                op = "["
                                for a in self.Mols[i].rings[k]:
                                    op += "{:4d}".format(a + 1)
                                op += "] ["
                                for a in self.Mols[j].rings[l]:
                                    op += "{:4d}".format(a + 1)
                                op += "]"
                                self.write("{0} {1:8.4f}{2:8.4f}{3:8.4f}{4:8.4f}{5:8.4f}{6:8.2f}"
                                            .format(op, np.linalg.norm(d), x1, x2, y1, y2,
                                            get_angle(self.Mols[i].ar_ring_norms[k], self.Mols[j].ar_ring_norms[l])))


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
        self.output = parent.output
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
        # if parent.trivalent != None:
        #     self.trivalent = atoms.intersection(set(parent.trivalent))
        #     self.neighbours = {}
        #     for key, value in parent.neighbours.items():
        #         if key in self.ind:
        #             self.neighbours[key] = value
        # if parent.planar != None:
        #     self.planar = atoms.intersection(set(parent.planar))
        #     self.planar_norm = {}
        #     for key, value in parent.planar_norm.items():
        #         if key in self.ind:
        #             self.planar_norm[key] = value
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

    def get_ar_ring_angles(self):
        """
        Check if there are neighbouring aromatic rings and if so, calculate the angle between their planes
        :return: void
        """
        if len(self.ar_rings) == 0:
            self.get_aromatic_rings()
        self.write("Ring 1                     Ring 2                    Atom1 Atom2 angle")
        for e in self.graph.edges():
            r1 = -1
            r2 = -1
            for i in range(len(self.rings)):
                if self.ar_rings[i]:
                    if e[0] in self.rings[i] and e[1] in self.rings[i]:
                        break
                    elif e[0] in self.rings[i]:
                        r1 = i
                    elif e[1] in self.rings[i]:
                        r2 = i
            if r1 != -1 and r2 != -1:
                op = "["
                for a in self.rings[r1]:
                    op += "{:4d}".format(a + 1)
                op += "] ["
                for a in self.rings[r2]:
                    op += "{:4d}".format(a + 1)
                op += "]"
                self.write("{0} {1:4d} {2:4d} {3:6.2f}".format(op, e[0]+1, e[1]+1,
                                           get_angle(self.ar_ring_norms[r1], self.ar_ring_norms[r2])))



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
    args.bond_cutoff = float(args.bond_cutoff)
    args.intermolecular_cutoff = float(args.intermolecular_cutoff)
    args.planar_cutoff = float(args.planar_cutoff)

    geom = Structure()
    geom.read_xyz(args.xyz)
    geom.get_connectivity(args.bond_cutoff)
    geom.process_mols(args.planar_cutoff)

    # output to be handled
    geom.get_intermolecular_pi_pi(args.intermolecular_cutoff)




if __name__ == "__main__": main()