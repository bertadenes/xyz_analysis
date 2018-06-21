import numpy as np
import argparse
import os
import sys
import networkx as nx

class NotPlanarException(Exception):
    pass

def get_plane_norm(points, planar_cutoff = 0.05):
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
    return np.nanmean(norms, axis=0)


class Structure:

    def __init__(self):
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
        print("With the connectivity criterium of {:f}, {:d} separate molecules were identified.".format(cutoff, len(self.molecules)))

    def get_trivalent(self, atoms = ['C', 'N']):
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

    def process_mols(self):
        for mol in self.molecules:
            self.Mols.append(Molecule(parent=self, atoms=mol))
            self.Mols[-1].get_aromatic_rings()

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
                                print(np.linalg.norm(self.Mols[i].centres[k] - self.Mols[j].centres[l]))
                                x1 = np.dot(d, self.Mols[i].ar_ring_norms[k])
                                if x1 < 0:
                                    x1 = np.dot(d, -self.Mols[i].ar_ring_norms[k])
                                y1 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x1) ** 2)
                                x2 = np.dot(d, self.Mols[j].ar_ring_norms[l])
                                if x2 < 0:
                                    x2 = np.dot(d, -self.Mols[j].ar_ring_norms[l])
                                y2 = np.sqrt(np.linalg.norm(d) ** 2 - np.linalg.norm(x2) ** 2)
                                print(x1, y1, x2, y2)
                                print(np.degrees(np.arccos(np.dot(self.Mols[i].ar_ring_norms[k], self.Mols[j].ar_ring_norms[l]))))




class Molecule(Structure):

    def __init__(self, parent, atoms):
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

    def get_aromatic_rings(self):
        for i in range(len(self.rings)):
            for a in self.rings[i]:
                self.centres[i] += self.coord[a]
            self.centres[i] = self.centres[i] / len(self.rings[i])
            points = [self.centres[i]]
            for a in self.rings[i]:
                points.append(self.coord[a])
            try:
                self.ar_ring_norms[i] = (get_plane_norm(points))
                self.ar_rings.append(True)
            except NotPlanarException:
                self.ar_rings.append(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz")
    # flag for atoms which can be in a ring
    # distance cutoff
    args = parser.parse_args()

    geom = Structure()
    geom.read_xyz(args.xyz)
    geom.get_connectivity()
    geom.process_mols()
    geom.get_intermolecular_pi_pi()

    print("end")




if __name__ == "__main__": main()