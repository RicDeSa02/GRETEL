from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer

import numpy as np, torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
BT = Chem.BondType

ATOMS = {
    "MUTAG": ["C", "N", "O", "F", "I", "Cl", "Br"],
    "AIDS": ["C", "O", "N", "Cl", "F", "S", "Se", "P", "Na", "I", "Co", "Br", "Li", "Si",
             "Mg", "Cu", "As", "B", "Pt", "Ru", "K", "Pd", "Au", "Te", "W", "Rh", "Zn",
             "Bi", "Pb", "Ge", "Sb", "Sn", "Ga", "Hg", "Ho", "Tl", "Ni", "Tb"],
}
BONDS = {
    "MUTAG": {0: BT.AROMATIC, 1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE},
    "AIDS": {0: BT.SINGLE, 1: BT.DOUBLE, 2: BT.TRIPLE},
}


class ChemicalValidityMetric(EvaluationMetric):
    """1.0 if the counterfactual is a chemically valid molecule, else 0.0."""

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Chemical_Validity'
        self.require_connected = True  # True -> fragmented molecules count as invalid
        self.num_structural_feat = None

    def evaluate(self, instance_1: GraphInstance, instance_2: GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """
        return float(self._is_valid(instance_2.data, instance_2, instance_1, dataset))


    def _is_valid(self, adj_matrix, atom_src, bond_src, dataset) -> bool:
        name = next((k for k in ("MUTAG", "AIDS", "BBBP") if k in dataset.name.split('-', 1)[0]), None)
        tpl = self.make_template(atom_src, bond_src, name, dataset) if name else None
        if tpl is None:
            return False
        symbols, charges, explicit_hs, bonds = tpl

        adj = np.asarray(adj_matrix) > 0.5
        if adj.shape[0] != len(symbols):
            return False

        mol = Chem.RWMol()
        for s, c, h in zip(symbols, charges, explicit_hs):
            if h < 0: return False # impossible H count -> invalid
            atom = Chem.Atom(s)
            atom.SetFormalCharge(int(c))
            atom.SetNumExplicitHs(int(h))
            mol.AddAtom(atom)

        for u, v in zip(*np.triu_indices(adj.shape[0], k=1)):
            u, v = int(u), int(v)
            if adj[u, v]:
                bt = bonds.get((u, v), BT.SINGLE)  # new edge -> single bond
                mol.AddBond(u, v, bt)
                if bt == BT.AROMATIC:
                    mol.GetBondBetweenAtoms(u, v).SetIsAromatic(True)
                    mol.GetAtomWithIdx(u).SetIsAromatic(True)
                    mol.GetAtomWithIdx(v).SetIsAromatic(True)
        try:
            Chem.SanitizeMol(mol)  # full check, including valence
        except Exception:
            return False
        require_connected = False if name == "BBBP" else True # True -> fragmented molecules count as invalid
        return not (require_connected and len(Chem.GetMolFrags(mol)) > 1)

    def make_template(self, atom_src, bond_src, name, dataset):
        """Atoms from atom_src (CF or original), bond types from bond_src (original)."""

        try:
            if name == "BBBP":
                m = dataset.node_features_map
                x = np.asarray(atom_src.node_features)
                pt = Chem.GetPeriodicTable()
                symbols = [pt.GetElementSymbol(int(round(z))) for z in x[:, m['AtomicNum']]]
                charges = [int(round(c)) for c in x[:, m['FormalCharge']]]
                hs = [int(round(h)) for h in x[:, m['NumExplicitHs']]]
                mol = bond_src.graph_features["mol"]
                bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondType()
                        for b in mol.GetBonds()}
                return symbols, charges, hs, bonds

            if name in ATOMS:
                x = np.asarray(atom_src.node_features)
                atom_label = np.argmax(x[:, dataset.data.num_node_attributes : x.shape[1] - self._structural_feat(dataset)], axis=1)
                symbols = [ATOMS[name][int(a)] for a in atom_label]
                charges = ([int(round(c)) for c in x[:, 1]] if name == "AIDS" else [0] * len(symbols))
                bonds = {k: BONDS[name][int(l)] for k, l in self.bond_labels_from(bond_src).items()}
                return symbols, charges, [0] * len(symbols), bonds
        except Exception:
            return None
        return None

    def bond_labels_from(self, ref):
        edges = torch.tensor(np.asarray(ref.data)).nonzero().t().tolist()
        labels = np.argmax(np.asarray(ref.edge_features), axis=1)
        assert len(edges[0]) == len(labels)
        return {(min(u, v), max(u, v)): int(l)
                for u, v, l in zip(edges[0], edges[1], labels)}

    def _structural_feat(self, dataset):
        if self.num_structural_feat is None:
            names = [type(m).__name__ for m in dataset.manipulators]
            self.num_structural_feat = (1 if 'Causality' in names else 0) + (7 if 'NodeCentrality' in names else 0)
        return self.num_structural_feat


