import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt

def gexf_to_arrays(path, feature_key="Feature"):
    G = nx.read_gexf(path, node_type=int)   # node_type=int keeps your 0..n-1 ids
    nodes = sorted(G.nodes())               # fixed ordering
    adj = nx.to_numpy_array(G, nodelist=nodes)   # weights preserved if present
    feats = np.array(
        [[G.nodes[n].get(feature_key, 0.0)] for n in nodes],
        dtype=np.float32
    )                                       # shape (n_nodes, 1)
    return adj, feats



ATOM_MAPS = {
    "MUTAG": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
    "Mutagenicity": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br", 7: "S"},
}
BOND_MAPS = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC
}

def nx_to_mol(G: nx.Graph, dataset_name: str, node_label_attr="label", edge_label_attr="label"):
    atom_map = ATOM_MAPS.get(dataset_name)
    if atom_map is None:
        return None  # atom types unknown for this dataset
    mol = Chem.RWMol()
    for node, data in G.nodes(data=True):
        symbol = atom_map.get(data.get(node_label_attr, 0), "C")
        mol.AddAtom(Chem.Atom(symbol))
    for u, v, data in G.edges(data=True):
        if u == v: continue  # skip self-loops
        bond_type = BOND_MAPS.get(data.get(edge_label_attr, 1), Chem.BondType.SINGLE)
        mol.AddBond(int(u), int(v), bond_type)
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return mol  # valid molecule
    except Exception:
        return None  # invalid molecule

def is_valid_molecule(G: nx.Graph, dataset_name: str, **kwargs) -> bool:
    return nx_to_mol(G, dataset_name, **kwargs) is not None

def get_cf_validity(instance, cf_adj: np.ndarray, dataset_name: str, atom_types) -> bool:
    """
    Check if a CF adjacency matrix corresponds to a valid molecule.
    Requires instance.atom_types to be set (from TUDataset populate()).
    Returns False if atom types are unavailable for this dataset.
    """
    if atom_types is None:
        return False
    G = nx.from_numpy_array(cf_adj)
    for i, atom_idx in enumerate(atom_types):
        G.nodes[i]['label'] = int(atom_idx)
    return is_valid_molecule(G, dataset_name)

def plot_molecule(mol, title="", filepath=None, ref_mol=None):
    # Generate 2D coords
    import random
    if ref_mol is not None:
        try:
            AllChem.GenerateDepictionMatching2DStructure(mol, ref_mol)
        except ValueError:
            AllChem.Compute2DCoords(mol)  # fallback to independent layout
    else:
        AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(300, 300))

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)
    if filepath:
        plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"Saved to {filepath}")
    plt.show()
    plt.close()
                