import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

if len(sys.argv) != 4:
    print ("usage: %s N input.smi output.sdf"% sys.argv[0])
    sys.exit(1)

n_confs = int(sys.argv[1])
input_smi = sys.argv[2]
output_sdf = sys.argv[3]

def RobustSmilesMolSupplier(filename):
    with open(filename) as f:
        for line in f:
            split = line.split()
            smile = split[0]
            name = split[1]
            mol = Chem.MolFromSmiles(smile)
            yield name, mol

reader = RobustSmilesMolSupplier(input_smi)
writer = Chem.SDWriter(output_sdf)

# nb. conformers to generate prior to energy minimization
# as an empirical function of the molecule's flexibility
def how_many_conformers(mol):
    nb_rot_bonds = AllChem.CalcNumRotatableBonds(mol)
    if nb_rot_bonds <= 7:
        return 50
    elif nb_rot_bonds <= 12:
        return 200
    else:
        return 300

# to prune too similar conformers
rmsd_threshold = 0.35 # Angstrom

# keep only conformers which are far enough from the reference conformer
# (the one of lowest energy)
def rmsd_filter(mol, ref_conf, l):
    # print "before: %d" % (len(l))
    res = []
    refConfId = ref_conf.GetId()
    for e, curr_conf in l:
        currConfId = curr_conf.GetId()
        rms = AllChem.GetBestRMS(mol, mol, refConfId, currConfId)
        # print "e: %f rms: %f" % (e, rms)
        if rms > rmsd_threshold:
            res.append((e, curr_conf))
    # print "after: %d" % (len(res))
    return res

for name, mol in reader:
    if mol:
        n = how_many_conformers(mol)
        print ("init pool size for %s: %d" % (name, n))
        mol_H = Chem.AddHs(mol)
        print ("generating starting conformers ...")
        confIds = AllChem.EmbedMultipleConfs(mol_H, n)
        conf_energies = []
        # FF minimization
        print ("FF minimization ...")
        for cid in confIds:
            ff = AllChem.UFFGetMoleculeForceField(mol_H, confId=cid)
            # print "E before: %f" % ff.CalcEnergy()
            ff.Minimize()
            energy = ff.CalcEnergy()
            # print "E after: %f" % energy
            conformer = mol_H.GetConformer(cid)
            # print "cid: %d e: %f" % (cid, energy)
            conf_energies.append((energy, conformer))
        # sort by increasing E
        conf_energies = sorted(conf_energies, key=lambda x: x[0])
        # output non neighbor conformers
        nb_out = 0
        kept = []
        print ("RMSD pruning ...")
        while nb_out < n_confs and len(conf_energies) > 0:
            nb_out += 1
            (e, conf) = conf_energies.pop(0)
            kept.append((e, conf))
            # remove neighbors
            conf_energies = rmsd_filter(mol_H, conf, conf_energies)
        # write them out
        print ("kept %d confs for %s" % (len(kept), name))
        res = Chem.Mol(mol_H)
        res.RemoveAllConformers()
        for e, conf in kept:
            cid = res.AddConformer(conf, assignId=True)
            name_cid = "%s_%04d" % (name, cid)
            res.SetProp("_Name", name_cid)
            # print "cid: %d" % cid
            writer.write(res, confId=cid)
writer.close()