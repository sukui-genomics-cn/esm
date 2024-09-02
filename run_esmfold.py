import esm

from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import py3Dmol

alphabet_list = list(ascii_uppercase + ascii_lowercase)


def to_cpu_and_numpy(x):
    if isinstance(x, torch.Tensor):
        # 检查张量是否在 CUDA 上
        if x.is_cuda:
            x = x.cpu()
        return x
    elif isinstance(x, dict):
        return {k: to_cpu_and_numpy(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_cpu_and_numpy(v) for v in x)
    else:
        return x


def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0, :, 1]

    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    sm_contacts = softmax(output["distogram_logits"], -1)[0]
    sm_contacts = sm_contacts[..., bins < 8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    o = {"pae": pae[mask, :][:, mask],
         "plddt": plddt[mask],
         "sm_contacts": sm_contacts[mask, :][:, mask],
         "xyz": xyz[mask]}
    return o


def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()


pymol_color_list = ["#33ff33", "#00ffff", "#ff33cc", "#ffff00", "#ff9999", "#e5e5e5", "#7f7fff", "#ff7f00",
                    "#7fff7f", "#199999", "#ff007f", "#ffdd5e", "#8c3f99", "#b2b2b2", "#007fff", "#c4b200",
                    "#8cb266", "#00bfbf", "#b27f7f", "#fcd1a5", "#ff7f7f", "#ffbfdd", "#7fffff", "#ffff7f",
                    "#00ff7f", "#337fcc", "#d8337f", "#bfff3f", "#ff7fff", "#d8d8ff", "#3fffbf", "#b78c4c",
                    "#339933", "#66b2b2", "#ba8c84", "#84bf00", "#b24c66", "#7f7f7f", "#3f3fa5", "#a5512b"]


def show_pdb(
        pdb_str,
        show_sidechains=False,
        show_mainchains=False,
        color="pLDDT",
        chains=None, vmin=50, vmax=90,
        size=(800, 480), hbondCutoff=4.0,
        Ls=None,
        animate=False):
    if chains is None:
        chains = 1 if Ls is None else len(Ls)
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])
    if animate:
        view.addModelsAsFrames(pdb_str, 'pdb', {'hbondCutoff': hbondCutoff})
    else:
        view.addModel(pdb_str, 'pdb', {'hbondCutoff': hbondCutoff})
    if color == "pLDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': vmin, 'max': vmax}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif color == "chain":
        for n, chain, color in zip(range(chains), alphabet_list, pymol_color_list):
            view.setStyle({'chain': chain}, {'cartoon': {'color': color}})
    if show_sidechains:
        BB = ['C', 'O', 'N']
        view.addStyle({'and': [{'resn': ["GLY", "PRO"], 'invert': True}, {'atom': BB, 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "GLY"}, {'atom': 'CA'}]},
                      {'sphere': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "PRO"}, {'atom': ['C', 'O'], 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
    if show_mainchains:
        BB = ['C', 'O', 'N', 'CA']
        view.addStyle({'atom': BB}, {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
    view.zoomTo()
    if animate: view.animate()
    return view


def predict(sequence="GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE", jobname="test"):
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/", ":").upper())
    sequence = re.sub(":+", ":", sequence)
    sequence = re.sub("^[:]+", "", sequence)
    sequence = re.sub("[:]+$", "", sequence)
    copies = 1  # @param {type:"integer"}
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 3  # @param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
    chain_linker = 25

    ID = jobname + "_" + get_hash(sequence)[:5]
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length", length)

    u_seqs = list(set(seqs))
    if len(seqs) == 1:
        mode = "mono"
    elif len(u_seqs) == 1:
        mode = "homo"
    else:
        mode = "hetero"

    model = esm.pretrained.esmfold_structure_module_only_8M()
    model = model.eval().cuda()
    model.eval().cuda().requires_grad_(False)

    # optimized for Tesla T4
    if length > 700:
        model.set_chunk_size(64)
    else:
        model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(
        sequence,
        num_recycles=num_recycles,
        chain_linker="X" * chain_linker,
        residue_index_offset=512
    )
    output = to_cpu_and_numpy(output)

    pdb_str = model.output_to_pdb(output)[0]
    ptm = output["ptm"][0]
    plddt = output["plddt"][0, ..., 1].mean()
    output_parse = parse_output(output)
    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
    os.system(f"mkdir -p {ID}")
    prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
    np.savetxt(f"{prefix}.pae.txt", output_parse["pae"], "%.3f")
    with open(f"{prefix}.pdb", "w") as out:
        out.write(pdb_str)
    return pdb_str, output_parse


def plot_ticks(Ls):
    Ln = sum(Ls)
    L_prev = 0
    for L_i in Ls[:-1]:
        L = L_prev + L_i
        L_prev += L_i
        plt.plot([0, Ln], [L, L], color="black")
        plt.plot([L, L], [0, Ln], color="black")
    ticks = np.cumsum([0] + Ls)
    ticks = (ticks[1:] + ticks[:-1]) / 2
    plt.yticks(ticks, alphabet_list[:len(ticks)])


def plot_confidence(O, Ls=None, dpi=100):
    if "lm_contacts" in O:
        plt.figure(figsize=(20, 4), dpi=dpi)
        plt.subplot(1, 4, 1)
    else:
        plt.figure(figsize=(15, 4), dpi=dpi)
        plt.subplot(1, 3, 1)

    plt.title('Predicted lDDT')
    plt.plot(O["plddt"])
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.xlim(0, O["plddt"].shape[0])
    plt.ylim(0, 100)
    plt.ylabel('plDDT')
    plt.xlabel('position')
    plt.subplot(1, 4 if "lm_contacts" in O else 3, 2)

    plt.title('Predicted Aligned Error')
    Ln = O["pae"].shape[0]
    plt.imshow(O["pae"], cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')

    if "lm_contacts" in O:
        plt.subplot(1, 4, 3)
        plt.title("contacts from LM")
        plt.imshow(O["lm_contacts"], cmap="Greys", vmin=0, vmax=1, extent=(0, Ln, Ln, 0))
        if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
        plt.subplot(1, 4, 4)
    else:
        plt.subplot(1, 3, 3)
    plt.title("contacts from Structure Module")
    plt.imshow(O["sm_contacts"], cmap="Greys", vmin=0, vmax=1, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    return plt


def main():
    color = "confidence"  # @param ["confidence", "rainbow", "chain"]
    if color == "confidence": color = "pLDDT"
    show_sidechains = False  # @param {type:"boolean"}
    show_mainchains = False  # @param {type:"boolean"}

    sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
    lengths = [len(s) for s in [sequence]]
    pdb_str, output_parse = predict(sequence, jobname="test")

    view = show_pdb(
        pdb_str, color=color,
        show_sidechains=show_sidechains,
        show_mainchains=show_mainchains,
        Ls=lengths).show()

    plot_confidence(output_parse, Ls=lengths, dpi=100)
    prefix = "esmfold_predict"
    # view.png(f'{prefix}_3d_mol.png')
    plt.savefig(f'{prefix}.png', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    main()
