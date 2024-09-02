def test_readme_esmfold():
    import torch
    import esm

    model = esm.pretrained.esmfold_structure_module_only_8M()
    model = model.eval().cuda()

    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    # Multimer prediction can be done with chains separated by ':'

    with torch.no_grad():
        output = model.infer_pdb(sequence)
    print(f"output:\n{output}")

    with open("result.pdb", "w") as f:
        f.write(output)

    # import biotite.structure.io as bsio
    # struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    # print(struct.b_factor.mean())  # this will be the pLDDT
    with open("result.pdb") as f:
        lines = [line for line in f.readlines() if line.startswith('ATOM')]
    bfactors = [float(line[60:66]) for line in lines]
    # assert torch.allclose(torch.Tensor(bfactors).mean(), torch.Tensor([88.3]), atol=1e-1)


if __name__ == '__main__':
    test_readme_esmfold()
