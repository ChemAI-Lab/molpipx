import os
import pandas as pd
import argparse
import time
import timeit

import jax
import jax.numpy as jnp
import jax.random as jrnd

import pipjax
from pipjax import PIPlayer as PIP
from pipjax import EnergyPIP
from pipjax import detect_molecule, get_functions, get_energy_and_forces


def get_number_of_atoms(molecule_dict):
    na = 0
    for i, k in enumerate(molecule_dict):
        na += molecule_dict[k]
    return na


def compute_gradients(molecule_type: str, poly_degree: int, batch_size: int):
    mol_dict, mol_sym = detect_molecule(molecule_type)
    print(molecule_type, mol_dict, mol_sym)
    na = get_number_of_atoms(mol_dict)

    f_mono, f_poly = get_functions(molecule_type, poly_degree)
    x = jnp.ones(int((na**2 - na)/2))
    d_mono = f_mono(x).shape[-1]

    rng = jrnd.PRNGKey(0)
    _, key = jrnd.split(rng)

    # random geometries to test the energy and forces
    geoms = jrnd.normal(key, (batch_size, na, 3))
    geom0 = geoms[:1]

    # load PIP-Flax model
    model_pip = PIP(f_mono, f_poly)
    model_energy = EnergyPIP(f_mono, f_poly)
    params = model_energy.init(key, geom0)
    params_pip = model_pip.init(key, geom0)
    z_pip0 = model_pip.apply(params_pip, geom0)
    d_pip = z_pip0.shape[-1]

    # timeit after jitting
    def my_function():
        e, f = get_energy_and_forces(
            model_energy.apply, geoms, params)
        # e = e.block_until_ready()
        # f = f.block_until_ready()
        return {'e': e, 'f': f}

    # pre-jitting
    n = 1
    timing_result = timeit.timeit(
        lambda: jax.block_until_ready(jax.jit(my_function)()),
        number=n
    )
    warmup_time = timing_result / n
    print("Time warm up: ", timing_result)

    # timeit after jitting
    n = 1
    timing_result = timeit.timeit(
        lambda: jax.block_until_ready(jax.jit(my_function)()),
        number=n
    )
    exec_time = timing_result / n
    print("Time jit exec: ", timing_result)

    r = {'mol': molecule_type,
         'pip': d_pip,
         'mono': d_mono,
         'poly degree': poly_degree,
         'bs': batch_size,
         'time wup': warmup_time,
         'time val and grad': exec_time,
         }
    return r


def plotting_v1(bs=100):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    results = pd.read_csv(f"{path}/value_and_gradient_time_jit_tst.csv")

    cmap = plt.cm.YlGnBu
    colors = [cmap((i+2)/6) for i in range(6)]

    results = results.loc[results['bs'] == bs]

    mols_ = ['A3', 'A2B', 'ABC',
             'A4', 'A3B', 'A2B2', 'A2BC', 'ABCD',
             'A5', 'A4B', 'A3B2', 'A3BC', 'A2B2C', 'A2BCD', 'ABCDE']

    fig, ax = plt.subplots(1, 1)

    results = results[results['bs'] == bs]
    poly_ = np.arange(3, 8, dtype=np.int32)
    for i, pi in enumerate(poly_):
        r_pi = []
        r_ppi = []
        for mi in mols_:
            condition = (results['mol'] == mi) & (results['poly degree'] == pi)
            exists = condition.any()
            if exists:
                r_pi.append(results[condition]['time val and grad'].iloc[0])

            condition = (results['mol'] == mi) & (results['poly degree'] == pi)
            exists = condition.any()
            if exists:
                r_ppi.append(results[condition]['time wup'].iloc[0])
        r_pi = np.array(r_pi)
        r_ppi = np.array(r_ppi)
        print(pi, r_pi)
        ax.plot(np.arange(len(mols_)), r_pi, marker='o', markersize=8,
                label=f"{pi}", color=colors[i])
        ax.plot(np.arange(len(mols_)), r_ppi, marker=None, ls='--',
                color=colors[i])

    ax.set_yscale('log')
    plt.legend(loc="upper left")
    mols_tex = [r"A$_{3}$", r"A$_{2}$B", r"ABC",
                r"A$_{4}$", f"A$_{3}$B", r"A$_{2}$B$_{2}$", r"A$_{2}$BC", "ABCD",
                r"A$_{5}$", r"A$_{4}$B", r"A$_{3}$B$_{2}$", r"A$_{3}$BC", r"A$_{2}$B$_{2}$C", r"A$_{2}$BCD", "ABCDE"]

    plt.grid(visible=True)
    plt.text(0.25, 205, f"N = {bs}", fontsize=17)
    plt.xticks(np.arange(len(mols_)), mols_tex, fontsize=13, rotation=90)
    # plt.xticks(fontsize=13, rotation=90)
    plt.xlabel("", fontsize=1)
    plt.ylabel('Time [s]', fontsize=16)
    # g.despine(left=True)
    plt.tight_layout()
    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    plt.savefig(f"{path}/pip_value_and_grad_{str(bs)}_jit3.png", dpi=600)


def plotting(bs=100, jit_style='jitting'):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    results = pd.read_csv(f"{path}/value_and_gradient_time_jit_tst2.csv")

    cmap = plt.cm.YlGnBu
    colors = [cmap((i+1)/5) for i in range(5)]

    results = results.loc[results['bs'] == bs]

    mols_ = ['A3', 'A2B', 'ABC',
             'A4', 'A3B', 'A2B2', 'A2BC', 'ABCD',
             'A5', 'A4B', 'A3B2', 'A3BC', 'A2B2C', 'A2BCD', 'ABCDE']

    fig, ax = plt.subplots(1, 1)

    results = results[results['bs'] == bs]
    poly_ = np.arange(3, 8, dtype=np.int32)
    for i, pi in enumerate(poly_):
        r_pi = []
        r_ppi = []
        for mi in mols_:
            condition = (results['mol'] == mi) & (results['poly degree'] == pi)
            exists = condition.any()
            if exists:
                r_pi.append(results[condition]['time val and grad'].iloc[0])

            condition = (results['mol'] == mi) & (results['poly degree'] == pi)
            exists = condition.any()
            if exists:
                r_ppi.append(results[condition]['time wup'].iloc[0])
        r_pi = np.array(r_pi)
        r_ppi = np.array(r_ppi)

        if jit_style == 'jitting':
            plt.plot(np.arange(len(mols_)), r_ppi, marker='o', markersize=8,
                     label=f"{pi}", color=colors[i])
            plt.text(1.5, 6E3, f"a)", fontsize=15)
        elif jit_style == 'jitted':
            plt.plot(np.arange(len(mols_)), r_pi, marker='s', markersize=8,
                     label=f"{pi}", color=colors[i])
            plt.text(1.5, 3.7E-3, f"b)", fontsize=15)

    plt.legend(loc="upper left")
    mols_tex = [r"A$_{3}$", r"A$_{2}$B", r"ABC",
                r"A$_{4}$", f"A$_{3}$B", r"A$_{2}$B$_{2}$", r"A$_{2}$BC", "ABCD",
                r"A$_{5}$", r"A$_{4}$B", r"A$_{3}$B$_{2}$", r"A$_{3}$BC", r"A$_{2}$B$_{2}$C", r"A$_{2}$BCD", "ABCDE"]

    plt.grid(visible=True)
    plt.xticks(np.arange(len(mols_)), mols_tex, fontsize=12, rotation=90)
    # plt.xticks(fontsize=13, rotation=90)
    plt.xlabel("", fontsize=1)
    if jit_style == 'jitting':
        ax.set_yscale('log')
        plt.ylabel('Jitting Time [s]', fontsize=16)
    elif jit_style == 'jitted':
        # plt.ylim(0.0001, 0.0021)
        plt.yticks(fontsize=10)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel('Time [s]', fontsize=16)
    # g.despine(left=True)
    plt.tight_layout()
    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    if jit_style == 'jitting':
        plt.savefig(
            f"{path}/pip_value_and_grad_{str(bs)}_jitting2.png", dpi=600)
    elif jit_style == 'jitted':
        plt.savefig(
            f"{path}/pip_value_and_grad_{str(bs)}_jitted2.png", dpi=600)


def list_of_strings(arg):
    return arg.split(',')


def main():
    parser = argparse.ArgumentParser(description="pipjax gradient test")
    parser.add_argument("--bs", type=int, default=1,)
    parser.add_argument("--mol-list", type=list_of_strings)  # , required=True
    parser.add_argument("--poly-list", type=list_of_strings)  # , required=True
    # # parser.add_argument("--poly", type=int, required=False)

    args = parser.parse_args()
    bs = args.bs  # 1000

    mols_ = ['A3', 'A2B', 'ABC',
             'A4', 'A3B', 'A2B2', 'A2BC', 'ABCD',
             'A5', 'A4B', 'A3B2', 'A3BC', 'A2B2C', 'A2BCD', 'ABCDE']
    poly_ = [3, 4, 5, 6]  # , 7
    mols_ = args.mol_list
    poly_ = args.poly_list

    path = '/scratch/r/ravh011/ravh011/pipjax_test'
    file = 'value_and_gradient_time_jit_tst2.csv'

    def get_pd_frame():
        if os.path.isfile(f"{path}/{file}"):
            results = pd.read_csv(
                f"{path}/{file}")
        else:
            results = pd.DataFrame()
        return results

    # for mi in mols_:
    #     for p in poly_:
    #         # for p in range(8, 9):
    #         ri = compute_gradients(mi, p, bs)
    #         print(ri)

    #         results = get_pd_frame()
    #         results = pd.concat(
    #             [results, pd.DataFrame(ri, index=[0])], ignore_index=True)
    #         results.to_csv(
    #             f"{path}/{file}", index=False)

    # plot
    bs = bs  # 100
    jit_ = 'jitting'  # 'jitted'
    plotting(bs, jit_)
    jit_ = 'jitted'  # 'jitted'
    plotting(bs, jit_)

    # assert 0


if __name__ == "__main__":
    main()
