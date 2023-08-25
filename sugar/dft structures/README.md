# Computational structures

### Introduction

This directory contains DFT optimized structures used to investigate the mechanism of a photochemical glucose to allose isomerization.

### Benchmarking

Conformers of various structure types were used to benchmark DFAs against CCSD(T) energies. Structures were generated using B3LYP def2-tzvp(-f) cpcm(dmso). The structures are sorted in each folder by CCSD(T) energy. The title in each *xyz* file contains the CCSD(T) energy.

`glu-acetate` conformers of methlyglucoside coordinated to an acetate molecule

`glucose` conformers of alpha-methylglucoside

`haa` conformers for the TS for H atom abstraction by quinuclidine radical cation on methyl glucoside at c3

`haa-ac` conformers for the TS for H atom abstraction by quinuclidine radical cation on methyl glucoside with an acetate coordinated

`had` conformers for the TS for H atom donation by adamantane thiol to C3 radical

`radical` conformers of the radical formed from H atom abstraction of methylglucoside at C3 position

`radical-acetate` conformers of the radical formed from H atom abstraction of methylglucoside at C3 position coordinated to an acetate molecule

### Conformational search

Each flexible molecule/transition state was extensively conformationally sampled and optimized. This directory contains relaxed structures as indicated below. Structures were generated using wB97X-D4, ma-def2-tzvp(-f) cpcm(dmso). The structures are sorted in each folder by Gibbs free energy. The title in each *xyz* file contains the energy and gibbs free energy at this level of theory.

`all-ac-confs` conformers of methylalloside coordinated to an acetate molecule

`alpha-sugars-confs` confomers of each alpha-methylpyranoside

`beta-sugar` conformers of each beta-methylpyranoside

`Glc-Ac-confs` confomers of methylglucoside coordinated to an acetate molecule

`glucosde-radicals-confs` conformers of the radical formed at each site after H atom abstraction on methylglucoside. Separated by site and presence an acetate molecule (indicated by -Ac)

`TS-HAA-confs` conformers of the TS for H atom abstraction by quinuclidine radical cation at sites C1-C5 of methylglucoside and C3 of methylalloside.

`TS-HAA-Ac-confs` conformers of the TS for H atom abstraction by quinuclidine radical cation at sites C2-C4 of methylglucoside and C3 of methylalloside with an acetate coordinated

`TS-HAD-confs` conformers of teh TS for H atom donation by adamantane thiol to deliver either methylglucoside(glc) or methylalloside(all) from the corresponding C3 radical

### Final structures

The lowest energy structure from each conformational search was then further optimized using wB97X-D4, ma-def2-tzvp cpcm(acetone). DLPNO-CCSD(T1) aug-cc-pvtz cpcm(acetone) energies were calculated on each of these optimized structures. The structures are sorted in each folder by Gibbs free energy (DFT). The title in each *xyz* file contains the energy and gibbs free energy at the DFT level and the energy at the DLPNO-CCSD(T1) level.

`Glc Radicals` the radicals formed after H atom abstraction at each site of methylglucoside. *-ac* indicates the presence of a coordinated acetate molecule.

`Sugars` each of the eight methylpyranosides in their alpha and beta anomeric forms.

`TS` H atom transfer TS to/from methylglucoside indicated as either H atom abstraction by quinuclidine radical cation at the site indicated (haa) or H atom donation by adamantane thiol at C3 (had). *-ac* indicates the presence of a coordinating acetate molecule. *-all* indicates that the TS was to/from methylalloside.

`Misc` contains other structures used in the study that do not fit into one of the above categories:
* *ac.xyz* (acetate molecule)
* *all-ac.xyz* (methylalloside coordinated to an acetate molecule)
* *glc-ac.xyz* (methylglucoside coordinated to an acetate molecule)
* *haa-c3-4clbenzoate.xyz* (HAA TS at the C3 position of methylglucoside coordinated to a 4-chlorobenzoate molecule)
* *ipa-rad.xyz* (isopropanol radical used for isodesmic calculations of BDE)
* *ipa.xyz* (isopropanol used for isodesmic calculations of BDE)
* *quinH.xyz* (quinuclidinium)
* *quinradcat.xyz* (quinuclidinium radical cation)
* *thiol.xyz* (adamantane thiol)
* *thiyl.xyz* (adamantane thiyl radical)