import os
from typing import Tuple, Union
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element, get_el_sp
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.units import FloatWithUnit
from pymatgen.ext.matproj import MPRester

import pandas as pd
import numpy as np
import sklearn as learn
import timeit
import pickle

from sklearn.feature_extraction import DictVectorizer

from lme.Alloy import Alloy


class Tome:
    def __init__(self, source="excel", **kwargs) -> None:
        if source == "excel" and "path" in kwargs and "sheet" in kwargs:
            self.list_of_entries = self.from_excel(
                kwargs["path"], kwargs["sheet"], verbose=kwargs.get("verbose", False)
            )
        elif source == "pickle" and "path" in kwargs:
            self.list_of_entries = self.from_pickle(kwargs.get("path"))
        else:
            self.list_of_entries = list()

    def prepare_for_gb_classifier(self, **kwargs):
        if len(self.list_of_entries) == 0:
            return None

        list_expanded_entries = [
            Tome.expand_tome_entry_dict(e) for e in self.list_of_entries
        ]

        vec = DictVectorizer()

        data = vec.fit_transform(list_expanded_entries).toarray()
        features = vec.get_feature_names_out()

        # hardcoded to where I know it is from my csv, find a better way to do this?
        sorted = data[np.argsort(data[:, 10])]
        

        # I know what it does in theory, but that line be dense. Thanks stackoverflow!
        sectioned = np.split(sorted, np.where(np.diff(sorted[:, 10]))[0] + 1)

        if "path" in kwargs:
            path = kwargs.get("path")
            for i in range(len(sectioned)):
                np.savetxt(
                    f"{os.path.splitext(path)[0]}_data_mech_{i}.csv",
                    sectioned[i],
                    delimiter=",",
                    header=",".join(str(e) for e in features),
                )
        print(sectioned[0][:,0])

        sect = {
            str(int(sectioned[i][1, 10])): {
                "feat": np.delete(sectioned[i], [0,10,14], 1),
                "mech": [int(x) for x in sectioned[i][:, 10]],
                "id" : [int(x) for x in sectioned[i][:, 0]],
            }
            for i in range(len(sectioned))
        }
        #print(sect["0"]["id"])

        features = np.delete(features, [0,10,14])
        
        return sect, features

    @staticmethod
    def from_excel(path, sheet, **kwargs) -> list:
        excel_tome = pd.read_excel(path, sheet)
        factory = TomeEntryFactory()
        list_of_entries = list()

        for row in excel_tome.iterrows():
            e = factory.FromExcelTomeLine(row[1])
            list_of_entries.append(e)
            if kwargs.get("verbose", False):
                print(e)

        with open(f"{os.path.splitext(path)[0]+'.txt'}", "wb") as f:
            pickle.dump(list_of_entries, f)

        return list_of_entries

    @staticmethod
    def from_pickle(path) -> list:
        with open(path, "rb") as f:
            list_of_entries = pickle.load(f)

        return list_of_entries

    @staticmethod
    def tome_to_training_array(tome_entry):
        pass

    @staticmethod
    def expand_tome_entry_dict(tome_entry) -> dict:
        d = dict()
        d["id"] = tome_entry.id
        d["t_mechanism"] = tome_entry.mechanism
        d["t_temperature"] = tome_entry.test_temperature
        d["t_s_homologous"] = tome_entry.solid_homologous
        d["t_intermetallic"] = tome_entry.energy_of_intermetallic
        d["t_phase_1"] = tome_entry.phase[0]
        d["t_phase_2"] = tome_entry.phase[1]
        #for el in tome_entry.liquid_metal.elements:
        #    d[f"l_{el.symbol}_at"] = tome_entry.liquid_metal.get_atomic_fraction(el)
        d["l_melting"] = tome_entry.tm_liquid
        d["l_boiling"] = tome_entry.bp_liquid
        d["l_radius"] = tome_entry.radius_liquid
        d["l_electronegativity"] = tome_entry.electronegativity_liquid
        #for el in tome_entry.base_metal.elements:
        #    d[f"s_{el.symbol}_at"] = tome_entry.base_metal.get_atomic_fraction(el)
        d["s_melting"] = tome_entry.tm_solid
        d["s_boiling"] = tome_entry.bp_solid
        d["s_radius"] = tome_entry.radius_solid
        d["s_electronegativity"] = tome_entry.electronegativity_solid

        return d


class TomeEntry:
    def __init__(self, d) -> None:
        self.id: int = d["id"]
        self.base_metal: Union[Element, Composition] = d["base_metal"]
        self.liquid_metal: Union[Element, Composition] = d["liquid_metal"]
        self.tm_liquid: FloatWithUnit = d["tm_liquid"]
        self.tm_solid: FloatWithUnit = d["tm_solid"]
        self.test_temperature: FloatWithUnit = d["test_temperature"]
        self.mechanism: int = d["mechanism"]
        self.phase: tuple[int, ...] = d["phase"]
        self.energy_of_intermetallic: FloatWithUnit = d["energy_of_intermetallic"]

    def __repr__(self) -> str:
        return f"  ID: {self.id} Mechanism: {self.mechanism}\n"

    def __str__(self) -> str:
        return (
            f"\nID: {self.id}\n"
            + f"Mechanism: {self.mechanism}\n"
            + f"Test Temperature: {self.test_temperature}\n"
            + f"Phase: {self.translate_bravais_code(self.phase[0])} {self.translate_bravais_code(self.phase[1]) if self.phase[1] != 0 else ''}\n"
            + f"E_f Most Likely Intermetallic: {self.energy_of_intermetallic}"
            + f"\n"
            + f" Solid: {self.base_metal.formula}\n"
            + f"    T_m: {self.tm_solid}\n"
            + f"    Tb: {self.bp_solid}\n"
            + f"     r: {self.radius_solid}\n"
            + f"     E_neg: {self.electronegativity_solid}\n"
            + f"Liquid: {self.liquid_metal.formula}\n"
            + f"    T_m: {self.tm_liquid}\n"
            + f"    Tb: {self.bp_liquid}\n"
            + f"     r: {self.radius_liquid}\n"
            + f"     E_neg: {self.electronegativity_liquid}\n"
        )

    @property
    def bp_solid(self) -> FloatWithUnit:
        return self.base_metal.boiling_point()

    @property
    def bp_liquid(self) -> FloatWithUnit:
        return self.liquid_metal.boiling_point()

    @property
    def electronegativity_solid(self) -> FloatWithUnit:
        return self.base_metal.electronegativity()

    @property
    def electronegativity_liquid(self) -> FloatWithUnit:
        return self.liquid_metal.electronegativity()

    @property
    def radius_solid(self) -> FloatWithUnit:
        return self.base_metal.atomic_radius()

    @property
    def radius_liquid(self) -> FloatWithUnit:
        return self.liquid_metal.atomic_radius()

    @property
    def solid_homologous(self) -> float:
        return self.test_temperature / self.tm_solid

    @staticmethod
    def translate_bravais_code(code: int) -> str:
        bvlmap = {
            "FCC": 1,
            "BCC": 2,
            "SC": 3,
            "HCP": 4,
            "Trigonal": 5,
            "BCT": 6,
            "Tetragonal": 7,
            "oF": 8,
            "oI": 9,
            "oS": 10,
            "oP": 11,
            "mS": 12,
            "mP": 13,
            "aP": 14,
            "Glass": 15,
        }

        return list(bvlmap.keys())[list(bvlmap.values()).index(code)]


class TomeEntryFactory:
    def __init__(self) -> None:
        pass

    def FromExcelComposition(self, entry, is_solid) -> Alloy:
        keys = entry[13:162].keys()

        if is_solid:
            keys = [
                match
                for match in keys
                if (match.lower().endswith("s)") and "wt%" in match.lower())
            ]
        else:
            keys = [
                match
                for match in keys
                if (match.lower().endswith("l)") and "wt%" in match.lower())
            ]

        components = {key[:2].strip(): entry[key] for key in keys if entry[key] > 0}

        return Alloy(components, init_with_wt=True)

    def QueryIntermetallic(self, solid, liquid) -> FloatWithUnit:

        solid_comp = {e.symbol: solid.get_atomic_fraction(e) for e in solid.elements}
        liquid_comp = {e.symbol: liquid.get_atomic_fraction(e) for e in liquid.elements}

        solid_comp = sorted(solid_comp.items(), key=lambda x: x[1], reverse=True)
        liquid_comp = sorted(liquid_comp.items(), key=lambda x: x[1], reverse=True)

        pd_elements = []

        if len(liquid_comp) > 1 and len(solid_comp) > 1:
            pd_elements.append(liquid_comp[0][0])
            pd_elements.append(liquid_comp[1][0])
            pd_elements.append(solid_comp[0][0])
            pd_elements.append(solid_comp[1][0])
        elif len(liquid_comp) == 1 and len(solid_comp) > 2:
            pd_elements.append(liquid_comp[0][0])
            pd_elements.append(solid_comp[0][0])
            pd_elements.append(solid_comp[1][0])
            pd_elements.append(solid_comp[2][0])
        elif len(liquid_comp) > 1 and len(solid_comp) == 1:
            pd_elements.append(liquid_comp[0][0])
            pd_elements.append(liquid_comp[1][0])
            pd_elements.append(solid_comp[0][0])
        elif len(liquid_comp) == 1 and len(solid_comp) > 1:
            pd_elements.append(liquid_comp[0][0])
            pd_elements.append(solid_comp[0][0])
            pd_elements.append(solid_comp[1][0])
        else:
            pd_elements.append(liquid_comp[0][0])
            pd_elements.append(solid_comp[0][0])

        with MPRester(api_key="qdudu9itpKvSDcC0") as a:
            start = timeit.default_timer()
            entries = a.get_entries_in_chemsys(pd_elements)
            stop = timeit.default_timer()
            print(f"Queried MaterialsProject in {(stop-start)} seconds")

        phasediagram = PhaseDiagram(entries)

        intermetallics = {
            p.composition.formula: phasediagram.get_form_energy_per_atom(p)
            for p in phasediagram.stable_entries
        }

        return min(intermetallics.values())

    def JustinStringPhaseToTupleOfInts(self, entry):
        structures = entry["Crystal structure"]

        bravias_lattice_map = {
            "cF": 1,
            "cI": 2,
            "cP": 3,
            "hP": 4,
            "hR": 5,
            "tI": 6,
            "tP": 7,
            "oF": 8,
            "oI": 9,
            "oS": 10,
            "oP": 11,
            "mS": 12,
            "mP": 13,
            "aP": 14,
            "gP": 15,
        }

        if "and" in structures:
            phases = tuple(
                [
                    bravias_lattice_map[structures[:2]],
                    bravias_lattice_map[structures[-2:]],
                ]
            )
        else:
            phases = tuple([bravias_lattice_map[structures[:2]], 0])

        return phases

    def FromExcelTomeLine(self, entry) -> TomeEntry:

        d = {}

        d["id"] = entry["ID Number (old)"]
        d["mechanism"] = entry["Mechanism (guess)"]
        d["tm_liquid"] = FloatWithUnit(entry["Tm of liquid"], "k")
        d["tm_solid"] = FloatWithUnit(entry["Tm of solid metal"], "k")
        d["test_temperature"] = FloatWithUnit(entry["test temperature"], "k")
        d["base_metal"] = self.FromExcelComposition(entry, True)
        d["liquid_metal"] = self.FromExcelComposition(entry, False)
        d["energy_of_intermetallic"] = self.QueryIntermetallic(
            d["base_metal"], d["liquid_metal"]
        )
        d["phase"] = self.JustinStringPhaseToTupleOfInts(entry)

        return TomeEntry(d)
