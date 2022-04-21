from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element, get_el_sp
from pymatgen.core.units import FloatWithUnit, unitized
from typing import Union


class Alloy(Composition):
    """
    Subclasses pymatgen.core.Composition to create a class that includes more useful information for alloys and has the ability to be instantiated with wt% composition.

    Adds properties for average atomic radius, boiling point, and electronegativity, as atomic average of the properties in pymatgen.core.periodic_table.Element
    """

    def __init__(self, *args, strict: bool = False, **kwargs) -> None:

        self.init_with_wt = kwargs.pop("init_with_wt", False)
        self.allow_negative = kwargs.pop("allow_negative", False)

        if len(args) == 1 and isinstance(args[0], (Composition, Alloy)):
            elmap = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            elmap = self._parse_formula(args[0])  # type: ignore
        else:
            elmap = dict(*args, **kwargs)  # type: ignore

        elamt = {}
        elsum = 0
        self._natoms = 0

        if self.init_with_wt:
            for k, v in elmap.items():
                elmap[k] = v * get_el_sp(k).atomic_mass
                elsum += elmap[k]

            for k, v in elmap.items():
                elmap[k] = v / elsum

        for k, v in elmap.items():
            if v < -Alloy.amount_tolerance and not self.allow_negative:
                raise ValueError("Amounts in Composition cannot be negative!")
            if abs(v) >= Alloy.amount_tolerance:
                elamt[get_el_sp(k)] = v
                self._natoms += abs(v)

        self._data = elamt

    @unitized("ang")
    def atomic_radius(self, **kwargs) -> FloatWithUnit:
        """
        :return: Average atomic radius of the alloy using atom fraction averaging. Use keyword arugments 'use_matrix'=true and 'matrix'=Element to return just the matrix atomic radius.
        """
        use_matrix = kwargs.pop("use_matrix", False)
        matrix = kwargs.pop("matrix", None)

        if use_matrix and matrix is not None:
            return matrix.atomic_radius
        else:
            if not any([el.atomic_radius is None for el, amt in self.items()]):
                return (
                    sum((el.atomic_radius * abs(amt) for el, amt in self.items()))
                    / self.num_atoms
                )

    @unitized("K")
    def boiling_point(self, **kwargs) -> FloatWithUnit:
        """
        :return: Average atomic radius of the alloy using atom fraction averaging. Use keyword arugments 'use_matrix'=true and 'matrix'=Element to return just the matrix atomic radius.
        """
        use_matrix = kwargs.pop("use_matrix", False)
        matrix = kwargs.pop("matrix", None)

        if use_matrix and matrix is not None:
            return matrix.boiling_point
        else:
            if not any([el.boiling_point is None for el, amt in self.items()]):
                return (
                    sum((el.boiling_point * abs(amt) for el, amt in self.items()))
                    / self.num_atoms
                )

    def electronegativity(self, **kwargs) -> float:
        """
        :return: Average electronegativity of the alloy using atom fraction averaging. Use keyword arugments 'use_matrix'=true and 'matrix'=Element to return just the matrix atomic radius.
        """
        use_matrix = kwargs.pop("use_matrix", False)
        matrix = kwargs.pop("matrix", None)

        if use_matrix and matrix is not None:
            return matrix.X
        else:
            if not any([el.X is None for el, amt in self.items()]):
                return (
                    sum((el.X * abs(amt) for el, amt in self.items())) / self.num_atoms
                )
