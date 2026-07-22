"""Optional MCP sub-server exposing xraydb reference data.

Mounted with namespace 'xraydb' by :func:`ptychodus_store.mcp_server.create_mcp_server`
when the ``xraydb`` optional extra is installed. Import-guarded: ``import xraydb`` at
module top means this module fails to import when the extra is absent, and the parent
skips mounting.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import xraydb
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel


class ElementInfo(BaseModel):
    symbol: str
    atomic_number: int
    atomic_mass_amu: float
    density_g_per_cm3: float


class XrayEdgeInfo(BaseModel):
    edge: str
    energy_eV: float  # noqa: N815
    fluorescence_yield: float
    jump_ratio: float


class XrayLineInfo(BaseModel):
    siegbahn_name: str
    energy_eV: float  # noqa: N815
    intensity: float
    initial_level: str
    final_level: str


class AnomalousFactorPoint(BaseModel):
    energy_eV: float  # noqa: N815
    f1_electrons: float
    f2_electrons: float


class MassAttenuationPoint(BaseModel):
    energy_eV: float  # noqa: N815
    mu_over_rho_cm2_per_g: float


class LinearAttenuationPoint(BaseModel):
    energy_eV: float  # noqa: N815
    mu_per_m: float


class RefractiveIndexInfo(BaseModel):
    delta: float
    beta: float
    attenuation_length_m: float


class FluorescenceYieldInfo(BaseModel):
    effective_yield: float
    weighted_energy_eV: float  # noqa: N815
    net_probability: float


class ElementIdentification(BaseModel):
    symbol: str
    edge: str


class NamedMaterialInfo(BaseModel):
    name: str
    formula: str
    density_g_per_cm3: float
    categories: list[str]


def _resolve_element(element: str | int) -> str:
    try:
        z = xraydb.atomic_number(str(element)) if isinstance(element, str) else int(element)
        return xraydb.atomic_symbol(z)
    except (ValueError, KeyError) as exc:
        raise ToolError(f'unknown element: {element!r}') from exc


async def _run(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
    return await asyncio.to_thread(func, *args, **kwargs)


def create_xraydb_mcp() -> FastMCP:
    """Build the xraydb MCP sub-server. Caller mounts it with ``namespace='xraydb'``."""
    mcp: FastMCP = FastMCP(name='xraydb')

    @mcp.tool()
    async def element_info(element: str | int) -> ElementInfo:
        """Return atomic number, symbol, atomic mass (AMU), and elemental density (g/cm^3).

        Args:
            element: Chemical symbol (case-sensitive, e.g. 'Fe') or atomic number.
        """
        symbol = _resolve_element(element)
        z = await _run(xraydb.atomic_number, symbol)
        mass = await _run(xraydb.atomic_mass, symbol)
        density = await _run(xraydb.atomic_density, symbol)
        return ElementInfo(
            symbol=symbol,
            atomic_number=int(z),
            atomic_mass_amu=float(mass),
            density_g_per_cm3=float(density),
        )

    @mcp.tool()
    async def xray_absorption_edges(element: str | int) -> list[XrayEdgeInfo]:
        """List all tabulated absorption edges (K, L1-L3, M1-M5, ...) for an element.

        Each entry gives edge energy in eV, fluorescence yield (0-1), and edge-jump ratio.

        Args:
            element: Chemical symbol or atomic number.
        """
        symbol = _resolve_element(element)
        edges = await _run(xraydb.xray_edges, symbol)
        return [
            XrayEdgeInfo(
                edge=name,
                energy_eV=float(row.energy),
                fluorescence_yield=float(row.fyield),
                jump_ratio=float(row.jump_ratio),
            )
            for name, row in edges.items()
        ]

    @mcp.tool()
    async def xray_emission_lines(
        element: str | int,
        excitation_energy_eV: float | None = None,  # noqa: N803
    ) -> list[XrayLineInfo]:
        """List characteristic X-ray fluorescence lines for an element.

        Returns Siegbahn-labelled lines (Ka1, Kb1, La1, ...) with emission energy in eV
        and normalized intensity within their initial-level manifold. If
        ``excitation_energy_eV`` is provided, only lines that can be excited at or below
        that energy are returned.

        Args:
            element: Chemical symbol or atomic number.
            excitation_energy_eV: Optional excitation energy in eV.
        """
        symbol = _resolve_element(element)
        lines = await _run(xraydb.xray_lines, symbol, None, excitation_energy_eV)
        return [
            XrayLineInfo(
                siegbahn_name=name,
                energy_eV=float(row.energy),
                intensity=float(row.intensity),
                initial_level=str(row.initial_level),
                final_level=str(row.final_level),
            )
            for name, row in lines.items()
        ]

    @mcp.tool()
    async def anomalous_scattering_factors(
        element: str | int,
        energies_eV: list[float],  # noqa: N803
    ) -> list[AnomalousFactorPoint]:
        """Real (f1) and imaginary (f2) anomalous atomic scattering factors, in electrons.

        Uses the Chantler tables. Useful for anomalous / resonant ptychography and for
        contrast calculations near an absorption edge.

        Args:
            element: Chemical symbol or atomic number.
            energies_eV: List of photon energies in eV. Must be non-empty.
        """
        if not energies_eV:
            raise ToolError('energies_eV must contain at least one entry')
        symbol = _resolve_element(element)
        f1 = await _run(xraydb.f1_chantler, symbol, energies_eV)
        f2 = await _run(xraydb.f2_chantler, symbol, energies_eV)
        return [
            AnomalousFactorPoint(
                energy_eV=float(e),
                f1_electrons=float(a),
                f2_electrons=float(b),
            )
            for e, a, b in zip(energies_eV, f1, f2, strict=True)
        ]

    @mcp.tool()
    async def mass_attenuation_coefficient(
        element: str | int,
        energies_eV: list[float],  # noqa: N803
        kind: Literal['total', 'photo', 'coh', 'incoh'] = 'total',
    ) -> list[MassAttenuationPoint]:
        """Mass attenuation coefficient mu/rho in cm^2/g for a pure element (Elam tables).

        Args:
            element: Chemical symbol or atomic number.
            energies_eV: List of photon energies in eV. Must be non-empty.
            kind: Cross-section component: 'total' (default), 'photo' (photoabsorption),
                'coh' (coherent scatter), or 'incoh' (incoherent scatter).
        """
        if not energies_eV:
            raise ToolError('energies_eV must contain at least one entry')
        symbol = _resolve_element(element)
        mu = await _run(xraydb.mu_elam, symbol, energies_eV, kind)
        return [
            MassAttenuationPoint(energy_eV=float(e), mu_over_rho_cm2_per_g=float(v))
            for e, v in zip(energies_eV, mu, strict=True)
        ]

    @mcp.tool()
    async def material_refractive_index(
        formula: str,
        density_g_per_cm3: float,
        energy_eV: float,  # noqa: N803
    ) -> RefractiveIndexInfo:
        """X-ray refractive index n = 1 - delta - i*beta at a single photon energy.

        Also returns the 1/e X-ray attenuation length in meters. Use this for phase-object
        design and thickness estimates.

        Args:
            formula: Chemical formula (case-sensitive, e.g. 'SiO2', 'C22H10N2O5') or a
                named material from ``list_named_materials``.
            density_g_per_cm3: Bulk density of the material in g/cm^3.
            energy_eV: Photon energy in eV. Scalar only (xraydb limitation).
        """
        try:
            delta, beta, atlen_cm = await _run(
                xraydb.xray_delta_beta, formula, density_g_per_cm3, energy_eV
            )
        except (ValueError, KeyError) as exc:
            raise ToolError(f'xray_delta_beta failed for {formula!r}: {exc}') from exc
        return RefractiveIndexInfo(
            delta=float(delta),
            beta=float(beta),
            attenuation_length_m=float(atlen_cm) * 1e-2,
        )

    @mcp.tool()
    async def material_attenuation(
        name_or_formula: str,
        energies_eV: list[float],  # noqa: N803
        density_g_per_cm3: float | None = None,
        kind: Literal['total', 'photo'] = 'total',
    ) -> list[LinearAttenuationPoint]:
        """Linear X-ray attenuation coefficient mu in 1/m for a compound or named material.

        The 1/e attenuation length is 1 / mu_per_m (meters).

        Args:
            name_or_formula: Chemical formula (case-sensitive) or a named material from
                ``list_named_materials`` (case-insensitive). For named materials, density
                may be omitted.
            energies_eV: List of photon energies in eV. Must be non-empty.
            density_g_per_cm3: Bulk density in g/cm^3. Optional if a named material
                supplies its own density.
            kind: 'total' (default) or 'photo' (photoabsorption only).
        """
        if not energies_eV:
            raise ToolError('energies_eV must contain at least one entry')
        try:
            mu = await _run(
                xraydb.material_mu, name_or_formula, energies_eV, density_g_per_cm3, kind
            )
        except (ValueError, KeyError) as exc:
            raise ToolError(f'material_mu failed for {name_or_formula!r}: {exc}') from exc
        return [
            LinearAttenuationPoint(energy_eV=float(e), mu_per_m=float(v) * 100.0)
            for e, v in zip(energies_eV, mu, strict=True)
        ]

    @mcp.tool()
    async def effective_fluorescence_yield(
        element: str | int,
        edge: str,
        line: str,
        excitation_energy_eV: float,  # noqa: N803
    ) -> FluorescenceYieldInfo:
        """Effective fluorescence yield for one emission line under a given excitation.

        Accounts for Coster-Kronig cascades within the L or M manifolds. Returns the
        effective yield, an intensity-weighted mean line energy in eV, and the net
        probability that the excitation produces this line.

        Args:
            element: Chemical symbol or atomic number.
            edge: IUPAC edge name (e.g. 'K', 'L3').
            line: Siegbahn line name (e.g. 'Ka1', 'La1').
            excitation_energy_eV: Excitation photon energy in eV. Must be at or above the
                edge energy for a non-zero result.
        """
        symbol = _resolve_element(element)
        try:
            fyield, weighted_energy, net_prob = await _run(
                xraydb.fluor_yield, symbol, edge, line, excitation_energy_eV
            )
        except (ValueError, KeyError) as exc:
            raise ToolError(
                f'fluor_yield failed for {symbol!r} edge={edge!r} line={line!r}: {exc}'
            ) from exc
        return FluorescenceYieldInfo(
            effective_yield=float(fyield),
            weighted_energy_eV=float(weighted_energy),
            net_probability=float(net_prob),
        )

    @mcp.tool()
    async def identify_element_by_edge_energy(
        observed_energy_eV: float,  # noqa: N803
        edges: list[str] | None = None,
    ) -> ElementIdentification:
        """Best-guess element and absorption edge for an observed edge energy.

        Useful for XANES / EXAFS feature identification. Not intended for identifying
        fluorescence emission lines - use ``xray_emission_lines`` for that.

        Args:
            observed_energy_eV: Approximate edge energy in eV.
            edges: Edges to consider. Defaults to ('K', 'L3', 'L2', 'L1', 'M5').
        """
        edges_tuple = tuple(edges) if edges else ('K', 'L3', 'L2', 'L1', 'M5')
        try:
            symbol, edge = await _run(xraydb.guess_edge, observed_energy_eV, edges_tuple)
        except (ValueError, KeyError) as exc:
            raise ToolError(f'guess_edge failed: {exc}') from exc
        return ElementIdentification(symbol=str(symbol), edge=str(edge))

    @mcp.tool()
    async def core_hole_width(element: str | int, edge: str | None = None) -> float:
        """Natural core-hole linewidth in eV. Returns the K-edge width if edge omitted.

        Args:
            element: Chemical symbol or atomic number.
            edge: Optional IUPAC edge name (e.g. 'L3'). If omitted, returns the K width.
        """
        symbol = _resolve_element(element)
        try:
            width = await _run(xraydb.core_width, symbol, edge)
        except (ValueError, KeyError) as exc:
            raise ToolError(f'core_width failed for {symbol!r}: {exc}') from exc
        return float(width)

    @mcp.tool()
    async def parse_chemical_formula(formula: str) -> dict[str, float]:
        """Parse a chemical formula into an {element_symbol: stoichiometric_count} map.

        Example: 'SiO2' -> {'Si': 1.0, 'O': 2.0}. Case-sensitive.

        Args:
            formula: Chemical formula string.
        """
        try:
            parsed = await _run(xraydb.chemparse, formula)
        except (ValueError, KeyError) as exc:
            raise ToolError(f'chemparse failed for {formula!r}: {exc}') from exc
        return {str(k): float(v) for k, v in parsed.items()}

    @mcp.tool()
    async def list_named_materials() -> list[NamedMaterialInfo]:
        """Enumerate xraydb's built-in named materials (e.g. 'kapton', 'water', 'silicon').

        The returned names are valid inputs to ``material_refractive_index`` and
        ``material_attenuation``. Includes chemical formula, density, and categories.
        """
        from xraydb.materials import get_materials

        mats = await _run(get_materials)
        return [
            NamedMaterialInfo(
                name=str(name),
                formula=str(mat.formula),
                density_g_per_cm3=float(mat.density),
                categories=list(mat.categories),
            )
            for name, mat in mats.items()
        ]

    return mcp
