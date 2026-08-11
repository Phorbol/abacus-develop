# fmt: off

'''
here the ase-abacus implementation is pasted and modified. 
Source:
https://gitlab.com/1041176461/ase-abacus/-/blob/master/ase/calculators/abacus.py

This module defines an ASE interface to ABACUS.
Created on Fri Jun  8 16:33:38 2018

ABACUS (Atomic-orbital Based Ab-initio Computation at UStc) is an open-source 
package based on density functional theory (DFT). The package utilizes both plane 
wave and numerical atomic basis sets with the usage of pseudopotentials to describe 
the interactions between nuclear ions and valence electrons. ABACUS supports LDA, 
GGA, meta-GGA, and hybrid functionals. Apart from single-point calculations, 
the package allows geometry optimizations and ab-initio molecular dynamics with 
various ensembles. The package also provides a variety of advanced functionalities 
for simulating materials, including the DFT+U, VdW corrections, and implicit solvation
model, etc. In addition, ABACUS strives to provide a general infrastructure to 
facilitate the developments and applications of novel machine-learning-assisted 
DFT methods (DeePKS, DP-GEN, DeepH, DeePTB etc.) in molecular and material simulations.

Modified on Wed Jun 20 15:00:00 2018
@author: Shen Zhen-Xiong

Modified on Wed Jun 03 23:00:00 2022
@author: Ji Yu-yang

Refactored from Sun Dec 07 21:41 2025
@author: Huang Yi-ke
'''

import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
    read_stdout
)
from ase.calculators.socketio import SocketIOCalculator
from ase.atoms import Atoms
from ase.dft.kpoints import BandPath
from ase.io import read

from abacuslite.io.generalio import (
    file_safe_backup,
    read_input,
    read_stru,
    read_kpt,
    species_group_indices,
    write_input,
    write_stru,
    write_kpt
)

__LEGACYIO__ = True
def switch_io_backend_version(version: str) -> bool:
    '''determine if the i/o is in legacy format by the version number,
    for detailed discussion, see issue #7260
    '''
    global __LEGACYIO__
    m = re.match(r'^v(\d+)\.(\d+)\.(\d+)(\.\d+|\-(alpha|beta|rc)\.\d+|\-(alpha|beta|rc)\d+)?$', version)
    assert m, f'Invalid format of version number, please check file version.h'
    assert int(m.group(1)) >= 3, f'ABACUS v2.x is not supported'
    if int(m.group(2)) >= 11:
        __LEGACYIO__ = False
    elif int(m.group(2)) == 9:
        if m.group(4) is not None:
            # it is also possible to divide the version more carefully,
            # but because 3.9.0.x are all on the develop branch, up to
            # now, there is no user submit issue to request such a careful
            # division
            __LEGACYIO__ = False
    else:
        __LEGACYIO__ = True
    return __LEGACYIO__

class AbacusProfile(BaseProfile):
    '''AbacusProfile for interacting the ASE with ABACUS that installed in
    the practical system'''
    configvars = {'pseudo_dir', 'orbital_dir'}

    def __init__(self, 
                 command: str, 
                 pseudo_dir: Optional[str | Path] = None, 
                 orbital_dir: Optional[str | Path] = None, 
                 omp_num_threads: Optional[int] = None,
                 **kwargs):
        '''Initialize ABACUS profile.
        
        Parameters
        ----------
        command : str
            The command to run ABACUS. NOTE: there may be the case for some
            sophisticated ABACUS user they call ABACUS with command like
            `OMP_NUM_THREADS=1 mpirun -np X abacus`. Here please do not set
            the number of omp threads in `command`, instead, use `nomp=1`.
        pseudo_dir : str or Path, optional
            The directory containing pseudopotential files.
        orbital_dir : str or Path, optional
            The directory containing orbital basis files. This is only necessary
            for an ABACUS-LCAO calculation
        omp_num_threads : int, optional
            The number of omp threads to use.
        '''
        assert isinstance(command, str)
        # further validation on the command will be in the __init__ of
        # the base class
        super().__init__(command, **kwargs)
        self.pseudo_dir  = pseudo_dir
        self.orbital_dir = orbital_dir

        if omp_num_threads is not None:
            # set the number of omp threads for the present process
            assert isinstance(omp_num_threads, int)
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    @staticmethod
    def parse_version(stdout) -> str:
        # MPI launchers may add informational lines before ABACUS output.
        match = re.search(r'ABACUS version (\S+)', stdout or '')
        if match is None:
            raise RuntimeError(
                'Could not parse ABACUS version from command output. '
                'Expected a line like "ABACUS version vX.Y.Z".'
            )
        return match.group(1)

    def get_calculator_command(self, inputfile) -> List[str]:
        # because ABACUS run in the folder where there are INPUT files, so the
        # additional inputfile argument is not used.
        return []

    def socketio_argv_inet(self, port: Optional[int] = None) -> List[str]:
        port = 31415 if port is None else port
        return [
            'env',
            f'ABACUS_SOCKET_ADDRESS=localhost:{port}',
            *self._split_command,
        ]

    def socketio_argv_unix(self, socket: str) -> List[str]:
        return [
            'env',
            f'ABACUS_SOCKET_ADDRESS=/tmp/ipi_{socket}:UNIX',
            *self._split_command,
        ]

    def version(self) -> str:
        '''get the abacus version information'''
        cmd_ = [*self._split_command, '--version']
        return AbacusProfile.parse_version(read_stdout(cmd_))

class AbacusTemplate(CalculatorTemplate):
    
    implemented_properties = [
        'energy', 'forces', 'stress', 'free_energy', 'magmom'
    ]
    _label = 'abacus'

    def __init__(self):
        super().__init__(
            'abacus',
            self.implemented_properties
        )
        self.non_convergence_ok = False
        # the redirect stdout and stderr
        self.inputname  = 'INPUT' # hard-coded
        self.outputname = f'{self._label}.out'
        self.errorname  = f'{self._label}.err'

        # fix: inconsistent atoms order may induce bugs, here a list
        # is kept to swap the order of atoms
        self.atomorder  = None

    '''because it may be not one-to-one mapping between the property
    desired to calculate and the keywords used in the calculation,
    in the following a series of functions for mapping the property
    calculation to the keywords settings are implemented'''
    @staticmethod
    def get_energy_keywords(self) -> Dict[str, str]:
        return {}

    @staticmethod
    def get_forces_keywords(self) -> Dict[str, str]:
        return {'cal_force': '1'}
    
    @staticmethod
    def get_stress_keywords(self) -> Dict[str, str]:
        return {'cal_stress': '1'}

    @staticmethod
    def get_free_energy_keywords(self) -> Dict[str, str]:
        return {}

    @staticmethod
    def get_magmom_keywords(self) -> Dict[str, str]:
        return {'nspin': '2'}

    def get_property_keywords(self,
                              parameters: Dict[str, str],
                              properties: List[str]) -> Dict[str, str]:
        '''Connect the relationship between the properties calculation and
        the ABACUS keywords. May be more complicated in the future, therefore
        it is better to have a separate mapping function instead of
        implementing in some other functions.
        
        Parameters
        ----------
        parameters : dict
            The parameters used to perform the calculation.
        properties : list of str
            The list of properties to calculate
        '''
        def keyword_compare_value(value):
            if isinstance(value, bool):
                return '1' if value else '0'
            if isinstance(value, (list, tuple, set)):
                return ' '.join(str(i) for i in value)
            return str(value)

        param_cache_ = {
            key: keyword_compare_value(value)
            for key, value in parameters.items()
            if value is not None
        }

        def counter(param_new: Dict[str, str]) -> Dict[str, str]:
            info = 'desired properties or explicit parameters required contradictory keywords'
            staged = {}
            for k, v in param_new.items():
                if v is None:
                    continue
                normalized_value = keyword_compare_value(v)
                if k in param_cache_ and param_cache_[k] != normalized_value:
                    raise ValueError(f'{info}: {k}={v} (now), {param_cache_[k]} (before)')
                staged[k] = normalized_value
            param_cache_.update(staged)
            return param_new

        # update the parameters with the keywords for the properties
        for p in properties:
            assert p in self.implemented_properties
            parameters.update(counter(getattr(self, f'get_{p}_keywords')(parameters)))
        
        # from the parameters, get the file path
        self.suffix = parameters.get('suffix', 'ABACUS')
        self.calculation = parameters.get('calculation', 'scf')
        # with the above two, the running log file can be positioned.
        return parameters

    def write_input(self, 
                    profile: AbacusProfile, 
                    directory: Path | str,
                    atoms: Atoms, 
                    parameters: Dict[str, str],
                    properties: List[str]) -> None:
        '''Write the input files for the calculation. This function connects
        the calculation in ASE language (atoms, properties, assisted by the
        parameters) to the input files of ABACUS.

        Parameters
        ----------
        profile : AbacusProfile
            The profile used to perform the calculation.
        directory : Path
            The working directory to store the input files.
        atoms : Atoms
            The atoms object to perform the calculation on. Because 
        parameters: dict
            The parameters used to perform the calculation.
        properties: list of str
            The list of properties to calculate
        '''
        # directory
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # copy the `parameters` because later we will modify it
        parameters = parameters.copy()

        # STRU
        _ = file_safe_backup(directory / parameters.get('stru_file', 'STRU'))
        # group atoms by first-occurrence species order. Keep the reverse map
        # so that we will recover the order in function read_results()
        ind = species_group_indices(atoms.get_chemical_symbols())
        self.atomorder = sorted(range(len(atoms)), key=lambda i: ind[i]) # revmap
        # then we write
        _ = write_stru(atoms[ind], 
                       outdir=directory, 
                       pp_file=parameters.get('pseudopotentials'),
                       orb_file=parameters.get('basissets'),
                       fname=parameters.get('stru_file', 'STRU'))

        # KPT, if needed
        if 'kpts' in parameters:
            _ = file_safe_backup(directory / parameters.get('kpoint_file', 'KPT'))
            _ = write_kpt(parameters['kpts'], 
                          directory / parameters.get('kpoint_file', 'KPT'))
        # should this function be responsible for checking the integrity
        # of information provided by the user? There may be the case that
        # user provides incomplete information, such that the ABACUS cannot
        # run with parameters.

        # INPUT
        # after writing the KPT and STRU, delete them from the parameters
        _ = parameters.pop('kpts', None)

        _ = parameters.pop('pseudopotentials', None)
        parameters.update({'pseudo_dir': profile.pseudo_dir})

        _ = parameters.pop('basissets', None)
        parameters.update({'orbital_dir': profile.orbital_dir})
        # update the parameters respect to the properties desired
        parameters = self.get_property_keywords(parameters, properties)
        # postprocess on the parameters: convert the key and values
        # from any to string. For the case where the value is a 
        # array, convert to the string spaced by whitespace
        for k, v in parameters.items():
            # if the v is iterable, convert to the string spaced by whitespace
            if isinstance(v, (list, tuple, set)):
                parameters[k] = ' '.join(str(i) for i in v)
        dst = directory / self.inputname
        _ = file_safe_backup(dst)
        # remove possible key-value pairs whose value is None
        parameters = {k: v for k, v in parameters.items() if v is not None}

        # FIXME: only support the ksdft esolver_type presently
        if parameters.get('esolver_type', 'ksdft') != 'ksdft':
            raise NotImplementedError(
                'ABACUS Lite only supports the ksdft esolver_type presently, '
                'which means the ABACUS should always be used as a DFT '
                'calculator. For other forcefields that ABACUS supports '
                'such as the LJ, DP, etc., please either use the ABACUS '
                'directly, or the implementation of interfaces to ASE '
                'directly.'
            )

        # write the INPUT file to the target directory
        _ = write_input(parameters, dst)

    def execute(self, 
                directory: Path | str, 
                profile: AbacusProfile):
        '''Execute the ABACUS Lite calculation.

        Parameters
        ----------
        directory : Path or str
            The working directory to store the input files.
        profile : AbacusProfile
            The profile used to perform the calculation.

        Raises
        ------
        SubprocessError
            If the ABACUS Lite calculation fails.
        '''
        from subprocess import SubprocessError
        try:
            profile.run(directory=directory, 
                        inputfile=None, 
                        outputfile=self.outputname, 
                        errorfile=self.errorname)
        except SubprocessError:
            message = ['ABACUS Lite calculation failed']
            with open(directory / self.outputname, 'r') as f:
                message.append(f.read())
            with open(directory / self.errorname, 'r') as f:
                message.append(f.read())
            raise SubprocessError('\n'.join(message))

    def read_results(self, directory) -> Dict:
        '''the function that returns the desired properties in dict'''
        read_abacus_out = lambda fn: None
        global __LEGACYIO__
        if __LEGACYIO__:
            from abacuslite.io.legacyio import read_abacus_out
        else:
            from abacuslite.io.latestio import read_abacus_out

        outdir = directory / f'OUT.{self.suffix}'
        # only the last frame
        atoms: Optional[Atoms] = read_abacus_out(
            outdir / f'running_{self.calculation}.log',
            sort_atoms_with=self.atomorder)[-1]
        assert atoms is not None

        return dict(atoms.calc.properties())

    def load_profile(self, cfg, **kwargs):
        return AbacusProfile.from_config(cfg, self.name, **kwargs)

class Abacus(GenericFileIOCalculator):
    def __init__(self, 
                 profile=None, 
                 directory='.', 
                 **kwargs):
        '''Construct the ABACUS calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' or any of ABACUS'
        native keywords.

        Parameters
        ----------
        profile: AbacusProfile
            the interface that interacts with the ABACUS executable.
        directory: str or Path
            the working directory to store the input files.
        pseudopotentials: dict
            A mapping from the element to the pseudopotential file name,
            e.g. ``{'O': 'O_ONCV_PBE-1.0.upf', 'H': 'H.upf'}``.
        baisssets: dict, optional
            A mapping from the element to the ABACUS numerical atomic 
            orbital file name. This is necessary only when it is an
            ABACUS-LCAO (Linear-Combination-of-Atomic-Orbitals) calculation
            e.g. ``{'O': 'O_gga_10au_100Ry_2s2p1d.orb', 
            'H': 'H_gga_10au_100Ry_2s1p.orb'}``.
        kpts: dict
            The k-points sampling should be given as a dict. For there
            are many modes of k-sampling supported, the content may differ
            in cases. A `mode` key should be used to specify the ksampling,
            allowed modes are: `mp-sampling`, `line` and `point`. For 
            `mp-sampling` mode, `gamma-centered`, `nk` and `kshift` should
            present. `gamma-centered` is a boolean, `nk` and `kshift` should
            be lists of three integers. ... TBD
        inp: dict
            parameters setting in INPUT of ABACUS. NOTE: if there are settings
            on the `pseudo_dir` and `orbital_dir`, these will overwrite the
            value in the profile. If you do not expect this, please only use
            the profile, because the profile stands for interfacing with the
            ASE calculator instance with the computational environment.

        **kwargs:
            Other parameters to be passed to the ABACUS calculator.
        '''
        # not recommended :(
        profile = AbacusProfile('abacus') if profile is None else profile

        # to be compatible with both the legacy and latest format of i/o, the
        # switch is needed. 
        _ = switch_io_backend_version(profile.version())

        # because ABACUS run job in folders, based on the assumption that
        # there is only one job in the folder. Therefore once there are already
        # files in the folder, will try to create a new one...(seriously?)
        inp = kwargs.pop('inp', {})

        super().__init__(
            template=AbacusTemplate(),
            profile=profile,
            parameters=kwargs | inp,
            directory=directory,
        )

    def write_input(self, atoms, properties=None, system_changes=None):
        if properties is None:
            properties = self.template.implemented_properties
        self.template.write_input(
            profile=self.profile,
            directory=Path(self.directory),
            atoms=atoms,
            parameters=self.parameters,
            properties=properties,
        )

    @classmethod
    def restart(cls, profile=None, directory='.', **kwargs):
        '''instantiate one ABACUS calculator from an existing job directory,
        optionally overwrite some keywords'''
        directory = Path(directory)
        inp_read = read_input(directory / 'INPUT')

        pporb_read = read_stru(directory / 'STRU')['species']
        pseudopotentials = kwargs.get(
            'pseudopotentials',
            {pporb['symbol']: pporb['pp_file'] for pporb in pporb_read}
        )
        if 'pseudopotentials' in kwargs:
            del kwargs['pseudopotentials']
        
        basissets = kwargs.get(
            'basissets',
            {pporb['symbol']: pporb.get('orb_file') for pporb in pporb_read}
        )
        if 'basissets' in kwargs:
            del kwargs['basissets']
        if all([forb is None for forb in basissets.values()]):
            basissets = {}
        assert all([forb is not None for forb in basissets.values()])

        kpts = kwargs.get('kpts', read_kpt(directory / inp_read.get('kpoint_file', 'KPT')))
        if 'kpts' in kwargs:
            del kwargs['kpts']

        inp = inp_read | kwargs.get('inp', {})
        if 'inp' in kwargs:
            del kwargs['inp']

        return cls(profile=profile, 
                   directory=directory,
                   pseudopotentials=pseudopotentials,
                   basissets=basissets,
                   kpts=kpts,
                   inp=inp,
                   **kwargs)

    def fixed_density(self,
                      kpts: BandPath | Dict[str, str | int | List[float]],
                      symmetry: str = 'off', 
                      profile=None, 
                      **kwargs) -> 'Abacus':
        '''spawn a new ABACUS calculator with fixed density, based on the present
        instance. This funcionality is mostly only useful when perform the 
        non-self-consistent calculations like band structure.
        This interface is referred from the ASE document at:
        https://ase-lib.org/gettingstarted/tut04_bulk/bulk.html#band-structure
        , however, we also note that it is from the implementation of the 
        GPAW python, not the ASE official.
        To make less development burden as possible, we use the same interface
        as the GPAW python.

        Parameters
        ----------
        kpts : BandPath | Dict[str, str | int | List[float]]
            The k-point path to be calculated. Can be either a BandPath object
            or a dictionary that contains the k-point information. For the latter
            case, see tbgen/calculators/abacus/generalio.py::write_kpt for more
            details.
        symmetry : str, optional
            The symmetry mode to be used. Default is 'off'. Now only the `off`
            mode is supported.
        profile : AbacusProfile, optional
            The profile to be used. Default is None. If None, the profile of
            the present instance will be used.
        **kwargs : dict
            Other parameters to be passed to the ABACUS calculator.
        
        Returns
        -------
        Abacus
            The new ABACUS calculator instance that can perform the nscf calculation
            tasks
        '''
        # we should overwrite the 'calculation' to 'nscf', and 'init_chg' to 'file'
        assert symmetry == 'off'
        
        kwargs.setdefault('inp', {}).update({'calculation': 'nscf',
                                             'init_chg': 'file',
                                             'symmetry': 0,
                                             'out_band': 1,
                                             'kspacing': 0.0,       # overwrite
                                             'gamma_only': False,
                                             'read_file_dir': 'OUT.ABACUS'})  # overwrite

        profile = self.profile if profile is None else profile

        # get the kpoint coordinates
        if isinstance(kpts, BandPath):
            kwargs['kpts'] = {
                'mode': 'point',
                'nk': len(kpts.kpts),
                'nkinterpl': np.ones(len(kpts.kpts), dtype=int).tolist(),
                'coordinate': 'direct',
                'kpoints': kpts.kpts.tolist(),
            }
        else:
            assert isinstance(kpts, dict)
            kwargs['kpts'] = kpts
        
        # return
        return Abacus.restart(profile=profile, 
                              directory=self.directory,
                              **kwargs)

    def band_structure(self, efermi=None):
        '''get the band structure from ABACUS. 
        (now not only GPAW can calculate the band structure ;) )'''
        from ase.spectrum.band_structure import get_band_structure
        return get_band_structure(calc=self, reference=efermi)

class AbacusSocketIO(SocketIOCalculator):
    """ASE socket I/O calculator that launches ABACUS as an i-PI client.

    A socket calculator owns one ABACUS process with one fixed INPUT/STRU
    setup. The i-PI protocol can update positions and, when explicitly
    enabled, the cell, but electronic-structure parameters such as k-points,
    spin, basis, pseudopotentials, and species require a new calculator
    instance. Energy, forces, and stress are independent ASE properties. The
    calculator starts with the requested property set and transparently
    restarts the socket if a later call needs an additional property. The
    fixed-layout i-PI response uses zero padding for absent wire fields and
    an explicit extras metadata record so padding is never exposed as a
    computed property.
    """

    def __init__(self,
                 profile=None,
                 directory='.',
                 port=None,
                 unixsocket=None,
                 timeout=None,
                 log=None,
                 variable_cell=False,
                 **kwargs):
        inp = dict(kwargs.pop('inp', {}))
        self.variable_cell = self._input_bool(
            variable_cell, 'variable_cell')
        self._property_constraints = {}
        for keyword, property_name in (('cal_force', 'forces'),
                                       ('cal_stress', 'stress')):
            if keyword in inp:
                self._property_constraints[property_name] = self._input_bool(
                    inp[keyword], keyword)
        self.implemented_properties = [
            'energy', 'free_energy', 'forces', 'stress']
        self._active_properties = None
        self._last_socket_metadata = None
        self.last_scf_converged = None
        inp = self._socket_inp(inp, self.variable_cell)
        self.abacus = Abacus(
            profile=profile,
            directory=directory,
            inp=inp,
            **kwargs,
        )
        self._reference_cell = None
        super().__init__(
            port=port,
            unixsocket=unixsocket,
            timeout=timeout,
            log=log,
            launch_client=self._launch_client,
        )

    def calculate(self, atoms=None, properties=None, system_changes=None):
        from ase.calculators.calculator import (
            PropertyNotImplementedError,
            all_changes,
        )
        from ase.stress import full_3x3_to_voigt_6_stress

        if system_changes is None:
            system_changes = all_changes
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError('AbacusSocketIO.calculate requires atoms')

        requested = self._normalize_socket_properties(properties)
        self._check_requested_properties(requested)
        bad = [change for change in system_changes
               if change not in self.supported_changes]
        if self.atoms is not None and any(bad):
            raise PropertyNotImplementedError(
                'Cannot change {} through IPI protocol. '
                'Please create new socket calculator.'
                .format(bad if len(bad) > 1 else bad[0]))

        self._check_variable_cell_geometry(atoms)
        desired = set(requested)
        desired.discard('free_energy')
        desired.add('energy')
        for property_name, enabled in getattr(
                self, '_property_constraints', {}).items():
            if enabled:
                desired.add(property_name)
        active = set(getattr(self, '_active_properties', ()) or ())
        if not active:
            # A pre-existing server with no recorded mask is kept for
            # backwards-compatible calculator subclasses and test doubles.
            active.update(desired)
        elif not desired.issubset(active):
            active.update(desired)
            if getattr(self, 'server', None) is not None:
                self._close_socket_session()
        self._active_properties = tuple(
            name for name in ('energy', 'forces', 'stress') if name in active)

        self._check_cell_change(atoms)
        order = self._socket_sort_indices(atoms)
        socket_atoms = atoms[order]
        self.atoms = atoms.copy()

        if self.server is None:
            self.server = self.launch_server()
            proc = self.launch_client(socket_atoms,
                                      list(self._active_properties),
                                      port=self._port,
                                      unixsocket=self._unixsocket)
            self.server.proc = proc

        raw_results = self.server.calculate(socket_atoms)
        if not isinstance(raw_results, dict):
            raise ValueError('ABACUS socket server returned a non-mapping result')
        results = dict(raw_results)
        metadata = self._decode_socket_metadata(results.pop('morebytes', None))
        self._last_socket_metadata = metadata
        if metadata is None:
            present = set(self._active_properties)
            self.last_scf_converged = None
        else:
            present = set(metadata['present'])
            self.last_scf_converged = metadata['scf_converged']

        if 'energy' not in present or 'energy' not in results:
            raise ValueError('ABACUS socket response did not provide energy')
        energy = float(results['energy'])
        if not np.isfinite(energy):
            raise ValueError('ABACUS socket energy is not finite')
        free_energy = float(results.get('free_energy', energy))
        if not np.isfinite(free_energy):
            raise ValueError('ABACUS socket free energy is not finite')
        current = {
            'energy': energy,
            'free_energy': free_energy,
        }

        if 'forces' in present:
            if 'forces' not in results:
                raise ValueError(
                    'ABACUS socket metadata advertises forces, but the wire response omitted them')
            forces = np.asarray(results['forces'], dtype=np.float64)
            expected_shape = (len(socket_atoms), 3)
            if forces.shape != expected_shape:
                raise ValueError(
                    'ABACUS socket force shape {} does not match {}'.format(
                        forces.shape, expected_shape))
            if not np.all(np.isfinite(forces)):
                raise ValueError('ABACUS socket forces are not finite')
            current['forces'] = self._forces_to_input_order(forces, order)

        if 'stress' in present:
            virial = results.get('virial')
            if virial is None:
                raise ValueError(
                    'ABACUS socket metadata advertises stress, but the wire response omitted virial')
            if self.atoms.cell.rank != 3 or not any(self.atoms.pbc):
                raise PropertyNotImplementedError(
                    'ABACUS socket stress requires a periodic rank-3 cell')
            virial = np.asarray(virial, dtype=np.float64)
            if virial.shape != (3, 3) or not np.all(np.isfinite(virial)):
                raise ValueError('ABACUS socket virial is not a finite 3x3 matrix')
            vol = float(atoms.get_volume())
            if not np.isfinite(vol) or vol <= 0.0:
                raise ValueError('ABACUS socket stress requires a positive cell volume')
            current['stress'] = -full_3x3_to_voigt_6_stress(virial) / vol

        # Replace the result map instead of updating it: an E-only or E+S
        # frame must not inherit a force/stress key from an earlier frame.
        self.results = current
        missing = [name for name in requested if name not in current]
        if missing:
            raise PropertyNotImplementedError(
                'ABACUS socket response did not provide requested {}'.format(
                    ', '.join(missing)))

    @staticmethod
    def _normalize_socket_properties(properties):
        from ase.calculators.calculator import PropertyNotImplementedError

        if properties is None:
            names = ['energy']
        elif isinstance(properties, str):
            names = [properties]
        else:
            names = list(properties)
        if not names:
            names = ['energy']
        allowed = {'energy', 'free_energy', 'forces', 'stress'}
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise PropertyNotImplementedError(
                'ABACUS socket does not implement {}'.format(
                    ', '.join(unknown)))
        return tuple(dict.fromkeys(names))

    def _check_requested_properties(self, requested):
        from ase.calculators.calculator import PropertyNotImplementedError

        constraints = getattr(self, '_property_constraints', {})
        keywords = {'forces': 'cal_force', 'stress': 'cal_stress'}
        for property_name, keyword in keywords.items():
            if property_name in requested and constraints.get(property_name) is False:
                raise PropertyNotImplementedError(
                    '{}={} disables requested {}'.format(
                        keyword, 0, property_name))

    def _close_socket_session(self):
        server = getattr(self, 'server', None)
        if server is not None:
            close = getattr(server, 'close', None)
            if callable(close):
                close()
        self.server = None
        self.results = {}

    @staticmethod
    def _decode_socket_metadata(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            payload = raw.encode('utf-8')
        elif isinstance(raw, (bytes, bytearray, memoryview)):
            payload = bytes(raw)
        else:
            payload = np.asarray(raw, dtype=np.uint8).tobytes()
        if not payload:
            return None
        try:
            metadata = json.loads(payload.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                'ABACUS socket extras are not valid UTF-8 JSON') from error
        if not isinstance(metadata, dict):
            raise ValueError('ABACUS socket extras must be a JSON object')
        if metadata.get('schema') != 'abacus.socket.properties.v1':
            raise ValueError('unsupported ABACUS socket extras schema')
        present = metadata.get('present')
        if not isinstance(present, list):
            raise ValueError('ABACUS socket extras present must be a list')
        allowed = {'energy', 'forces', 'stress'}
        if any(not isinstance(name, str) or name not in allowed
               for name in present):
            raise ValueError('ABACUS socket extras contain an unknown property')
        if 'energy' not in present:
            raise ValueError('ABACUS socket extras must include energy')
        scf_converged = metadata.get('scf_converged')
        if not isinstance(scf_converged, bool):
            raise ValueError(
                'ABACUS socket extras scf_converged must be Boolean')
        return {
            'present': tuple(dict.fromkeys(present)),
            'scf_converged': scf_converged,
        }

    def _check_cell_change(self, atoms):
        from ase.calculators.calculator import PropertyNotImplementedError

        if self.variable_cell:
            return
        cell = atoms.cell.array.copy()
        if self._reference_cell is None:
            self._reference_cell = cell
            return
        max_delta = np.max(np.abs(cell - self._reference_cell))
        if max_delta > 1.0e-10:
            raise PropertyNotImplementedError(
                'AbacusSocketIO is fixed-cell only; create a new socket '
                'calculator for a changed cell, or use the normal Abacus '
                'FileIO calculator for variable-cell workflows.'
            )

    def _check_variable_cell_geometry(self, atoms):
        if not self.variable_cell:
            return
        if atoms.cell.rank != 3 or not np.all(np.asarray(atoms.pbc, dtype=bool)):
            raise ValueError(
                'AbacusSocketIO variable-cell mode requires a full rank-3 '
                'periodic cell')

    def set(self, **kwargs):
        if kwargs:
            raise ValueError(
                'AbacusSocketIO input parameters are fixed after construction; '
                'create a new AbacusSocketIO calculator to change k-points, '
                'spin, basis, pseudopotentials, species, or other INPUT/STRU '
                'settings.'
            )
        return super().set(**kwargs)

    def _launch_client(self, atoms, properties=None, port=None, unixsocket=None):
        from subprocess import Popen

        if properties is None:
            properties = list(getattr(self, '_active_properties', ()) or ('energy',))

        directory = Path(self.abacus.directory)
        directory.mkdir(exist_ok=True, parents=True)

        if hasattr(self.abacus, 'write_inputfiles'):
            self.abacus.write_inputfiles(atoms, properties)
        else:
            self.abacus.write_input(atoms, properties=properties)

        if unixsocket is not None:
            argv = self.abacus.profile.socketio_argv_unix(socket=unixsocket)
        else:
            argv = self.abacus.profile.socketio_argv_inet(port=port)

        stdout = open(directory / self.abacus.template.outputname, 'w')
        stderr = open(directory / self.abacus.template.errorname, 'w')
        try:
            return Popen(argv, cwd=directory, env=os.environ,
                         stdout=stdout, stderr=stderr)
        finally:
            stdout.close()
            stderr.close()

    @staticmethod
    def _socket_inp(inp, variable_cell):
        inp = dict(inp)
        variable_cell = AbacusSocketIO._input_bool(
            variable_cell, 'variable_cell')
        if 'socket_variable_cell' in inp:
            socket_variable_cell = AbacusSocketIO._input_bool(
                inp.pop('socket_variable_cell'), 'socket_variable_cell')
            if socket_variable_cell != variable_cell:
                raise ValueError(
                    'socket_variable_cell conflicts with variable_cell')
        calculation = inp.get('calculation', 'scf')
        if calculation != 'scf':
            raise ValueError('ABACUS socket I/O requires calculation="scf"')
        for keyword in ('cal_force', 'cal_stress'):
            if keyword in inp:
                inp[keyword] = int(AbacusSocketIO._input_bool(
                    inp[keyword], keyword))
        inp.update({
            'calculation': 'scf',
            'socket_driver': 1,
        })
        if variable_cell:
            inp['socket_variable_cell'] = 1
        return inp

    @staticmethod
    def _input_bool(value, name):
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('true', '1'):
                return True
            if normalized in ('false', '0'):
                return False
        raise ValueError(
            '{} must be one of true, false, 1, or 0'.format(name))

    @staticmethod
    def _socket_sort_indices(atoms):
        return species_group_indices(atoms.get_chemical_symbols())

    @staticmethod
    def _forces_to_input_order(forces, order):
        reordered = np.empty_like(forces)
        for sorted_index, original_index in enumerate(order):
            reordered[original_index] = forces[sorted_index]
        return reordered


class TestAbacusCalculator(unittest.TestCase):

    here = Path(__file__).parent
    pporb = here.parent.parent.parent / 'tests' / 'PP_ORB'

    def _make_socketio(self, **kwargs):
        from unittest.mock import patch

        with patch(__name__ + '.Abacus') as abacus:
            calc = AbacusSocketIO(**kwargs)
            socket_inp = abacus.call_args.kwargs['inp']
        self.addCleanup(calc.close)
        return calc, socket_inp

    def test_socketio_variable_cell_does_not_force_properties(self):
        try:
            inp = AbacusSocketIO._socket_inp({}, variable_cell=True)
        except TypeError as error:
            self.fail('variable-cell socket input API is missing: {}'.format(
                error))
        self.assertEqual(inp['socket_variable_cell'], 1)
        self.assertNotIn('cal_force', inp)
        self.assertNotIn('cal_stress', inp)

    def test_socketio_rejects_variable_cell_input_conflict(self):
        try:
            for input_value, variable_cell in ((0, True), (1, False)):
                with self.subTest(input_value=input_value,
                                  variable_cell=variable_cell):
                    with self.assertRaisesRegex(
                            ValueError, 'socket_variable_cell'):
                        AbacusSocketIO._socket_inp(
                            {'socket_variable_cell': input_value},
                            variable_cell=variable_cell)
        except TypeError as error:
            self.fail('variable-cell socket input API is missing: {}'.format(
                error))

    def test_socketio_boolean_spellings(self):
        parser = getattr(AbacusSocketIO, '_input_bool', None)
        self.assertIsNotNone(parser, 'socket Boolean parser is missing')

        for value in (True, 1, 'true', '1'):
            with self.subTest(value=value):
                self.assertIs(parser(value, 'variable_cell'), True)
        for value in (False, 0, 'false', '0'):
            with self.subTest(value=value):
                self.assertIs(parser(value, 'variable_cell'), False)

    def test_socketio_rejects_ambiguous_boolean_strings(self):
        parser = getattr(AbacusSocketIO, '_input_bool', None)
        self.assertIsNotNone(parser, 'socket Boolean parser is missing')

        for value in ('yes', 'no', 'on', 'off', ''):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'variable_cell'):
                    parser(value, 'variable_cell')

    def test_variable_cell_skips_fixed_cell_guard(self):
        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = True
        calc._reference_cell = None
        checker = getattr(calc, '_check_cell_change', None)
        self.assertIsNotNone(checker, 'cell-change mode guard is missing')

        first = Atoms('Si', cell=[5, 5, 5], pbc=True)
        second = first.copy()
        second.cell[0, 1] = 0.2
        checker(first)
        checker(second)

    def test_socketio_variable_cell_requires_full_periodic_cell(self):
        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = True
        checker = getattr(calc, '_check_variable_cell_geometry', None)
        self.assertIsNotNone(checker)

        valid = Atoms('Si', cell=[5.0, 5.0, 5.0], pbc=True)
        checker(valid)
        for invalid in (
                Atoms('Si', cell=[5.0, 5.0, 5.0], pbc=[True, True, False]),
                Atoms('Si', cell=[5.0, 5.0, 0.0], pbc=True)):
            with self.assertRaisesRegex(ValueError, 'full rank-3 periodic'):
                checker(invalid)

    def test_socketio_default_properties_are_independently_available(self):
        calc, inp = self._make_socketio(
            variable_cell='false',
            inp={'socket_variable_cell': 'false'},
        )

        self.assertFalse(getattr(calc, 'variable_cell', False))
        self.assertNotIn('socket_variable_cell', inp)
        self.assertIn('forces', calc.implemented_properties)
        self.assertIn('stress', calc.implemented_properties)

    def test_socketio_fixed_real_stress_is_implemented(self):
        calc, _ = self._make_socketio(inp={'cal_stress': 1})

        self.assertIn('stress', calc.implemented_properties)

    def test_socketio_variable_cell_stress_is_implemented(self):
        calc, _ = self._make_socketio(variable_cell='true')

        self.assertTrue(getattr(calc, 'variable_cell', False))
        self.assertIn('stress', calc.implemented_properties)

    def test_socketio_variable_cell_accepts_disabled_force(self):
        calc, inp = self._make_socketio(
            variable_cell=True, inp={'cal_force': 0})
        self.assertEqual(inp['cal_force'], 0)
        self.assertFalse(calc._property_constraints['forces'])

    def test_socketio_variable_cell_accepts_disabled_stress(self):
        calc, inp = self._make_socketio(
            variable_cell=True, inp={'cal_stress': 0})
        self.assertEqual(inp['cal_stress'], 0)
        self.assertFalse(calc._property_constraints['stress'])

    def test_socketio_variable_cell_rejects_ambiguous_force(self):
        with self.assertRaisesRegex(ValueError, 'cal_force'):
            self._make_socketio(
                variable_cell=True, inp={'cal_force': 'maybe'})

    def test_socketio_variable_cell_rejects_ambiguous_stress(self):
        with self.assertRaisesRegex(ValueError, 'cal_stress'):
            self._make_socketio(
                variable_cell=True, inp={'cal_stress': 'maybe'})

    def test_socketio_metadata_marks_padded_properties_absent(self):
        metadata = json.dumps({
            'schema': 'abacus.socket.properties.v1',
            'present': ['energy'],
            'scf_converged': True,
        }).encode('utf-8')

        class FakeServer:
            @staticmethod
            def calculate(atoms):
                return {
                    'energy': -1.0,
                    'forces': np.zeros((len(atoms), 3)),
                    'virial': np.zeros((3, 3)),
                    'morebytes': np.frombuffer(metadata, dtype=np.int8),
                }

        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = False
        calc.implemented_properties = [
            'energy', 'free_energy', 'forces', 'stress']
        calc._property_constraints = {}
        calc._active_properties = None
        calc._reference_cell = None
        calc.atoms = None
        calc.server = FakeServer()
        calc.results = {}

        atoms = Atoms('Si', cell=[5, 5, 5], pbc=True)
        calc.calculate(atoms, properties=['energy'], system_changes=[])

        self.assertIn('energy', calc.results)
        self.assertNotIn('forces', calc.results)
        self.assertNotIn('stress', calc.results)
        self.assertTrue(calc.last_scf_converged)

    def test_socketio_rejects_malformed_metadata_property(self):
        malformed = json.dumps({
            'schema': 'abacus.socket.properties.v1',
            'present': [['forces']],
            'scf_converged': True,
        }).encode('utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown property'):
            AbacusSocketIO._decode_socket_metadata(
                np.frombuffer(malformed, dtype=np.int8))

    def test_socketio_results_replace_stale_force(self):
        force_metadata = json.dumps({
            'schema': 'abacus.socket.properties.v1',
            'present': ['energy', 'forces'],
            'scf_converged': True,
        }).encode('utf-8')
        energy_metadata = json.dumps({
            'schema': 'abacus.socket.properties.v1',
            'present': ['energy'],
            'scf_converged': True,
        }).encode('utf-8')

        class FakeServer:
            def __init__(self):
                self.calls = 0

            def calculate(self, atoms):
                self.calls += 1
                result = {
                    'energy': -float(self.calls),
                    'forces': np.ones((len(atoms), 3)),
                    'virial': np.zeros((3, 3)),
                }
                if self.calls == 1:
                    result['morebytes'] = np.frombuffer(
                        force_metadata, dtype=np.int8)
                else:
                    result['morebytes'] = np.frombuffer(
                        energy_metadata, dtype=np.int8)
                return result

        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = False
        calc.implemented_properties = [
            'energy', 'free_energy', 'forces', 'stress']
        calc._property_constraints = {}
        calc._active_properties = None
        calc._reference_cell = None
        calc.atoms = None
        calc.server = FakeServer()
        calc.results = {}

        atoms = Atoms('Si')
        calc.calculate(atoms, properties=['forces'], system_changes=[])
        self.assertIn('forces', calc.results)
        calc.calculate(atoms, properties=['energy'], system_changes=[])
        self.assertNotIn('forces', calc.results)

    def test_socketio_fixed_mode_discards_legacy_zero_virial(self):
        class FakeServer:
            @staticmethod
            def calculate(atoms):
                return {
                    'energy': -1.0,
                    'forces': np.zeros((len(atoms), 3)),
                    'virial': np.zeros((3, 3)),
                }

        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = False
        calc.implemented_properties = ['energy', 'free_energy', 'forces']
        calc._reference_cell = None
        calc.atoms = None
        calc.server = FakeServer()
        calc.results = {}

        atoms = Atoms('Si', cell=[5, 5, 5], pbc=True)
        calc.calculate(atoms, properties=['energy'], system_changes=[])

        self.assertNotIn('stress', calc.results)

    def test_socketio_real_stress_converts_virial(self):
        class FakeServer:
            @staticmethod
            def calculate(atoms):
                return {
                    'energy': -1.0,
                    'forces': np.zeros((len(atoms), 3)),
                    'virial': 125.0 * np.eye(3),
                }

        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = True
        calc.implemented_properties = [
            'energy', 'free_energy', 'forces', 'stress']
        calc._reference_cell = None
        calc.atoms = None
        calc.server = FakeServer()
        calc.results = {}

        atoms = Atoms('Si', cell=[5, 5, 5], pbc=True)
        calc.calculate(atoms, properties=['stress'], system_changes=[])

        np.testing.assert_allclose(
            calc.results['stress'], [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0])

    def test_socketio_species_order_mapping(self):
        atoms = Atoms(symbols=['Si', 'O', 'C', 'Si', 'O', 'C'])
        order = AbacusSocketIO._socket_sort_indices(atoms)
        self.assertEqual(order, [0, 3, 1, 4, 2, 5])

        socket_forces = np.arange(18).reshape(6, 3)
        input_forces = AbacusSocketIO._forces_to_input_order(
            socket_forces, order)

        expected = np.empty_like(socket_forces)
        for sorted_index, original_index in enumerate(order):
            expected[original_index] = socket_forces[sorted_index]
        np.testing.assert_array_equal(input_forces, expected)

    def test_socketio_rejects_parameter_changes(self):
        calc = object.__new__(AbacusSocketIO)
        with self.assertRaisesRegex(ValueError, 'fixed after construction'):
            calc.set(kpts={'mode': 'mp-sampling', 'nk': [2, 2, 2]})

    def test_socketio_rejects_cell_changes(self):
        from ase.calculators.calculator import PropertyNotImplementedError

        calc = object.__new__(AbacusSocketIO)
        calc.variable_cell = False
        calc.atoms = Atoms('Si', cell=[5.0, 5.0, 5.0], pbc=True)
        calc._reference_cell = calc.atoms.cell.array.copy()
        checker = getattr(calc, '_check_cell_change', None)
        self.assertIsNotNone(checker, 'cell-change mode guard is missing')

        changed = calc.atoms.copy()
        changed.cell[0, 0] = 5.1
        with self.assertRaisesRegex(PropertyNotImplementedError, 'fixed-cell'):
            checker(changed)

    def test_parse_version_allows_launcher_noise(self):
        stdout = 'launcher info\nABACUS version v3.11.0-beta6\n'
        self.assertEqual(AbacusProfile.parse_version(stdout), 'v3.11.0-beta6')

    def test_parse_version_rejects_missing_version(self):
        with self.assertRaisesRegex(RuntimeError, 'ABACUS version'):
            AbacusProfile.parse_version('launcher failed before abacus started')

    def test_calculator_results(self):
        from ase.build.bulk import bulk
        silicon = bulk('Si', crystalstructure='diamond', a=5.43)
        aprof = AbacusProfile(
            command='mpirun -np 2 abacus',
            pseudo_dir=self.pporb,
            orbital_dir=self.pporb,
            omp_num_threads=1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            calculator = Abacus(aprof,
                                directory=tmpdir,
                                pseudopotentials={'Si': 'Si_ONCV_PBE-1.0.upf'},
                                basissets={'Si': 'Si_gga_6au_100Ry_2s2p1d.orb'},
                                inp={'calculation': 'scf',
                                    'basis_type': 'lcao',
                                    'ks_solver': 'genelpa',
                                    'ecutwfc': 40,
                                    'symmetry': 1,
                                    'nspin': 1,
                                    'gamma_only': True,
                                    'cal_force': 1,
                                    'cal_stress': 1})
            silicon.calc = calculator
            e = silicon.get_potential_energy()
        
        # check!
        self.assertAlmostEqual(e, -194.953053309)
        self.assertIsNotNone(calculator.results)
        self.assertIsInstance(calculator.results, dict)
        for k in ['nspins', 'nkpts', 'nbands', 'eigenvalues', 'occupations',
                  'fermi_level', 'kpoint_weights', 'ibz_kpoints', 'energy', 
                  'free_energy', 'natoms', 'forces', 'stress', 'magmoms']:
            self.assertIn(k, calculator.results)
        self.assertEqual(calculator.results['nspins'], 1)
        self.assertEqual(calculator.results['nkpts'], 1)
        self.assertEqual(calculator.results['nbands'], 14)
        self.assertEqual(calculator.results['energy'], e)
        self.assertEqual(calculator.results['free_energy'], e)
        self.assertEqual(calculator.results['natoms'], 2)
        
        for k in ['eigenvalues', 'occupations', 'ibz_kpoints', 'forces', 'stress', 'magmoms']:
            self.assertIsInstance(calculator.results[k], np.ndarray)

        self.assertEqual(calculator.results['eigenvalues'].shape, (1, 1, 14))
        ekb = [-4.82194,  7.62727,  7.62727,  7.62737, 10.2436 , 10.2436 ,
                10.2436 , 10.9884 , 16.057  , 16.057  , 23.8353 , 25.421  ,
                25.421  , 25.4212 ]
        self.assertTrue(np.allclose(calculator.results['eigenvalues'][0, 0, :], np.array(ekb)))

        self.assertEqual(calculator.results['occupations'].shape, (1, 1, 14))
        occ = [2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(calculator.results['occupations'][0, 0, :], np.array(occ)))
        
        self.assertEqual(calculator.results['ibz_kpoints'].shape, (1, 3))
        self.assertTrue(np.allclose(calculator.results['ibz_kpoints'][0, :], np.array([0,0,0])))

        self.assertEqual(calculator.results['forces'].shape, (2, 3))
        self.assertTrue(np.allclose(calculator.results['forces'], np.zeros((2, 3))))

        self.assertEqual(calculator.results['stress'].shape, (6,))
        stress = [-0.19327923, -0.19327923, -0.19327923, -0.        ,  0.        ,   0.        ]
        self.assertTrue(np.allclose(calculator.results['stress'], np.array(stress)))
        
        self.assertEqual(calculator.results['magmoms'].shape, (2,))
        self.assertTrue(np.allclose(calculator.results['magmoms'], np.zeros(2)))

    def test_restart(self):
        from ase.build.bulk import bulk
        silicon = bulk('Si', crystalstructure='diamond', a=5.43)
        aprof = AbacusProfile(
            command='mpirun -np 2 abacus',
            pseudo_dir=self.pporb,
            orbital_dir=self.pporb,
            omp_num_threads=1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            calculator = Abacus(aprof,
                                directory=tmpdir,
                                pseudopotentials={'Si': 'Si_ONCV_PBE-1.0.upf'},
                                basissets={'Si': 'Si_gga_6au_100Ry_2s2p1d.orb'},
                                inp={'calculation': 'scf',
                                    'basis_type': 'lcao',
                                    'ks_solver': 'genelpa',
                                    'ecutwfc': 40,
                                    'symmetry': 1,
                                    'nspin': 1,
                                    'gamma_only': True,
                                    'cal_force': 1,
                                    'cal_stress': 1})
            silicon.calc = calculator
            e = silicon.get_potential_energy()
        
            # restart
            silicon.calc = Abacus.restart(aprof, directory=tmpdir)
            e2 = silicon.get_potential_energy()
            self.assertAlmostEqual(e2, e)

    def test_version_number_check(self):
        # not a version number
        with self.assertRaises(AssertionError):
            switch_io_backend_version('not-a-version-number')
        # too old version
        with self.assertRaises(AssertionError):
            switch_io_backend_version('v2.2.2')
        self.assertTrue(switch_io_backend_version('v3.8.4'))
        self.assertFalse(switch_io_backend_version('v3.9.0.25'))
        self.assertTrue(switch_io_backend_version('v3.10.0'))
        self.assertFalse(switch_io_backend_version('v3.11.0-beta.2'))
        self.assertFalse(switch_io_backend_version('v3.11.0'))

    def test_property_keywords_reject_conflicting_user_parameters(self):
        template = AbacusTemplate()
        with self.assertRaises(ValueError):
            template.get_property_keywords({'nspin': 1}, ['magmom'])

        parameters = template.get_property_keywords({'nspin': 2}, ['magmom'])
        self.assertEqual(str(parameters['nspin']), '2')

    def test_property_keywords_accept_equivalent_boolean_user_parameters(self):
        template = AbacusTemplate()

        parameters = template.get_property_keywords(
            {'cal_force': True, 'cal_stress': True},
            ['forces', 'stress']
        )

        self.assertEqual(str(parameters['cal_force']), '1')
        self.assertEqual(str(parameters['cal_stress']), '1')

    def test_property_keywords_reject_conflicting_boolean_user_parameters(self):
        template = AbacusTemplate()

        with self.assertRaises(ValueError):
            template.get_property_keywords({'cal_force': False}, ['forces'])

        with self.assertRaises(ValueError):
            template.get_property_keywords({'cal_stress': False}, ['stress'])

    def test_property_keywords_treat_string_values_as_scalars(self):
        template = AbacusTemplate()
        template.implemented_properties = ['probe']
        template.get_probe_keywords = lambda parameters: {'custom_switch': 'true'}

        parameters = template.get_property_keywords(
            {'custom_switch': 'true'}, ['probe']
        )

        self.assertEqual(parameters['custom_switch'], 'true')

    def test_property_keywords_compare_iterables_like_input_writer(self):
        template = AbacusTemplate()
        template.implemented_properties = ['probe']
        template.get_probe_keywords = lambda parameters: {
            'custom_vector': [1, 'true']
        }

        with self.assertRaises(ValueError):
            template.get_property_keywords(
                {'custom_vector': [True, 'true']}, ['probe']
            )

    def test_property_keywords_reject_conflicting_properties(self):
        template = AbacusTemplate()
        template.implemented_properties = ['prop_a', 'prop_b']
        template.get_prop_a_keywords = lambda parameters: {'calculation': 'scf'}
        template.get_prop_b_keywords = lambda parameters: {'calculation': 'md'}

        with self.assertRaises(ValueError):
            template.get_property_keywords({}, ['prop_a', 'prop_b'])

    def test_dipole_property_is_not_implemented(self):
        template = AbacusTemplate()
        self.assertNotIn('dipole', template.implemented_properties)
        with self.assertRaises(AssertionError):
            template.get_property_keywords({}, ['dipole'])

if __name__ == '__main__':
    unittest.main()
