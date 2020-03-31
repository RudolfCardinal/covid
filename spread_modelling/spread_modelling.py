#!/usr/bin/env python

"""
spread_modelling.py

===============================================================================

    Copyright (C) 2020-2020 Rudolf Cardinal (rudolf@pobox.com).

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.

===============================================================================

Models the spread of an infection such as SARS-CoV-2 coronavirus (COVID-19
disease) under different models of community mental health care.

Time units are days.

Re transmission risk (infectivity):

- The basic reproduction number R0 is the expected number of secondary cases
  produced by a single infection in a completely susceptible population. It is
  OF THE ORDER of 2.4 for SARS-CoV-2 (the virus) or COVID-19 (the disease).
  
  - https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
    ... using 2.4 (but examing range 2.0-2.6)

  - https://www.ncbi.nlm.nih.gov/pubmed/32097725
    ... 2.28

  - https://academic.oup.com/jtm/article/27/2/taaa021/5735319
    ... mean 3.28, median 2.79

  - https://www.nature.com/articles/s41421-020-0148-0
   ... using estimates in the range 1.9, 2.6, 3.1

- Then R0 is equal to:

  τ × c_bar × d

  where:

  .. code-block:: none

    τ     = transmissibility
          = p(infection | contact between a susceptible and an infected
                          individual)

    c_bar = average rate of contact between susceptible and infected
             (contacts per unit time)

    d     = duration of infectiousness

  ... see https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf

"""  # noqa

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import cProfile
import csv
from enum import Enum
from itertools import combinations, product
import logging
import multiprocessing
import os
import random
from typing import Any, Callable, Dict, TextIO, Iterable, List, Optional

from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from cardinal_pythonlib.randomness import coin as cpl_coin
import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

FuncType = Callable[[Any], Any]
NUMPY_COIN = False  # True (numpy) is slower
THIS_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_BEHAV_INFECTIVITY_MULTIPLE_IF_SYMPTOMATIC = 0.1
# DEFAULT_OUTPUT_DIR = os.path.expanduser("~/tmp/cpft_covid_modelling")
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "results")

N_ITERATIONS = 2000


class Appointment(Enum):
    """
    The way in which clinical appointments are conducted.
    """
    HOME_VISIT = "home_visit"
    CLINIC = "clinic"
    REMOTE = "remote"

    def __str__(self):
        return self.value


class RunMode(Enum):
    """
    Mode of running the program.
    """
    DEBUG_SINGLE = "debug_single"
    DEBUG_PROFILE = "debug_profile"
    DEBUG_CHECK_N_ITERATIONS = "debug_check_n_iterations"
    FULL = "full"


class InfectionStatus(Enum):
    """
    S-E-I-R model.
    """
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


# =============================================================================
# Helper functions
# =============================================================================

def numpy_coin(p: float) -> bool:
    """
    Flip a coin with a certain probability, via Numpy.
    """
    return bool(np.random.binomial(n=1, p=p))


# Choose our coin-flip function
if NUMPY_COIN:
    coin = numpy_coin
else:
    coin = cpl_coin


def do_cprofile(func: FuncType) -> FuncType:
    """
    Print profile stats to screen. To be used as a decorator for the function
    or method you want to profile.
    """
    def profiled_func(*args, **kwargs) -> Any:
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort="cumulative")
    return profiled_func


def product_dict(**kwargs: Iterable) -> Iterable[Dict]:
    """
    See
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists.

    Takes keyword arguments, and yields dictionaries containing every
    combination of possibilities for each keyword.
    
    Examples:
    
    .. code-block:: python
    
        >>> list(product_dict(a=[1, 2], b=[3, 4]))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]

        >>> list(product_dict(a="x", b=range(3)))
        [{'a': 'x', 'b': 0}, {'a': 'x', 'b': 1}, {'a': 'x', 'b': 2}]

        >>> product_dict(a="x", b=range(3))
        <generator object product_dict at 0x7fb328070678>
    """  # noqa
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


# =============================================================================
# Config
# =============================================================================

class Config(object):
    """
    Represents a configuration for a single "run".
    """
    def __init__(
            self,
            # Patient: biology and some social aspects
            pathogenicity_p_symptoms_if_infected: float = 0.67,
            biological_asymptomatic_infectivity_factor: float = 0.5,
            t_infected_to_symptoms_incubation_period: float = 5.1,
            t_infected_to_infectious: float = 4.6,
            infectious_for_t: float = 7.0,
            symptoms_for_t: float = 7.0,
            p_infect_for_full_day_exposure: float = 0.2,
            behavioural_infectivity_multiple_if_symptomatic: float =
            DEFAULT_BEHAV_INFECTIVITY_MULTIPLE_IF_SYMPTOMATIC,  # noqa
            # Population: parameters governing social interaction etc.
            appointment_type: Appointment = Appointment.CLINIC,
            clinicians_meet_each_other: bool = False,
            n_clinicians: int = 20,
            n_patients_per_clinician_per_day: int = 5,
            variable_n_family: bool = True,  # needless reduction in power
            mean_n_family_members_per_patient: float = 1.37,
            n_days: int = 60,
            p_baseline_infected: float = 0.05,
            p_external_infection_per_day: float = 0.0,
            prop_day_clinician_patient_interaction: float = 1 / 24,
            prop_day_family_interaction: float = 8 / 24,
            prop_day_clinician_clinician_interaction: float = 1 / 24,
            prop_day_clinician_family_intervention: float = 0.2 / 24,
            n_random_social_contacts_per_day: int = 0,
            prop_day_random_social_contact: float = 0.1 / 24,
            # Simulation
            iteration: int = 0) -> None:
        """
        Args:
            pathogenicity_p_symptoms_if_infected:
                (PATIENT.)
                Probability of developing symptoms if infected.
                Figure of 0.67 is based on influenza [2].
            biological_asymptomatic_infectivity_factor:
                (PATIENT.)
                Infectivity is multiplied by this amount if asymptomatic.
                Default is 0.5 [1, 2]: people are half as infectious if
                asymptomatic.
            t_infected_to_symptoms_incubation_period:
                (PATIENT.)
                Incubation period; 5.1 days [1].
            t_infected_to_infectious:
                (PATIENT.)
                For SARS-CoV-2 (the virus) or COVID-19 (the disease), symptoms
                after 5.1 days (incubation period) but infectious after 4.6
                [1].
            infectious_for_t:
                (PATIENT.)
                Duration of infectivity. This is extremely crude.
                Stepwise function in which people are infectious at a constant
                rate for this many days. IMPRECISE; 7 is the number of days of
                self-isolation recommended by the UK government after symptoms,
                2020-03-24.
            symptoms_for_t:
                (PATIENT.)
                Time for which symptoms persist, once developed.
                Also IMPRECISE. See below re why this shouldn't matter much.
            p_infect_for_full_day_exposure:
                (PATIENT.)
                Probability, when infectious, of infecting someone if exposed
                to them for 24 hours per day.
                IMPRECISE. As long as it's constant, it's not so important --
                we are interested in relative change due to social
                manipulations, rather than absolute rates.
            behavioural_infectivity_multiple_if_symptomatic:
                (PATIENT.)
                If this patient is symptomatic, how should infectivity be
                multiplied *behaviourally*? This is expected to be <1
                (infection control measures can be taken for symptomatic
                people, e.g. wearing a mask) and is distinct from the
                biological change (see above).

            appointment_type:
                Type of appointment (see text).
            clinicians_meet_each_other:
                Do clinicians in this team meet each other daily?
            n_clinicians:
                Number of clinicians in the team.
            n_patients_per_clinician_per_day:
                Number of (different) patients seen by each clinician each day,
                assumed to be all new patients each day.
            variable_n_family:
                Allow the number of family members to vary?
                Warning: increases variability, so loses power.
                But adds realism.
            mean_n_family_members_per_patient:
                Mean number of other family members per patient (Poisson
                lambda). If ``variable_n_family`` is ``False``, this number is
                rounded to give the (fixed) number of other family members per
                patient.
                The mean UK household size in 2019 was 2.37 (Office for
                National Statistics), so 1.37 people other than the patient.
            n_days:
                Number of days to simulate.
            p_baseline_infected:
                Probability of being infected on day 0.
            p_external_infection_per_day:
                Probability that any person is infected each day, independent
                of any within-group interactions.
            prop_day_clinician_patient_interaction:
                Proportion of a day for a clinician-patient encounter.
            prop_day_family_interaction:
                Proportion of a day for a patient-family member encounter.
            prop_day_clinician_clinician_interaction:
                Proportion of a day for a clinician-clinician encounter.
            prop_day_clinician_family_intervention:
                Proportion of a day for a clinician-family member encounter
                (on home visits only).
            n_random_social_contacts_per_day:
                number of random contacts per day, for each person, with
                another member of the group
            prop_day_random_social_contact:
                Proportion of a day for a random interaction.
                
            iteration:
                Iteration of simulation.

        Refs:

        [1] Ferguson, Neil M., Daniel Laydon, Gemma Nedjati-Gilani, Natsuko
            Imai, Kylie Ainslie, Marc Baguelin, Sangeeta Bhatia, et al. “Report
            9: Impact of Non-Pharmaceutical Interventions (NPIs) to Reduce
            COVID-19 Mortality and Healthcare Demand.” COVID-19 Reports.
            Imperial College London, March 16, 2020.
            https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf.

        [2] Halloran, M. Elizabeth, Neil M. Ferguson, Stephen Eubank, Ira M.
            Longini, Derek A. T. Cummings, Bryan Lewis, Shufu Xu, et al.
            “Modeling Targeted Layered Containment of an Influenza Pandemic in
            the United States.” Proceedings of the National Academy of Sciences
            of the United States of America 105, no. 12 (March 25, 2008):
            4639–44. https://doi.org/10.1073/pnas.0706849105.

        """  # noqa

        assert behavioural_infectivity_multiple_if_symptomatic >= 0.0

        self.pathogenicity_p_symptoms_if_infected = \
            pathogenicity_p_symptoms_if_infected
        self.biological_asymptomatic_infectivity_factor = \
            biological_asymptomatic_infectivity_factor
        self.t_infected_to_symptoms_incubation_period = \
            t_infected_to_symptoms_incubation_period
        self.t_infected_to_infectious = t_infected_to_infectious
        self.infectious_for_t = infectious_for_t
        self.symptoms_for_t = symptoms_for_t
        self.p_infect_for_full_day_exposure = p_infect_for_full_day_exposure
        self.behavioural_infectivity_multiple_if_symptomatic = \
            behavioural_infectivity_multiple_if_symptomatic

        self.appointment_type = appointment_type
        self.clinicians_meet_each_other = clinicians_meet_each_other
        self.n_clinicians = n_clinicians
        self.n_patients_per_clinician_per_day = \
            n_patients_per_clinician_per_day
        self.variable_n_family = variable_n_family
        self.mean_n_family_members_per_patient = \
            mean_n_family_members_per_patient
        self.n_days = n_days
        self.p_baseline_infected = p_baseline_infected
        self.p_external_infection_per_day = p_external_infection_per_day
        self.prop_day_clinician_patient_interaction = \
            prop_day_clinician_patient_interaction
        self.prop_day_family_interaction = prop_day_family_interaction
        self.prop_day_clinician_clinician_interaction = \
            prop_day_clinician_clinician_interaction
        self.prop_day_clinician_family_intervention = \
            prop_day_clinician_family_intervention
        self.n_random_social_contacts_per_day = \
            n_random_social_contacts_per_day
        self.prop_day_random_social_contact = prop_day_random_social_contact

        self.iteration = iteration


# =============================================================================
# Metaconfig
# =============================================================================

class Metaconfig(object):
    """
    Creates multiple :class:`Config` objects for systematic variation of
    parameters.
    """
    def __init__(self,
                 n_iterations: int = 50,
                 **kwargs: Iterable) -> None:
        """
        Parameters are iterables (e.g. lists) of possible values for the
        equivalently named parameters in :class:`Config`, except for
        ``n_iterations``, the number of runs to perform per condition.
        """
        self.n_iterations = n_iterations
        self.kwargs = kwargs

    def gen_configs(self) -> Iterable[Config]:
        """
        Generates all possible configs.
        """
        gen_combinations = product_dict(
            iteration=range(1, self.n_iterations + 1),
            **self.kwargs
        )
        for config_kwargs in gen_combinations:
            yield Config(**config_kwargs)

    def varnames(self) -> List[str]:
        """
        Names of variables being manipulated.
        """
        return list(self.kwargs.keys())

    def announce_sim(self, config: Config) -> None:
        """
        Announces a simulation, with relevant variables.
        """
        paramlist = [f"{var}={getattr(config, var)}"
                     for var in self.varnames()]
        paramlist.append(f"iteration={config.iteration}")
        params = ", ".join(paramlist)
        log.info(f"Simulating: {params}...")

    def csv_header_row(self) -> List[str]:
        """
        Returns a CSV header for the parameters manipulated by this
        metaconfig.
        """
        return self.varnames() + ["iteration"]

    def csv_data_row(self, config: Config) -> List[Any]:
        """
        Returns a CSV data row for a specific config generated by this
        metaconfig.
        """
        return (
            [getattr(config, varname) for varname in self.varnames()] +
            [config.iteration]
        )

    def simulate_one(self, config: Config) -> "Population":
        """
        Performs a single run.
        """
        pop = Population(config=config)
        self.announce_sim(config)
        pop.simulate()
        return pop

    def simulate_all(self, totals_file: TextIO, daily_file: TextIO) -> None:
        """
        Simulate everything and save the results.

        Args:
            totals_file:
                output file for whole-run totals
            daily_file:
                output file for day-by-day totals
        """
        writer_totals = csv.writer(totals_file)
        writer_people = csv.writer(daily_file)

        control_header_row = self.csv_header_row()
        writer_totals.writerow(control_header_row +
                               Population.totals_csv_header_row())
        writer_people.writerow(control_header_row +
                               Population.daily_csv_header_row())

        # - Threading doesn't help much (probably because of the Python global
        #   interpreter lock). Use processes; much quicker.
        #
        # - Models of use include:
        #
        #       with ProcessPoolExecutor(...) as executor:
        #           results = executor.map(function, iterable)
        #       # ... will complete all? And then
        #       for result in results:
        #           pass
        #
        #       with ProcessPoolExecutor(...) as executor:
        #           for result in executor.map(function, iterable):
        #               # ... also waits for many to complete before we get
        #               #     here -- but not all? Seems to operate batch-wise.
        #               pass

        n_cpus = multiprocessing.cpu_count()
        log.info(f"Using {n_cpus} processes")
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            for pop in executor.map(self.simulate_one, self.gen_configs()):
                control_data_row = self.csv_data_row(pop.config)
                writer_totals.writerow(control_data_row +
                                       pop.totals_csv_data_row())
                for r in pop.daily_csv_data_rows():
                    writer_people.writerow(control_data_row + r)


# =============================================================================
# Person
# =============================================================================

class Person(object):
    """
    Represents a person.
    """
    def __init__(self, name: str, config: Config):
        """
        Args:
            name:
                Identifying name.
            config:
                :class:`Config` object.

        """  # noqa
        self.name = name

        # Copy for speed (saves self.config.X indirection each time):
        self.biological_asymptomatic_infectivity_factor = \
            config.biological_asymptomatic_infectivity_factor
        self.infectious_for_t = config.infectious_for_t
        self.p_infect_for_full_day_exposure = \
            config.p_infect_for_full_day_exposure
        self.behavioural_infectivity_multiple_if_symptomatic = \
            config.behavioural_infectivity_multiple_if_symptomatic
        self.symptoms_for_t = config.symptoms_for_t
        self.t_infected_to_infectious = config.t_infected_to_infectious
        self.t_infected_to_symptoms_incubation_period = \
            config.t_infected_to_symptoms_incubation_period

        self._symptomatic_if_infected = coin(
            config.pathogenicity_p_symptoms_if_infected)

        self.infected = False  # infected at some point?
        self.infected_at = None  # type: Optional[float]

    def __str__(self) -> str:
        """
        Time-independent string representation.
        """
        if self.infected:
            status = f"infected_at_{self.infected_at}"
        else:
            status = "susceptible"
        return f"{self.name}<{status}>"

    def seir_status(self, now: float) -> InfectionStatus:
        """
        SEIR status.
        """
        if self.susceptible():
            return InfectionStatus.SUSCEPTIBLE
        t_since_infected = self.time_since_infected(now)
        if t_since_infected < self.t_infected_to_infectious:
            return InfectionStatus.EXPOSED
        if t_since_infected < (self.t_infected_to_infectious +
                               self.infectious_for_t):
            return InfectionStatus.INFECTIOUS
        return InfectionStatus.RECOVERED

    def status_at(self, now: float) -> str:
        """
        Time-specific string representation.
        """
        return self.seir_status(now).value

    def susceptible(self) -> bool:
        """
        Is this person susceptible to infection?
        """
        return not self.infected

    def time_since_infected(self, now: float) -> Optional[float]:
        """
        Time since infection occurred, or ``None`` if never infected.

        Args:
            now: time now
        """
        if not self.infected:
            return None
        return now - self.infected_at

    def time_since_infectious(self, now: float) -> Optional[float]:
        """
        Time since the patient became infectious to others, or ``None`` if
        never infected.

        Args:
            now: time now
        """
        t_since_infected = self.time_since_infected(now)
        if t_since_infected is None:
            return None
        if t_since_infected < self.t_infected_to_infectious:
            return None
        return t_since_infected - self.t_infected_to_infectious

    def symptomatic(self, now: float) -> bool:
        """
        Is the patient currently symptomatic?

        Args:
            now: time now
        """
        if not self._symptomatic_if_infected:
            return False
        t_since_infected = self.time_since_infected(now)
        if t_since_infected is None:
            return False
        if t_since_infected < self.t_infected_to_symptoms_incubation_period:
            return False
        return (
            t_since_infected - self.t_infected_to_symptoms_incubation_period <
            self.symptoms_for_t
        )

    def infectivity(self, now: float, prop_full_day: float) -> float:
        """
        Probability that another person will be infected by ``self``, when
        exposed to a fraction of a full day determined by ``prop_full_day``.

        Args:
            now:
                The time now.
            prop_full_day:
                The fraction of a full day for which exposure occurred (on
                the basis that longer exposure increases risk).

        """
        # Very crude model:
        t_since_infectious = self.time_since_infectious(now)
        if t_since_infectious is None:
            return 0.0
        if t_since_infectious >= self.infectious_for_t:
            # Past the infectious period
            return 0.0

        p_infect = self.p_infect_for_full_day_exposure * prop_full_day
        if self.symptomatic(now):
            # Symptomatic; people may wear PPE etc.
            # log.debug(f"Symptomatic: *= {self.behavioural_infectivity_multiple_if_symptomatic}")  # noqa
            p_infect *= self.behavioural_infectivity_multiple_if_symptomatic
        else:
            # Asymptomatic; probably less infectious biologically
            # log.debug(f"Asymptomatic: *= {self.biological_asymptomatic_infectivity_factor}")  # noqa
            p_infect *= self.biological_asymptomatic_infectivity_factor
        return p_infect

    def infect(self, now: float, source: "Person" = None) -> None:
        """
        Infect this patient.

        Args:
            now: time now
            source: (for debugging) source of infection
        """
        if self.infected:
            return
        log.debug(f"{self.name} infected at {now} by {source}")
        self.infected = True
        self.infected_at = now

    def expose_to(self, other: "Person", now: float,
                  prop_full_day: float) -> None:
        """
        Exposes ``self`` to ``other``, which may result in infection.
        Repeat the other way round for a full interaction.

        Args:
            other:
                the potential "donor"
            now:
                The time now.
            prop_full_day:
                The fraction of a full day for which exposure occurred (on
                the basis that longer exposure increases risk).
        """
        if not other.infected:
            return  # much quicker to skip in this situation
        infectivity = other.infectivity(now, prop_full_day)
        if infectivity <= 0.0:
            return
        if coin(infectivity):
            self.infect(now, other)

    def mutually_expose(self, other: "Person", now: float,
                        prop_full_day: float) -> None:
        """
        Expose ``self`` to ``other``, and ``other`` to ``self``.

        Args:
            other:
                the potential "donor"
            now:
                The time now.
            prop_full_day:
                The fraction of a full day for which exposure occurred (on
                the basis that longer exposure increases risk).
        """
        self.expose_to(other, now, prop_full_day)
        other.expose_to(self, now, prop_full_day)


# =============================================================================
# Population
# =============================================================================

class Population(object):
    """
    Represents the population being studied, and their interactions.
    """
    def __init__(self, config: Config) -> None:
        """
        Args:
            config:
                :class:`Config` object.
        """
        self.config = config

        self.n_patients_per_clinician = \
            config.n_patients_per_clinician_per_day * config.n_days
        self.n_patients = config.n_clinicians * self.n_patients_per_clinician

        self.clinicians = [
            Person(name=f"Clinician_{i + 1}", config=config)
            for i in range(self.config.n_clinicians)
        ]
        self.family = []  # type: List[Person]
        self.patients = [
            Person(name=f"Patient_{i + 1}", config=config)
            for i in range(self.n_patients)
        ]
        # Map of patient to family members:
        self.p2f = {}  # type: Dict[Person, List[Person]]
        self._n_people = len(self.clinicians) + len(self.patients)

        for pnum, p in enumerate(self.patients, start=1):
            if config.variable_n_family:
                n_family = np.random.poisson(
                    lam=config.mean_n_family_members_per_patient)
            else:
                n_family = np.round(config.mean_n_family_members_per_patient)
            family_members = []  # type: List[Person]
            for i in range(n_family):
                f = Person(name=f"Family_{i + 1}_of_patient_{pnum}",
                           config=config)
                family_members.append(f)
                self.family.append(f)
            self.p2f[p] = family_members
            self._n_people += len(family_members)

        # Baseline infection
        for p in self.gen_people():
            if coin(self.config.p_baseline_infected):
                p.infect(0)

        # Logging. Keys are day number.
        # This is redundancy but important for speed.
        self.total_n_contacts = 0
        self.daily_contacts = defaultdict(int)  # type: Dict[int, int]
        self._n_people_infected_by = defaultdict(int)  # type: Dict[int, int]
        self._n_clinicians_infected_by = defaultdict(int)  # type: Dict[int, int]  # noqa
        self._n_patients_infected_by = defaultdict(int)  # type: Dict[int, int]
        self._n_family_infected_by = defaultdict(int)  # type: Dict[int, int]

    def gen_people(self) -> Iterable[Person]:
        """
        Yield all people in turn.
        """
        for p in self.clinicians:
            yield p
        for p in self.patients:
            yield p
        for p in self.family:
            yield p

    def person_at_idx(self, idx: int) -> Person:
        """
        Returns the person at a given zero-based index, across all types of
        people.
        """
        assert idx >= 0
        if idx < len(self.clinicians):
            return self.clinicians[idx]
        idx -= len(self.clinicians)

        assert idx >= 0
        if idx < len(self.patients):
            return self.patients[idx]
        idx -= len(self.patients)

        assert 0 <= idx < len(self.family)
        return self.family[idx]

    def patients_for_clinician(self, clinician: Person,
                               today: int) -> Iterable[Person]:
        """
        Yield patients to be seen today by a given clinician.
        """
        assert clinician in self.clinicians
        assert 1 <= today <= self.config.n_days
        today_zb = today - 1
        clinician_idx = self.clinicians.index(clinician)
        first_patient_idx = (
            clinician_idx * self.n_patients_per_clinician +
            today_zb * self.config.n_patients_per_clinician_per_day
        )
        for patient_idx in range(
                first_patient_idx,
                first_patient_idx +
                self.config.n_patients_per_clinician_per_day):
            yield self.patients[patient_idx]

    def simulate_day(self, today: int) -> None:
        """
        Simulate a specific day's interactions.
        """
        def expose(p1_: Person, p2_: Person, prop_day: float) -> None:
            """
            Expose two people to each other.
            """
            assert p1_ is not p2_
            self.total_n_contacts += 1
            self.daily_contacts[today] += 1
            p1_.mutually_expose(p2_, now, prop_day)

        log.debug(f"Simulating day {today}...")
        now = float(today)

        # Simulate infection from outside our group:
        p_external_infection = self.config.p_external_infection_per_day
        for p in self.gen_people():
            if p.susceptible() and coin(p_external_infection):
                p.infect(now)

        # Household interactions:
        prop_day_family = self.config.prop_day_family_interaction
        for p in self.patients:
            # Patients interact with their family:
            family = self.p2f[p]
            for f in family:
                expose(p, f, prop_day_family)
            # Family interact with each other
            for f1, f2 in combinations(family, r=2):  # type: Person, Person
                expose(f1, f2, prop_day_family)

        appointment_type = self.config.appointment_type
        if appointment_type in [Appointment.HOME_VISIT, Appointment.CLINIC]:
            # Clinicians meet their patients:
            prop_day_c_p = self.config.prop_day_clinician_patient_interaction
            for c in self.clinicians:
                for p in self.patients_for_clinician(c, today):
                    expose(c, p, prop_day_c_p)
                    # Clinicians might meet the family:
                    if appointment_type == Appointment.HOME_VISIT:
                        family = self.p2f[p]
                        for f in family:
                            expose(c, f, prop_day_c_p)
        elif appointment_type == Appointment.REMOTE:
            # Nobody meets up
            pass
        else:
            raise ValueError("Bad appointment type")

        # Clinicians may get together (e.g. share a base, meet up):
        if self.config.clinicians_meet_each_other:
            prop_day_c_c = self.config.prop_day_clinician_clinician_interaction
            for c1, c2 in combinations(self.clinicians, r=2):  # type: Person, Person  # noqa
                expose(c1, c2, prop_day_c_c)

        # Random contacts within the population?
        if self.config.n_random_social_contacts_per_day > 0:
            prop_day_random = self.config.prop_day_random_social_contact
            n_random_contacts = min(
                self.config.n_random_social_contacts_per_day,
                self.n_people()
            )
            log.debug(f"Random social contacts with "
                      f"{n_random_contacts} people each")
            for p1 in self.gen_people():
                p2idxlist = np.random.choice(
                    a=self.n_people(),  # will use range from 0 to this minus 1
                    size=n_random_contacts,
                    replace=False
                )
                for p2idx in p2idxlist:
                    p2 = self.person_at_idx(p2idx)
                    if p2 is p1:
                        continue
                    expose(p1, p2, prop_day_random)

        # Done. Tot up the day's totals for efficiency later.
        self._update_day_totals(today)
        log.debug(f"... day {today} had {self.daily_contacts[today]} contacts")

    def _update_day_totals(self, today: int) -> None:
        """
        Faster to calculate this day by day and with a single iteration.
        """
        self._n_patients_infected_by[today] = 0
        self._n_clinicians_infected_by[today] = 0
        self._n_patients_infected_by[today] = 0
        self._n_family_infected_by[today] = 0
        for p in self.clinicians:
            if p.infected:
                self._n_people_infected_by[today] += 1
                self._n_clinicians_infected_by[today] += 1
        for p in self.patients:
            if p.infected:
                self._n_people_infected_by[today] += 1
                self._n_patients_infected_by[today] += 1
        for p in self.family:
            if p.infected:
                self._n_people_infected_by[today] += 1
                self._n_family_infected_by[today] += 1

    def simulate(self) -> None:
        """
        Simulate all days.
        """
        for day in range(1, self.config.n_days + 1):
            self.simulate_day(day)

    def n_people_infected(self) -> int:
        """
        The number of people who have been infected at some point.
        """
        return sum(1 for p in self.gen_people() if p.infected)

    def n_people(self) -> int:
        """
        The total number of people.
        """
        return self._n_people

    def n_patients_infected(self) -> int:
        """
        The number of patients who have been infected at some point.
        """
        return sum(1 for p in self.patients if p.infected)

    def n_clinicians_infected(self) -> int:
        """
        The number of clinicians who have been infected at some point.
        """
        return sum(1 for c in self.clinicians if c.infected)

    def n_family(self) -> int:
        """
        The total number of family members of patients.
        """
        return len(self.family)

    def n_family_infected(self) -> int:
        """
        The number of family members who have been infected.
        """
        return sum(1 for f in self.family if f.infected)
    
    @staticmethod
    def totals_csv_header_row() -> List[str]:
        """
        CSV header row for the "totals" output.
        """
        return [
            "n_patients", "n_patients_infected",
            "n_clinicians", "n_clinicians_infected",
            "n_family", "n_family_infected",
            "n_people", "n_people_infected",
            "n_contacts",
        ]
    
    def totals_csv_data_row(self) -> List[Any]:
        """
        CSV data row for the "totals" output.
        """
        return [
            self.n_patients, self.n_patients_infected(),
            self.config.n_clinicians, self.n_clinicians_infected(),
            self.n_family(), self.n_family_infected(),
            self.n_people(), self.n_people_infected(),
            self.total_n_contacts,
        ]

    @staticmethod
    def daily_csv_header_row() -> List[str]:
        """
        CSV header row for the "people" output.
        """
        return [
            "day",
            "n_people_infected",
            "n_clinicians_infected",
            "n_patients_infected",
            "n_family_infected",
            "n_contacts",
        ]

    def daily_csv_data_rows(self) -> Iterable[List[Any]]:
        """
        CSV data row for the "people" output.
        """
        for day in range(1, self.config.n_days + 1):
            yield [
                day,
                self.n_people_infected_by(day),
                self.n_clinicians_infected_by(day),
                self.n_patients_infected_by(day),
                self.n_family_infected_by(day),
                self.n_contacts_by(day),
            ]

    def n_people_infected_by(self, day: int) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_people_infected_by[day]

    def n_clinicians_infected_by(self, day: int) -> int:
        """
        Number of clinicians infected by a particular point in time.
        """
        return self._n_clinicians_infected_by[day]

    def n_patients_infected_by(self, day: int) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_patients_infected_by[day]

    def n_family_infected_by(self, day: int) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_family_infected_by[day]

    def n_contacts_by(self, last_day: int, first_day: int = 1) -> int:
        """
        Number of contacts up to and including ``day``.
        """
        assert first_day <= last_day
        return sum(self.daily_contacts[d]
                   for d in range(first_day, last_day + 1))


# =============================================================================
# Simulation framework
# =============================================================================

def simulate_all(filename_totals: str,
                 filename_daily: str,
                 runmode: RunMode) -> None:
    """
    Simulate our interactions and save the results.

    Args:
        filename_totals:
            output filename for whole-run totals
        filename_daily:
            output filename for day-by-day details
        runmode:
            Method of running the program.
            number of iterations to run
    """
    if runmode in [RunMode.DEBUG_SINGLE, RunMode.DEBUG_PROFILE]:
        n_iterations = 1 if runmode == RunMode.DEBUG_SINGLE else 24
        mc = Metaconfig(
            n_iterations=n_iterations,
            appointment_type=[Appointment.HOME_VISIT],
            clinicians_meet_each_other=[True],
            behavioural_infectivity_multiple_if_symptomatic=[0.1],
            p_baseline_infected=[0.05],
            p_external_infection_per_day=[0.01],
        )
    elif runmode == RunMode.DEBUG_CHECK_N_ITERATIONS:
        mc = Metaconfig(
            n_iterations=N_ITERATIONS,
            appointment_type=[Appointment.HOME_VISIT],
            clinicians_meet_each_other=[True],
            behavioural_infectivity_multiple_if_symptomatic=[0.1],
            p_baseline_infected=[0.01],
            p_external_infection_per_day=[0.0],
        )
    elif runmode == RunMode.FULL:
        # ---------------------------------------------------------------------
        # MAIN EXPERIMENTAL SETTINGS
        # ---------------------------------------------------------------------
        mc = Metaconfig(
            n_iterations=N_ITERATIONS,
            appointment_type=[x for x in Appointment],
            # ... can do list(Appointment) but the type checker complains
            clinicians_meet_each_other=[True, False],
            behavioural_infectivity_multiple_if_symptomatic=[0.1, 1.0],
            p_baseline_infected=[0.01, 0.05],
            p_external_infection_per_day=[0.0, 0.02],
        )
    else:
        raise ValueError(f"Unknown runmode: {runmode!r}")

    log.info(f"Writing totals to: {filename_totals}")
    log.info(f"Writing daily results to: {filename_daily}")
    with open(filename_totals, "wt") as ft, open(filename_daily, "wt") as fd:
        mc.simulate_all(totals_file=ft, daily_file=fd)


# =============================================================================
# Self-tests
# =============================================================================

def selftest() -> None:
    """
    Self-tests.
    """
    log.warning("Testing random number generator")
    n = 1000
    for p in np.arange(0, 1.01, step=0.1):
        x = sum(coin(p) for _ in range(n))
        log.info(f"coin({p}) × {n} → {x} heads")

    log.warning("Testing infection model")
    cfg = Config()
    cfg.pathogenicity_p_symptoms_if_infected = 1.0
    p1 = Person("Test_person_1_symptomatic", cfg)
    cfg.pathogenicity_p_symptoms_if_infected = 0.0
    p2 = Person("Test_person2_asymptomatic", cfg)
    for p in [p1, p2]:
        log.info(f"Person: {p}")
        for now in range(-5, 30 + 1):
            if now == 0:
                p.infect(now)
            print(
                f"Day {now}: "
                f"infected = {p.infected}, "
                f"infected_at = {p.infected_at}, "
                f"symptomatic = {p.symptomatic(now)}, "
                f"infectivity for 24h = {p.infectivity(now, 1.0)}, "
                f"status = {p.status_at(now)}"
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Command-line entry point function.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--outfile_totals", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "disease_spread_totals.csv"),
        help="Seed for random number generator"
    )
    parser.add_argument(
        "--outfile_daily", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "disease_spread_daily.csv"),
        help="Seed for random number generator"
    )
    parser.add_argument(
        "--runmode", type=str, default=RunMode.FULL.value,
        choices=[x.value for x in RunMode],
        help="Method of running the program"
    )
    parser.add_argument(
        "--seed", default=1234, type=str,
        help="Seed for random number generator ('None' gives a random seed)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Be verbose"
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Perform self-tests and quit"
    )
    args = parser.parse_args()

    # Logging
    main_only_quicksetup_rootlogger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        with_process_id=True
    )

    # Self-tests
    if args.selftest:
        selftest()
        return

    # RNG seed
    try:
        seed = int(args.seed)
    except TypeError:  # args.seed likely None
        seed = None
    except ValueError:  # a string
        if args.seed.lower() == "none":
            seed = None
        else:
            raise
    log.info(f"Using random number generator seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # Run mode
    runmode = RunMode(args.runmode)

    # Profile?
    if runmode == RunMode.DEBUG_PROFILE:
        runfunc = do_cprofile(simulate_all)
    else:
        runfunc = simulate_all

    # Go
    runfunc(filename_totals=args.outfile_totals,
            filename_daily=args.outfile_daily,
            runmode=RunMode(args.runmode))


if __name__ == "__main__":
    main()
