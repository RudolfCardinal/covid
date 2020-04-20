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

Purpose
=======

Models the spread of an infection such as SARS-CoV-2 coronavirus (COVID-19
disease) under different models of community mental health care.


Notes
=====

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


Changelog
=========

- First internal circulation 2020-04-02.

- 2020-04-02: Bugfix: Exposure to family members was at the patient rate.
  Fixed.

- 2020-04-07:

  - Speedup.
  
  - Daily, not cumulative, values in the "per-day" output.
  
  - Record clinician-patient contact numbers (for when clinicians get sick,
    as below).
    
  - Symptomatic clinicians don't see patients or meet each other.
  
  - Top level restructed slightly.

- 2020-04-08:

  - Option to merge households (n_patients_per_household)
  
- 2020-04-19:

  - Bugfix: bug allowed infection to be double-counted.
  - Shift assistance functions to cardinal_pythonlib==1.0.86.

"""  # noqa

import argparse
from collections import defaultdict
import copy
import csv
from enum import Enum
from itertools import combinations
import logging
import math
import multiprocessing
import os
import random
import resource
from timeit import default_timer, Timer
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    TextIO, Tuple)

from cardinal_pythonlib.contexts import dummy_context_mgr
from cardinal_pythonlib.iterhelp import product_dict
from cardinal_pythonlib.lists import chunks
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from cardinal_pythonlib.parallel import gen_parallel_results_efficiently
from cardinal_pythonlib.profile import do_cprofile
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
    DEBUG_SHORT = "debug_short"
    DEBUG_PROFILE = "debug_profile"
    DEBUG_CHECK_N_ITERATIONS = "debug_check_n_iterations"
    SELFTEST = "selftest"
    EXP1 = "experiment_1"
    EXP1B = "experiment_1b"
    EXP2 = "experiment_2"


class InfectionStatus(Enum):
    """
    S-E-I-R model.
    """
    SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class NoClinicians(Exception):
    pass


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
    coin = cpl_coin  # faster


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
            n_patients_per_day: int = 100,
            max_n_patients_per_clinician_per_day: int = math.inf,
            variable_n_family: bool = True,  # needless reduction in power
            mean_n_family_members_per_patient: float = 1.37,
            n_days: int = 60,
            p_baseline_infected: float = 0.05,
            p_external_infection_per_day: float = 0.0,
            prop_day_clinician_patient_interaction: float = 1 / 24,
            prop_day_family_interaction: float = 8 / 24,
            prop_day_clinician_clinician_interaction: float = 1 / 24,
            prop_day_clinician_family_interaction: float = 0.2 / 24,
            n_random_social_contacts_per_day: int = 0,
            prop_day_random_social_contact: float = 0.1 / 24,
            n_patients_per_household: int = 1,
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
            n_patients_per_day:
                Number of (different) patients to be seen each day, assumed to
                be all new patients each day.
            max_n_patients_per_clinician_per_day:
                Maximum clinician capacity when fully stretched covering
                sick colleagues.
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
            prop_day_clinician_family_interaction:
                Proportion of a day for a clinician-family member encounter
                (on home visits only).
            n_random_social_contacts_per_day:
                number of random contacts per day, for each person, with
                another member of the group
            prop_day_random_social_contact:
                Proportion of a day for a random interaction.
            n_patients_per_household:
                Number of patients per household (usually 1 but used to
                examine further "cluster" effects).
                
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

        self.behavioural_infectivity_multiple_if_symptomatic = behavioural_infectivity_multiple_if_symptomatic  # noqa
        self.biological_asymptomatic_infectivity_factor = biological_asymptomatic_infectivity_factor  # noqa
        self.infectious_for_t = infectious_for_t
        self.p_infect_for_full_day_exposure = p_infect_for_full_day_exposure
        self.pathogenicity_p_symptoms_if_infected = pathogenicity_p_symptoms_if_infected  # noqa
        self.symptoms_for_t = symptoms_for_t
        self.t_infected_to_infectious = t_infected_to_infectious
        self.t_infected_to_symptoms_incubation_period = t_infected_to_symptoms_incubation_period  # noqa

        self.appointment_type = appointment_type
        self.clinicians_meet_each_other = clinicians_meet_each_other
        self.max_n_patients_per_clinician_per_day = max_n_patients_per_clinician_per_day  # noqa
        self.mean_n_family_members_per_patient = mean_n_family_members_per_patient  # noqa
        self.n_clinicians = n_clinicians
        self.n_days = n_days
        self.n_patients_per_day = n_patients_per_day
        self.n_patients_per_household = n_patients_per_household
        self.n_random_social_contacts_per_day = n_random_social_contacts_per_day  # noqa
        self.p_baseline_infected = p_baseline_infected
        self.p_external_infection_per_day = p_external_infection_per_day
        self.prop_day_clinician_clinician_interaction = prop_day_clinician_clinician_interaction  # noqa
        self.prop_day_clinician_family_interaction = prop_day_clinician_family_interaction  # noqa
        self.prop_day_clinician_patient_interaction = prop_day_clinician_patient_interaction  # noqa
        self.prop_day_family_interaction = prop_day_family_interaction
        self.prop_day_random_social_contact = prop_day_random_social_contact
        self.variable_n_family = variable_n_family

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

    # -------------------------------------------------------------------------
    # Configs
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Aspects that vary across configs
    # -------------------------------------------------------------------------

    def varnames(self) -> List[str]:
        """
        Names of variables being manipulated.
        """
        return list(self.kwargs.keys())

    # -------------------------------------------------------------------------
    # Cosmetics
    # -------------------------------------------------------------------------

    def announce_sim(self, config: Config) -> None:
        """
        Announces a simulation, with relevant variables.
        """
        paramlist = [f"{var}={getattr(config, var)}"
                     for var in self.varnames()]
        paramlist.append(f"iteration={config.iteration}")
        params = ", ".join(paramlist)
        log.info(f"Simulating: {params} ...")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Run simulations
    # -------------------------------------------------------------------------

    def simulate_one(self, config: Config) -> "Population":
        """
        Performs a single run.
        """
        pop = Population(config=config)
        self.announce_sim(config)
        start = default_timer()
        pop.simulate()
        end = default_timer()
        log.debug(f"Sim took {end - start} s")
        return pop

    def _simulate_all_main(self,
                           totals_file: TextIO,
                           daily_file: Optional[TextIO],
                           debug_usage: bool = False) -> None:
        """
        Simulate everything and save the results.

        Args:
            totals_file:
                output file for whole-run totals
            daily_file:
                output file for day-by-day totals
            debug_usage:
                show memory usage as we go

        Concurrency performance notes
        =============================

        - Threading doesn't help much (probably because of the Python global
          interpreter lock). Use processes; much quicker.

        - With ProcessPoolExecutor, models of use include:

          .. code-block:: python

            with ProcessPoolExecutor(...) as executor:
                futures = executor.map(function, iterable)
            # ... will complete all? And then
            for result in futures:
                pass

          .. code-block:: python

            with ProcessPoolExecutor(...) as executor:
                for result in executor.map(function, iterable):
                    # ... also waits for many to complete before we get
                    #     here -- but not all? Seems to operate batch-wise.
                    pass
                    
        - Also BoundedProcessPoolExecutor;
          https://pypi.org/project/bounded-pool-executor/
        
        - Also
          https://alexwlchan.net/2019/10/adventures-with-concurrent-futures/

        - Profiling shows that most time is spent in ``{method 'acquire' of
          '_thread.lock' objects}``. But that's because I was profiling the
          top-level process, not the worker.

        - Going faster:

          - https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map
          - You can't 
          - You can't pickle open file objects, or csv.writer objects, and
            therefore you can't use them as arguments to
            ProcessPoolExecutor map() calls.

        """  # noqa
        use_daily_file = bool(daily_file)

        writer_totals = csv.writer(totals_file)
        if use_daily_file:
            writer_daily = csv.writer(daily_file)
        else:
            writer_daily = None  # for linter; won't be used

        control_header_row = self.csv_header_row()
        writer_totals.writerow(control_header_row +
                               Population.totals_csv_header_row())
        if use_daily_file:
            writer_daily.writerow(control_header_row +
                                  Population.daily_csv_header_row())

        n_cpus = multiprocessing.cpu_count()
        log.info(f"Using {n_cpus} processes")

        results = gen_parallel_results_efficiently(
            self.simulate_one,
            self.gen_configs(),
            max_workers=n_cpus
        )
        for pop in results:
            if debug_usage:
                usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                log.debug(f"Top-level process memory usage: {usage}")
                # ... e.g. 1,386,028 for debug_short via
                # BoundedProcessPoolExecutor; 133708 for the same thing via
                # parallelize_processes_efficiently().
            log.debug("Processing result")
            control_data_row = self.csv_data_row(pop.config)
            writer_totals.writerow(control_data_row +
                                   pop.totals_csv_data_row())
            if use_daily_file:
                for r in pop.daily_csv_data_rows():
                    writer_daily.writerow(control_data_row + r)

    def simulate_all(self,
                     totals_filename: str,
                     daily_filename: str = None) -> None:
        """
        Simulate everything and save the results.
        As for :meth:`_simulate_all_main` but using filenames.
        """
        log.info(f"Writing totals to: {totals_filename}")
        if daily_filename:
            log.info(f"Writing daily results to: {daily_filename}")
        with open(totals_filename, "wt") as ft:
            with (open(daily_filename, "wt") if daily_filename
                  else dummy_context_mgr()) as fd:
                self._simulate_all_main(totals_file=ft, daily_file=fd)


# =============================================================================
# Person
# =============================================================================

class Person(object):
    """
    Represents a person.
    """
    def __init__(self, name: str, config: Config,
                 is_clinician: bool = False,
                 is_patient: bool = False,
                 is_family: bool = False):
        """
        Args:
            name:
                Identifying name.
            config:
                :class:`Config` object.
            is_clinician:
                Is this person a clinician?
            is_patient:
                Is this person a patient?
            is_family:
                Is this person a family member of a patient?

        """  # noqa
        self.name = name
        self.is_clinician = is_clinician
        self.is_patient = is_patient
        self.is_family = is_family

        assert sum((is_clinician, is_patient, is_family)) == 1

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
        self._t_infected_to_no_longer_infectious = (
            # Precalculate for speed
            self.t_infected_to_infectious +
            self.infectious_for_t
        )

        self.infected = False  # infected at some point?
        self.infected_at = None  # type: Optional[float]

    # -------------------------------------------------------------------------
    # Descriptives
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Time-independent string representation.
        """
        if self.infected:
            status = f"infected_at_{self.infected_at}"
        else:
            status = "susceptible"
        return f"{self.name}<{status}>"

    def status_at(self, now: float) -> str:
        """
        Time-specific string representation.
        """
        return self.seir_status(now).value

    def str_at(self, now: float) -> str:
        """
        Time-specific more full string representation.
        """
        seir = self.status_at(now)
        symptomatic = "SYMPTOMATIC" if self.symptomatic(now) else "asymptomatic"  # noqa
        return f"{str(self)}<{seir}><{symptomatic}>"

    # -------------------------------------------------------------------------
    # Infection status
    # -------------------------------------------------------------------------

    def seir_status(self, now: float) -> InfectionStatus:
        """
        SEIR status.
        """
        t_since_infected = self.time_since_infected(now)
        if t_since_infected is None:  # not infected
            return InfectionStatus.SUSCEPTIBLE
        if t_since_infected < self.t_infected_to_infectious:
            return InfectionStatus.EXPOSED
        if t_since_infected < self._t_infected_to_no_longer_infectious:
            return InfectionStatus.INFECTIOUS
        return InfectionStatus.RECOVERED

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

    def symptomatic(self, now: float) -> bool:
        """
        Is the patient currently symptomatic?

        Args:
            now: time now
        """
        if not self._symptomatic_if_infected:
            return False
        t_since_infected = self.time_since_infected(now)
        if t_since_infected is None:  # not infected
            return False
        if t_since_infected < self.t_infected_to_symptoms_incubation_period:
            return False
        t_since_symptom_onset = (
            t_since_infected -
            self.t_infected_to_symptoms_incubation_period
        )
        return t_since_symptom_onset < self.symptoms_for_t

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
        if self.seir_status(now) != InfectionStatus.INFECTIOUS:
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

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    # noinspection PyUnusedLocal
    def infect(self, now: float, source: "Person" = None) -> None:
        """
        Infect this patient.

        Args:
            now: time now
            source: (for debugging) source of infection
        """
        if self.infected:
            return
        # log.debug(f"{self.name} infected at {now} by {source}")
        self.infected = True
        self.infected_at = now

    def expose_to(self, other: "Person", now: float,
                  prop_full_day: float) -> bool:
        """
        Exposes ``self`` to ``other``, which may result in infection of
        ``self``. Repeat the other way round for a full interaction.

        Args:
            other:
                the potential "donor"
            now:
                The time now.
            prop_full_day:
                The fraction of a full day for which exposure occurred (on
                the basis that longer exposure increases risk).

        Returns:
            bool: was ``self`` infected by this encounter?
        """
        if self.infected:
            return False  # already infected; can't happen twice
        if not other.infected:
            return False  # much quicker to skip in this situation
        infectivity = other.infectivity(now, prop_full_day)
        if infectivity <= 0.0:
            return False
        if coin(infectivity):
            self.infect(now, other)
            return True
        else:
            return False

    def mutually_expose(self, other: "Person", now: float,
                        prop_full_day: float) -> Tuple[bool, bool]:
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

        Returns:
            tuple: ``self_infected, other_infected``
        """
        self_infected = self.expose_to(other, now, prop_full_day)
        other_infected = other.expose_to(self, now, prop_full_day)
        return self_infected, other_infected


# =============================================================================
# Population
# =============================================================================

class Population(object):
    """
    Represents the population being studied, and their interactions.

    The main data-recording class.
    """
    def __init__(self, config: Config) -> None:
        """
        Args:
            config:
                :class:`Config` object.
        """
        self.config = config

        # Create clinicians
        self.clinicians = [
            Person(name=f"Clinician_{i + 1}", config=config,
                   is_clinician=True)
            for i in range(self.config.n_clinicians)
        ]

        # Create patients
        self.n_patients = config.n_patients_per_day * config.n_days
        self.patients = [
            Person(name=f"Patient_{i + 1}", config=config,
                   is_patient=True)
            for i in range(self.n_patients)
        ]

        # Assign patients randomly to days
        patientcopy = copy.copy(self.patients)  # shallow copy
        random.shuffle(patientcopy)
        self.day2patients = {
            day: patients
            for day, patients in enumerate(chunks(patientcopy,
                                                  n=config.n_patients_per_day),
                                           start=1)
        }  # type: Dict[int, List[Person]]

        # List of all households:
        self.households = []  # type: List[List[Person]]
        # Map of patient to household:
        self.p2h = {}  # type: Dict[Person, List[Person]]
        self._n_people = len(self.clinicians) + len(self.patients)
        self.all_nonpatient_family_members = []  # type: List[Person]

        # Assign people consecutively to families/households.
        # (Compare the random assignment of patients to days.)
        current_household = []  # type: List[Person]
        n_patients_in_household = 0
        for pnum, p in enumerate(self.patients, start=1):
            # Create family
            if config.variable_n_family:
                n_family = np.random.poisson(
                    lam=config.mean_n_family_members_per_patient)
            else:
                n_family = np.round(config.mean_n_family_members_per_patient)
            family_members = []  # type: List[Person]
            for i in range(n_family):
                f = Person(name=f"Family_{i + 1}_of_patient_{pnum}",
                           config=config, is_family=True)
                family_members.append(f)
                self.all_nonpatient_family_members.append(f)
            self._n_people += len(family_members)

            # Keep a record of which household the patient lives in
            self.p2h[p] = current_household

            # Add patient and family to household
            current_household.append(p)
            current_household.extend(family_members)

            # Move to next household?
            n_patients_in_household += 1
            if n_patients_in_household >= self.config.n_patients_per_household:
                self.households.append(current_household)
                current_household = []  # type: List[Person]
                n_patients_in_household = 0

        # Leftovers?
        if len(current_household) > 0:
            self.households.append(current_household)

        # Logging. Keys are day number.
        self.daily_contacts = defaultdict(int)  # type: Dict[int, int]
        self.daily_clinician_patient_contacts = defaultdict(int)  # type: Dict[int, int]  # noqa
        self._n_people_infected_on = defaultdict(int)  # type: Dict[int, int]
        self._n_clinicians_infected_on = defaultdict(int)  # type: Dict[int, int]  # noqa
        self._n_patients_infected_on = defaultdict(int)  # type: Dict[int, int]
        self._n_family_infected_on = defaultdict(int)  # type: Dict[int, int]

        # Baseline infection
        for p in self.gen_people():
            if coin(self.config.p_baseline_infected):
                p.infect(0)
                # By definition they weren't infected before, so:
                self.log_infection(p, 0)

    # -------------------------------------------------------------------------
    # Descriptives
    # -------------------------------------------------------------------------

    def day_to_patient_str(self) -> str:
        """
        Debugging representation of self.day2patients
        """
        d2p = []  # type: List[str]
        for day, patients in self.day2patients.items():
            numpatients = len(patients)
            pstr = ", ".join(p.name for p in patients)
            d2p.append(f"Day {day} ({numpatients} patients): {pstr}")
        return "\n".join(d2p)

    def households_str(self) -> str:
        """
        Debugging representation of self.households
        """
        h2p = []  # type: List[str]
        for hnum, household in enumerate(self.households, start=1):
            people = ", ".join(p.name for p in household)
            h2p.append(f"Household {hnum}: {people}")
        return "\n".join(h2p)

    # -------------------------------------------------------------------------
    # People
    # -------------------------------------------------------------------------

    def gen_people(self) -> Iterable[Person]:
        """
        Yield all people in turn.
        """
        for p in self.clinicians:
            yield p
        for p in self.patients:
            yield p
        for p in self.all_nonpatient_family_members:
            yield p

    def gen_available_clinicians_cyclically(self, now: float) \
            -> Generator[Person, None, None]:
        """
        Generates asymptomatic clinicians available for work today.
        Assumes that each clinician is used to see a patient, and notes that.
        Cycles through clinicians infinitely or until they are all overworked.
        Raises ``NoClinicians`` when it runs out.

        Args:
            now:
                time now
        """
        clinicians = [c for c in self.clinicians if not c.symptomatic(now)]
        # log.info(f"Using {len(clinicians)} asymptomatic clinicians out of "
        #          f"{len(self.clinicians)} total clinicians")
        if not clinicians:
            raise NoClinicians("All sick")
        n_patients_seen_per_clinician = defaultdict(int)  # type: Dict[Person, int]  # noqa
        max_patients_per_clinician = self.config.max_n_patients_per_clinician_per_day  # noqa
        while True:
            found_one = False
            for c in clinicians:
                if n_patients_seen_per_clinician[c] >= max_patients_per_clinician:  # noqa
                    # log.info(f"Skipping overworked clinician: {c}")
                    continue
                found_one = True
                n_patients_seen_per_clinician[c] += 1
                yield c
            if not found_one:
                raise NoClinicians("All clinicians at max patients per day")

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

        assert 0 <= idx < len(self.all_nonpatient_family_members)
        return self.all_nonpatient_family_members[idx]

    def patients_for_today(self, today: int) -> Iterable[Person]:
        """
        Yield patients needing to be seen today.
        """
        assert 1 <= today <= self.config.n_days
        for patient in self.day2patients[today]:
            yield patient

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def log_infection(self, p: Person, today: int) -> None:
        """
        Record the infection of a person.
        """
        self._n_people_infected_on[today] += 1
        if p.is_clinician:
            self._n_clinicians_infected_on[today] += 1
        if p.is_patient:
            self._n_patients_infected_on[today] += 1
        if p.is_family:
            self._n_family_infected_on[today] += 1

    def simulate_day(self, today: int) -> None:
        """
        Simulate a specific day's interactions.
        """
        def expose_pair(p1_: Person, p2_: Person, prop_day: float) -> None:
            """
            Expose two people to each other.

            - Record infections now, for speed.
            - Record details of a contact (except clinician-patient contact
              logging, performed below).
            """
            # assert p1_ is not p2_
            self.daily_contacts[today] += 1
            p1_infected, p2_infected = p1_.mutually_expose(p2_, now, prop_day)
            if p1_infected:
                self.log_infection(p1_, today)
            if p2_infected:
                self.log_infection(p2_, today)

        # log.debug(f"Simulating day {today}...")
        now = float(today)

        # Simulate infection from outside our group:
        p_external_infection = self.config.p_external_infection_per_day
        for p in self.gen_people():
            # p.susceptible() is faster than coin()
            if p.susceptible() and coin(p_external_infection):
                p.infect(now)
                # They weren't infected before (as they were susceptible), so:
                self.log_infection(p, today)

        # Household interactions:
        prop_day_family = self.config.prop_day_family_interaction
        for household in self.households:  # patients and family
            # Every person interacts with every other
            for p1, p2 in combinations(household, r=2):  # type: Person, Person
                expose_pair(p1, p2, prop_day_family)

        appointment_type = self.config.appointment_type
        if appointment_type in [Appointment.HOME_VISIT, Appointment.CLINIC]:
            prop_day_c_p = self.config.prop_day_clinician_patient_interaction
            prop_day_c_f = self.config.prop_day_clinician_family_interaction
            clingen = self.gen_available_clinicians_cyclically(now)
            try:
                for p in self.patients_for_today(today):
                    c = next(clingen)
                    # Clinicians meet their patients:
                    expose_pair(c, p, prop_day_c_p)
                    self.daily_clinician_patient_contacts[today] += 1
                    # Clinicians might meet the rest of the household:
                    if appointment_type == Appointment.HOME_VISIT:
                        household = self.p2h[p]
                        for h in household:
                            expose_pair(c, h, prop_day_c_f)
            except NoClinicians:
                log.warning(f"Out of clinicians on day {today}")
        elif appointment_type == Appointment.REMOTE:
            # Nobody meets up
            pass
        else:
            raise ValueError("Bad appointment type")

        # Clinicians may get together (e.g. share a base, meet up),
        # if they are asymptomatic.
        if self.config.clinicians_meet_each_other:
            prop_day_c_c = self.config.prop_day_clinician_clinician_interaction
            asymptomatic_clinicians = [c for c in self.clinicians
                                       if not c.symptomatic(now)]
            for c1, c2 in combinations(asymptomatic_clinicians, r=2):  # type: Person, Person  # noqa
                expose_pair(c1, c2, prop_day_c_c)

        # Random contacts within the population?
        if self.config.n_random_social_contacts_per_day > 0:
            # -----------------------------------------------------------------
            # NOT USED -- IGNORE
            # If resurrected: does not account for self-isolation of those who
            # are symptomatic (not even clinicians).
            # -----------------------------------------------------------------
            prop_day_random = self.config.prop_day_random_social_contact
            n_random_contacts = min(
                self.config.n_random_social_contacts_per_day,
                self.n_people()
            )
            # log.debug(f"Random social contacts with "
            #           f"{n_random_contacts} people each")
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
                    expose_pair(p1, p2, prop_day_random)

        # log.debug(f"... day {today} had {self.daily_contacts[today]} contacts")  # noqa

    def simulate(self) -> None:
        """
        Simulate all days.
        """
        for day in range(1, self.config.n_days + 1):
            self.simulate_day(day)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def n_people_infected(self) -> int:
        """
        The number of people who have been infected at some point.
        """
        # return sum(1 for p in self.gen_people() if p.infected)
        return sum(v for v in self._n_people_infected_on.values())

    def n_people(self) -> int:
        """
        The total number of people.
        """
        return self._n_people

    def n_patients_infected(self) -> int:
        """
        The number of patients who have been infected at some point.
        """
        # return sum(1 for p in self.patients if p.infected)
        return sum(v for v in self._n_patients_infected_on.values())

    def n_clinicians_infected(self) -> int:
        """
        The number of clinicians who have been infected at some point.
        """
        return sum(1 for c in self.clinicians if c.infected)

    def n_family(self) -> int:
        """
        The total number of family members of patients.
        """
        return len(self.all_nonpatient_family_members)

    def n_family_infected(self) -> int:
        """
        The number of family members who have been infected.
        """
        # return sum(1 for f in self.all_nonpatient_family_members if f.infected)  # noqa
        return sum(v for v in self._n_family_infected_on.values())

    def total_n_contacts(self) -> int:
        """
        The total number of person-to-person interactions.
        """
        return sum(v for v in self.daily_contacts.values())

    def n_clinician_patient_contacts(self) -> int:
        """
        The total number of clinician-patient interactions.
        """
        return sum(v for v in self.daily_clinician_patient_contacts.values())

    # -------------------------------------------------------------------------
    # CSV data
    # -------------------------------------------------------------------------

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
            "n_contacts", "n_clinician_patient_contacts",
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
            self.total_n_contacts(), self.n_clinician_patient_contacts(),
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
            "n_clinician_patient_contacts",
        ]

    def daily_csv_data_rows(self) -> Iterable[List[Any]]:
        """
        CSV data row for the "people" output.
        Includes 0, the day before simulation starts.
        """
        for day in range(0, self.config.n_days + 1):
            yield [
                day,
                self._n_people_infected_on[day],
                self._n_clinicians_infected_on[day],
                self._n_patients_infected_on[day],
                self._n_family_infected_on[day],
                self.daily_contacts[day],
                self.daily_clinician_patient_contacts[day],
            ]


# =============================================================================
# Simulation framework
# =============================================================================

def test_profile_sim(n_iterations: int) -> None:
    """
    Profile. There's little point in profiling the top-level functions, since
    when the parallel processing component starts, we don't see the functions
    being called. Hence this.
    """
    mc = Metaconfig(
        n_iterations=n_iterations,
        appointment_type=[Appointment.HOME_VISIT],
        clinicians_meet_each_other=[True],
        behavioural_infectivity_multiple_if_symptomatic=[0.1],
        p_baseline_infected=[0.05],
        p_external_infection_per_day=[0.01],
        n_patients_per_household=[1],
    )
    for config in mc.gen_configs():
        mc.simulate_one(config)


def test_check_n_iterations(totals_filename: str,
                            daily_filename: str) -> None:
    """
    Check that the default number of iterations is sensible.

    Args:
        totals_filename:
            output filename for whole-run totals
        daily_filename:
            output filename for day-by-day details
    """
    mc = Metaconfig(
        n_iterations=N_ITERATIONS,
        appointment_type=[Appointment.HOME_VISIT],
        clinicians_meet_each_other=[True],
        behavioural_infectivity_multiple_if_symptomatic=[0.1],
        p_baseline_infected=[0.01],
        p_external_infection_per_day=[0.0],
        n_patients_per_household=[1],
    )
    mc.simulate_all(totals_filename=totals_filename,
                    daily_filename=daily_filename)


def experiment_1(totals_filename: str,
                 daily_filename: str,
                 n_iterations: int = N_ITERATIONS) -> None:
    """
    Experiment 1 (main experiment).
    Simulate our interactions and save the results.

    Args:
        totals_filename:
            output filename for whole-run totals
        daily_filename:
            output filename for day-by-day details
        n_iterations:
            number of iterations
    """
    mc = Metaconfig(
        n_iterations=n_iterations,
        appointment_type=[x for x in Appointment],
        # ... can do list(Appointment) but the type checker complains
        clinicians_meet_each_other=[True, False],
        behavioural_infectivity_multiple_if_symptomatic=[0.1, 1.0],
        p_baseline_infected=[0.01, 0.05],
        p_external_infection_per_day=[0.0, 0.02],  # 0%, 2%
        n_patients_per_household=[1],
    )
    mc.simulate_all(totals_filename=totals_filename,
                    daily_filename=daily_filename)


def experiment_1b(totals_filename: str,
                  daily_filename: str,
                  n_iterations: int = N_ITERATIONS) -> None:
    """
    Experiment 1B: external infection a bit lower (may be at ceiling in terms
    of dominating the other effect in Exp 1).
    """
    mc = Metaconfig(
        n_iterations=n_iterations,
        appointment_type=[x for x in Appointment],
        clinicians_meet_each_other=[True, False],
        behavioural_infectivity_multiple_if_symptomatic=[0.1, 1.0],
        p_baseline_infected=[0.01, 0.05],
        p_external_infection_per_day=[0.005, 0.01],  # 0.5%, 1%
        n_patients_per_household=[1],
    )
    mc.simulate_all(totals_filename=totals_filename,
                    daily_filename=daily_filename)


def experiment_2(totals_filename: str) -> None:
    """
    An additional test, relating to "hubs" or connectedness in the population
    and the effect of clinician-clinician interactions (and appointment type)
    then.

    Args:
        totals_filename:
            output filename for whole-run totals
    """
    mc = Metaconfig(
        n_iterations=N_ITERATIONS,
        appointment_type=[Appointment.HOME_VISIT,
                          Appointment.CLINIC,
                          Appointment.REMOTE],
        clinicians_meet_each_other=[True, False],
        behavioural_infectivity_multiple_if_symptomatic=[1.0],
        p_baseline_infected=[0.01],
        p_external_infection_per_day=[0.0],
        n_patients_per_household=list(range(1, 10 + 1)),
    )
    mc.simulate_all(totals_filename=totals_filename)


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
    p1 = Person("Test_person_1_symptomatic", cfg, is_patient=True)
    cfg.pathogenicity_p_symptoms_if_infected = 0.0
    p2 = Person("Test_person2_asymptomatic", cfg, is_patient=True)
    for p in [p1, p2]:
        log.info(f"Person: {p}")
        for now in range(-3, 20 + 1):
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

    log.warning("Speed-testing functions")
    n_timeit = 10000
    t = Timer(lambda: coin(0.5))
    log.info(f" coin(0.5): {t.timeit(number=n_timeit)}")
    t = Timer(lambda: p.susceptible())
    log.info(f" p.susceptible(): {t.timeit(number=n_timeit)}")

    log.warning("Testing clinician redistribution")
    n_clinicians = 10
    ill_clinicians = n_clinicians // 2
    infection_day = 0
    test_day = 8
    n_test_patients = 20
    max_n_patients_per_clinician_per_day = n_test_patients
    cfg = Config(
        n_clinicians=n_clinicians,
        n_days=test_day,
        n_patients_per_day=30,
        n_patients_per_household=7,
        max_n_patients_per_clinician_per_day=max_n_patients_per_clinician_per_day  # noqa
    )
    pop = Population(cfg)
    for c in pop.clinicians[:ill_clinicians]:
        c.infect(infection_day)
    clingen = pop.gen_available_clinicians_cyclically(test_day)
    for i in range(1, n_test_patients + 1):
        c = next(clingen)
        log.info(f"For patient {i}, clinician: {c.str_at(test_day)}")

    log.warning("Testing patient-to-day allocation")
    log.info(f"day2patients:\n{pop.day_to_patient_str()}")
    dpstr = ", ".join(p.name for p in pop.patients_for_today(test_day))
    log.info(f"Patients for day {test_day}, another way:\n{dpstr}")

    log.warning("Testing household allocation")
    log.info(f"households:\n{pop.households_str()}")
    first_household = ", ".join(p.name for p in pop.p2h[pop.patients[0]])
    log.info(f"Household of first patient: {first_household}")


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
        "runmode", type=str,
        choices=[x.value for x in RunMode],
        help="Method of running the program"
    )
    parser.add_argument(
        "--outfile_exp1_totals", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "exp1_totals.csv"),
        help="Experient 1: Main output file for totals"
    )
    parser.add_argument(
        "--outfile_exp1_daily", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "exp1_daily.csv"),
        help="Experiment 1: Main output file for daily results"
    )
    parser.add_argument(
        "--outfile_exp1b_totals", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "exp1b_totals.csv"),
        help="Experient 1B: Main output file for totals"
    )
    parser.add_argument(
        "--outfile_exp1b_daily", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "exp1b_daily.csv"),
        help="Experiment 1B: Main output file for daily results"
    )
    parser.add_argument(
        "--outfile_exp2_totals", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "exp2_totals.csv"),
        help="Experiment 2: output file for totals"
    )
    parser.add_argument(
        "--outfile_test_totals", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "test_totals.csv"),
        help="Test output file for totals"
    )
    parser.add_argument(
        "--outfile_test_daily", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "test_daily.csv"),
        help="Test output file for daily results"
    )
    parser.add_argument(
        "--seed", default=1234, type=str,
        help="Seed for random number generator ('None' gives a random seed); "
             "ignored for predefined experiments."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Be verbose"
    )
    args = parser.parse_args()

    # Logging
    main_only_quicksetup_rootlogger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        with_process_id=True
    )

    # What we're doing
    runmode = RunMode(args.runmode)

    # RNG seed
    if runmode in [RunMode.EXP1, RunMode.EXP2]:
        seed = 1234  # for consistency
    elif runmode == RunMode.EXP1B:
        # Not that it is likely to make any difference, but ideally this should
        # be different for Exp 1B and Exp 1.
        seed = 2345
    else:
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

    # Do something useful
    if runmode == RunMode.SELFTEST:
        selftest()
    elif runmode == RunMode.DEBUG_SHORT:
        log.warning(f"Pausing... Fire up 'top -p {os.getpid()}' "
                    f"to watch memory usage...")
        input("Press Enter to continue: ")
        experiment_1(totals_filename=args.outfile_test_totals,
                     daily_filename=args.outfile_test_daily,
                     n_iterations=2)
    elif runmode == RunMode.DEBUG_PROFILE:
        do_cprofile(test_profile_sim)(n_iterations=8)
    elif runmode == RunMode.DEBUG_CHECK_N_ITERATIONS:
        test_check_n_iterations(totals_filename=args.outfile_test_totals,
                                daily_filename=args.outfile_test_daily)
    elif runmode == RunMode.EXP1:
        experiment_1(totals_filename=args.outfile_exp1_totals,
                     daily_filename=args.outfile_exp1_daily)
    elif runmode == RunMode.EXP1B:
        experiment_1b(totals_filename=args.outfile_exp1b_totals,
                      daily_filename=args.outfile_exp1b_daily)
    elif runmode == RunMode.EXP2:
        experiment_2(totals_filename=args.outfile_exp2_totals)
    else:
        raise ValueError(f"Unknown runmode: {runmode!r}")

    log.info("Done.")


if __name__ == "__main__":
    main()
