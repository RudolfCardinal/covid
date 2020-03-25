#!/usr/bin/env python
# spread_modelling.py

"""
Time units are days.

Re infectivity:

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
          = p(infection | contact between a susceptible and an infected individual)

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
from itertools import combinations, product
import logging
import multiprocessing
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional

from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from cardinal_pythonlib.randomness import coin as cpl_coin
import numpy as np

log = logging.getLogger(__name__)

FuncType = Callable[[Any], Any]
PROFILE = False
NUMPY_COIN = False  # True is slower
THIS_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_SOCIAL_INFECTIVITY_MULTIPLE_IF_SYMPTOMATIC = 0.5
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/tmp/cpft_covid_modelling")


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


class Person(object):
    """
    Represents a person.
    """
    def __init__(
            self,
            name: str,
            pathogenicity_p_symptoms_if_infected: float = 0.67,
            biological_asymptomatic_infectivity_factor: float = 0.5,
            t_infected_to_symptoms_incubation_period: float = 5.1,
            t_infected_to_infectious: float = 4.6,
            infectious_for_t: float = 7.0,
            symptoms_for_t: float = 7.0,
            p_infect_for_full_day_exposure: float = 0.2,
            social_infectivity_multiple_if_symptomatic: float = DEFAULT_SOCIAL_INFECTIVITY_MULTIPLE_IF_SYMPTOMATIC):  # noqa
        """
        Args:
            name:
                Identifying name.
            pathogenicity_p_symptoms_if_infected:
                Probability of developing symptoms if infected.
                Figure of 0.67 is based on influenza [2].
            biological_asymptomatic_infectivity_factor:
                Infectivity is multiplied by this amount if asymptomatic.
                Default is 0.5 [1, 2]: people are half as infectious if
                asymptomatic.
            t_infected_to_symptoms_incubation_period:
                Incubation period; 5.1 days [1].
            t_infected_to_infectious:
                For SARS-CoV-2 (the virus) or COVID-19 (the disease), symptoms
                after 5.1 days (incubation period) but infectious after 4.6
                [1].
            infectious_for_t:
                Duration of infectivity. This is extremely crude.
                Stepwise function in which people are infectious at a constant
                rate for this many days. 7 is the number of days of
                self-isolation recommended by the UK government after symptoms,
                2020-03-24.
            symptoms_for_t:
                Time for which symptoms persist, once developed.
                Also a GUESS. See below re why this shouldn't matter much.
            p_infect_for_full_day_exposure:
                Probability, when infectious, of infecting someone if exposed
                to them for 24 hours per day.
                GUESS. As long as it's constant, it's not so important -- we
                are interested in relative change due to social manipulations,
                rather than absolute rates.
            social_infectivity_multiple_if_symptomatic:
                If this patient is symptomatic, how should infectivity be
                multiplied? This might include numbers >1 (symptomatic people
                are more infectious) or <1 (infection control measures can be
                taken for symptomatic people, e.g. wearing a mask).

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
        self.biological_asymptomatic_infectivity_factor = biological_asymptomatic_infectivity_factor  # noqa
        self.name = name
        self.infectious_for_t = infectious_for_t
        self.p_infect_for_full_day_exposure = p_infect_for_full_day_exposure
        self.social_infectivity_multiple_if_symptomatic = social_infectivity_multiple_if_symptomatic  # noqa
        self.symptoms_for_t = symptoms_for_t
        self.t_days_infected_to_infectious = t_infected_to_infectious
        self.t_infected_to_symptoms_incubation_period = t_infected_to_symptoms_incubation_period  # noqa

        self.infected = False  # infected at some point?
        self.infected_at = None  # type: Optional[float]
        self._symptomatic_if_infected = coin(pathogenicity_p_symptoms_if_infected)  # noqa

    def __str__(self) -> str:
        if self.infected:
            status = f"infected_at_{self.infected_at}"
        else:
            status = "susceptible"
        return f"{self.name}<{status}>"

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
        return float(now - self.infected_at)

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
        if t_since_infected < self.t_days_infected_to_infectious:
            return None
        return t_since_infected - self.t_days_infected_to_infectious

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
            t_since_infected - self.t_infected_to_symptoms_incubation_period >
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
        t_since_infectious = self.time_since_infectious(now)
        if t_since_infectious is None:
            return 0.0
        # Very crude model:

        p_infect = (
            self.p_infect_for_full_day_exposure
            if t_since_infectious < self.infectious_for_t
            else 0.0
        )
        p_infect *= prop_full_day
        if self.symptomatic(now):
            # Symptomatic; people may wear PPE etc.
            p_infect *= self.social_infectivity_multiple_if_symptomatic
        else:
            # Asymptomatic; probably less infectious biologically
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
        if coin(infectivity):  # noqa
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


class Population(object):
    """
    Represents the population being studied, and their interactions.
    """
    def __init__(
            self,
            iteration: int,
            home_visits: bool = False,
            clinicians_meet_each_other: bool = False,
            n_clinicians: int = 20,
            n_patients_per_clinician_per_day: int = 5,
            variable_n_family: bool = False,  # needless reduction in power  # noqa
            mean_n_family_members_per_patient: float = 2,
            n_days: int = 60,
            p_baseline_infected: float = 0.05,
            p_external_infection_per_day: float = 0.0,  # 0.001,
            prop_day_clinician_patient_interaction: float = 1/24,
            prop_day_family_interaction: float = 8/24,
            prop_day_clinician_clinician_interaction: float = 1/24,
            prop_day_clinician_family_intervention: float = 0.2/24,
            social_infectivity_multiple_if_symptomatic: float = DEFAULT_SOCIAL_INFECTIVITY_MULTIPLE_IF_SYMPTOMATIC,  # noqa
            n_random_social_contacts_per_day: int = 0,
            prop_day_random_social_contact: float = 0.1/24):
        """
        Args:
            iteration:
                Internal label.
            home_visits:
                Do clinicians visit patients at home (and thus their family)?
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
            mean_n_family_members_per_patient:
                Mean number of other family members per patient (Poisson
                lambda). If ``variable_n_family`` is ``False``, this number is
                rounded to give the (fixed) number of other family members per
                patient.
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
            social_infectivity_multiple_if_symptomatic:
                See :class:`Person`.
            n_random_social_contacts_per_day:
                number of random contacts per day, for each person, with
                another member of the group
            prop_day_random_social_contact:
                Proportion of a day for a random interaction.
        """
        assert social_infectivity_multiple_if_symptomatic >= 0.0
        self.clinicians_meet_each_other = clinicians_meet_each_other
        self.home_visits = home_visits
        self.iteration = iteration
        self._n_clinicians = n_clinicians
        self.n_days = n_days
        self.n_patients_per_clinician_per_day = n_patients_per_clinician_per_day  # noqa
        self.p_external_infection_per_day = p_external_infection_per_day
        self.n_patients_per_clinician = n_patients_per_clinician_per_day * n_days  # noqa
        self._n_patients = n_clinicians * self.n_patients_per_clinician
        self.n_random_social_contacts_per_day = n_random_social_contacts_per_day  # noqa
        self.p_baseline_infected = p_baseline_infected
        self.prop_day_clinician_clinician_interaction = prop_day_clinician_clinician_interaction  # noqa
        self.prop_day_clinician_family_intervention = prop_day_clinician_family_intervention  # noqa
        self.prop_day_clinician_patient_interaction = prop_day_clinician_patient_interaction  # noqa
        self.prop_day_family_interaction = prop_day_family_interaction
        self.prop_day_random_social_contact = prop_day_random_social_contact
        self.social_infectivity_multiple_if_symptomatic = social_infectivity_multiple_if_symptomatic  # noqa

        self.total_n_contacts = 0
        self.daily_contacts = defaultdict(int)  # type: Dict[int, int]

        patient_kwargs = dict(
            social_infectivity_multiple_if_symptomatic=social_infectivity_multiple_if_symptomatic,  # noqa
        )

        self.clinicians = [
            Person(name=f"Clinician_{i + 1}", **patient_kwargs)
            for i in range(n_clinicians)
        ]
        self.family = []  # type: List[Person]
        self.patients = [
            Person(name=f"Patient_{i + 1}", **patient_kwargs)
            for i in range(self._n_patients)
        ]
        # Map of patient to family members:
        self.p2f = {}  # type: Dict[Person, List[Person]]
        self._n_people = len(self.clinicians) + len(self.patients)

        for pnum, p in enumerate(self.patients, start=1):
            if variable_n_family:
                n_family = np.random.poisson(lam=mean_n_family_members_per_patient)  # noqa
            else:
                n_family = np.round(mean_n_family_members_per_patient)
            family_members = []  # type: List[Person]
            for i in range(n_family):
                f = Person(name=f"Family_{i + 1}_of_patient_{pnum}",
                           **patient_kwargs)
                family_members.append(f)
                self.family.append(f)
            self.p2f[p] = family_members
            self._n_people += len(family_members)

        # Baseline infection
        for p in self.gen_people():
            if coin(p_baseline_infected):
                p.infect(0)

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
        assert 1 <= today <= self.n_days
        today_zb = today - 1
        clinician_idx = self.clinicians.index(clinician)
        first_patient_idx = (
            clinician_idx * self.n_patients_per_clinician +
            today_zb * self.n_patients_per_clinician_per_day
        )
        for patient_idx in range(
                first_patient_idx,
                first_patient_idx + self.n_patients_per_clinician_per_day):
            yield self.patients[patient_idx]

    def simulate_day(self, today: int) -> None:
        """
        Simulate a specific day's interactions.
        """
        def expose(p1_: Person, p2_: Person, prop_day: float) -> None:
            assert p1_ is not p2_
            self.total_n_contacts += 1
            self.daily_contacts[today] += 1
            p1_.mutually_expose(p2_, now, prop_day)

        log.debug(f"Simulating day {today}...")
        now = float(today)
        # Simulate infection from outside our group:
        for p in self.gen_people():
            if p.susceptible() and coin(self.p_external_infection_per_day):
                p.infect(now)
        for p in self.patients:
            # Patients interact with their family:
            family = self.p2f[p]
            for f in family:
                expose(p, f, self.prop_day_family_interaction)
            # Family interact with each other
            for f1, f2 in combinations(family, r=2):  # type: Person, Person
                expose(f1, f2, self.prop_day_family_interaction)
        # Clinicians see their patients:
        for c in self.clinicians:
            for p in self.patients_for_clinician(c, today):
                expose(c, p, self.prop_day_clinician_patient_interaction)
                # Clinicians might meet the family:
                if self.home_visits:
                    family = self.p2f[p]
                    for f in family:
                        expose(c, f,
                               self.prop_day_clinician_family_intervention)

        # Clinicians may get together (e.g. share a base, meet up):
        if self.clinicians_meet_each_other:
            for c1, c2 in combinations(self.clinicians, r=2):  # type: Person, Person  # noqa
                expose(c1, c2,
                       self.prop_day_clinician_clinician_interaction)

        if self.n_random_social_contacts_per_day > 0:
            n_random_contacts = min(self.n_random_social_contacts_per_day,
                                    self.n_people())
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
                    expose(p1, p2, self.prop_day_random_social_contact)

        log.debug(f"... day {today} had {self.daily_contacts[today]} contacts")

    def simulate(self) -> None:
        """
        Simulate all days.
        """
        for day in range(1, self.n_days + 1):
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

    def n_patients(self) -> int:
        """
        The total number of patients.
        """
        return self._n_patients

    def n_patients_infected(self) -> int:
        """
        The number of patients who have been infected at some point.
        """
        return sum(1 for p in self.patients if p.infected)

    def n_clinicians(self) -> int:
        """
        The total number of clinicians.
        """
        return self._n_clinicians

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
            self.n_patients(), self.n_patients_infected(),
            self.n_clinicians(), self.n_clinicians_infected(),
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
        for day in range(1, self.n_days + 1):
            fday = float(day)
            yield [
                day,
                self.n_people_infected_by(fday),
                self.n_clinicians_infected_by(fday),
                self.n_patients_infected_by(fday),
                self.n_family_infected_by(fday),
                self.n_contacts_by(day),
            ]

    @staticmethod
    def _n_infected_by(who: Iterable[Person], when: float) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return sum(1 for p in who
                   if p.infected and p.infected_at <= when)

    def n_people_infected_by(self, when: float) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_infected_by(self.gen_people(), when)

    def n_clinicians_infected_by(self, when: float) -> int:
        """
        Number of clinicians infected by a particular point in time.
        """
        return self._n_infected_by(self.clinicians, when)

    def n_patients_infected_by(self, when: float) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_infected_by(self.patients, when)

    def n_family_infected_by(self, when: float) -> int:
        """
        Number of people infected by a particular point in time.
        """
        return self._n_infected_by(self.family, when)

    def n_contacts_by(self, last_day: int, first_day: int = 1) -> int:
        """
        Number of contacts up to and including ``day``.
        """
        assert first_day <= last_day
        return sum(self.daily_contacts[d]
                   for d in range(first_day, last_day + 1))


def simulate_one(args: List[Any]) -> Population:
    """
    Performs a single run.
    """
    (
        home_visits,
        clinicians_meet_each_other,
        social_infectivity_multiple_if_symptomatic,
        iteration
    ) = args
    log.info(
        f"Simulating: home_visits={home_visits}, "
        f"clinicians_meet_each_other={clinicians_meet_each_other}, "
        f"social_infectivity_multiple_if_symptomatic={social_infectivity_multiple_if_symptomatic}, "  # noqa
        f"iteration={iteration}...")
    pop = Population(
        iteration=iteration,
        home_visits=home_visits,
        clinicians_meet_each_other=clinicians_meet_each_other,
        social_infectivity_multiple_if_symptomatic=social_infectivity_multiple_if_symptomatic  # noqa
    )
    pop.simulate()
    return pop


def simulate_all(filename_totals: str,
                 filename_daily: str,
                 iterations: int,
                 debug: bool = False) -> None:
    """
    Simulate our interactions and save the results.

    Args:
        filename_totals:
            output filename for whole-run totals
        filename_daily:
            output filename for day-by-day details
        iterations:
            number of iterations to run
        debug:
            single iteration, simplified mode
    """
    with open(filename_totals, "wt") as ft, open(filename_daily, "wt") as fp:
        writer_totals = csv.writer(ft)
        control_header_row = [
            "home_visits",
            "clinicians_meet_each_other",
            "social_infectivity_multiple_if_symptomatic",
            "iteration",
        ]
        writer_totals.writerow(control_header_row +
                               Population.totals_csv_header_row())
        writer_people = csv.writer(fp)
        writer_people.writerow(control_header_row +
                               Population.daily_csv_header_row())
        if debug:
            combos = [(
                True,  # home_visits
                True,  # clinicians_meet_each_other
                0.1,  # social_infectivity_multiple_if_symptomatic
                1,  # iterations
            )]
        else:
            combos = product(
                [True, False],  # home_visits
                [True, False],  # clinicians_meet_each_other
                [0.1, 1.0],  # social_infectivity_multiple_if_symptomatic
                range(1, iterations + 1),  # iterations
            )
        n_cpus = multiprocessing.cpu_count()
        log.info(f"Using {n_cpus} processes")

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

        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            for pop in executor.map(simulate_one, combos):
                control_data_row = [
                    pop.home_visits,
                    pop.clinicians_meet_each_other,
                    pop.social_infectivity_multiple_if_symptomatic,
                    pop.iteration,
                ]
                writer_totals.writerow(control_data_row +
                                       pop.totals_csv_data_row())
                for r in pop.daily_csv_data_rows():
                    writer_people.writerow(control_data_row + r)


def selftest() -> None:
    """
    Self-tests.
    """
    log.warning("Testing random number generator")
    n = 1000
    for p in np.arange(0, 1.01, step=0.1):
        x = sum(coin(p) for _ in range(n))
        log.info(f"coin({p}) × {n} → {x} heads")


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
        "--iterations", type=int, default=50,
        help="Number of iterations per condition"
    )
    parser.add_argument(
        "--seed", default=None,
        help="Seed for random number generator (None gives a random seed)"
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

    main_only_quicksetup_rootlogger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        with_process_id=True
    )
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.selftest:
        selftest()
        return

    simulate_all(filename_totals=args.outfile_totals,
                 filename_daily=args.outfile_daily,
                 iterations=args.iterations)


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


if PROFILE:
    main = do_cprofile(main)


if __name__ == "__main__":
    main()
