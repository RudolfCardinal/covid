# spread_modelling.R
# Rudolf Cardinal, 23 Mar 2020
# Disclaimer: I am not an epidemiologist.
# ABANDONED -- use Python version instead.
#
# =============================================================================
# Purpose
# =============================================================================
#
# Is it better to have a home visit system or a people-come-to-base system?
#
# =============================================================================
# Notes
# =============================================================================
#
# - The basic reproduction number R0 is the expected number of secondary cases
#   produced by a single infection in a completely susceptible population.
#   It is OF THE ORDER of 2.4 for SARS-CoV-2 (the virus) or COVID-19 (the
#   disease).
#
#   - https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
#     ... using 2.4 (but examing range 2.0-2.6)
#   - https://www.ncbi.nlm.nih.gov/pubmed/32097725
#     ... 2.28
#   - https://academic.oup.com/jtm/article/27/2/taaa021/5735319
#     ... mean 3.28, median 2.79
#   - https://www.nature.com/articles/s41421-020-0148-0
#    ... using estimates in the range 1.9, 2.6, 3.1
#
# - Then R0 is equal to:
#
#   τ × c_bar × d
#
#   where:
#       τ     = transmissibility
#             = p(infection | contact between a susceptible and an infected individual)
#
#       c_bar = average rate of contact between susceptible and infected
#               individuals (contacts per unit time)
#
#       d     = duration of infectiousness
#
#   ... see https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf


# =============================================================================
# Libraries
# =============================================================================

library(conflicted)  # make conflicts explicit

source("https://egret.psychol.cam.ac.uk/rlib/debugfunc.R")
source("https://egret.psychol.cam.ac.uk/rlib/miscfile.R")

debugfunc$wideScreen()

# =============================================================================
# Directories, filenames
# =============================================================================

THIS_SCRIPT_DIR = miscfile$current_script_directory()
FIT_CACHE_DIR <- file.path(THIS_SCRIPT_DIR, "fitcache")
if (!file.exists(FIT_CACHE_DIR)) {
    dir.create(FIT_CACHE_DIR)
}


# =============================================================================
# Seed random number generator (for "consistent randomness")
# =============================================================================

set.seed(918)


# =============================================================================
# Helper functions
# =============================================================================

# =============================================================================
# Main function
# =============================================================================

sim <- function(
    # The choice of interest
    home_visits = TRUE,
    clinicians_meet_each_other = TRUE,
    # Our situation
    n_clinicians = 10,
    n_patients_per_clinician_per_day = 10,
    mean_n_family_members_per_patient = 2,  # Poisson distribution
    n_days = 100,
    p_external_infection_per_day = 0.01,  # probability of being infected by someone external to our situation
    p_single_clinician_interaction_infects = 0.05,  # 1 h of talking
    p_day_with_infected_family_member_infects = 0.2,
    # COVID-19 biology
    pathogenicity_p_symptoms_if_infected = 0.3,  # *** ??
    asymptomatic_infectivity_factor = 0.5,  # half as infectious if asymptomatic [1]
    t_days_infected_to_infectious = 4.6  # symptoms after 5.1 days (incubation period) but infectious after 4.6 [1]
)
{
    n_patients <- n_patients_per_clinician_per_day * n_days
    vn_family <- rpois(n = n_patients, lambda = mean_n_family_members_per_patient)
    n_family <- sum(vn_family)
    n_people <- n_clinicians + n_patients + n_family
    idx_clinicians <- 1:n_clinicians
    idx_patients <- n_clinicians + 1:n_patients
    idx_family <- (n_clinicians + n_patients + 1):n_people
    cat(paste0("Simulating for ", n_people, " people...\n"))

    infected_on_day <- rep(0, n_people)  # 0 for not infected
    symptomatic <- rbinom(n = n_people, size = 1, prob = pathogenicity_p_symptoms_if_infected)
    propensity_to_be_infectious <- rgamma(n = n_people, rate = 1, shape = 0.25)  # [1]
    family_member_to_pt <- c(
        rep(0, n_clinicians + n_patients),
        inverse.rle(
            structure(list(lengths = vn_family, values = idx_patients),
                      class = "rle")
        )
    )

    contacts_every_day <- matrix(0, nrow = n_people, ncol = n_people)
    for (p in idx_patients) {
        idx_family_members <- family_member_to_pt[family_member_to_pt == p]
        for (f in idx_family_members) {
            # Patient meets family member, family member meets patient
            contacts_every_day[p, f] <- 1
            contacts_every_day[f, p] <- 1

        }
    }
    if (clinicians_meet_each_other) {
        for (c1 in idx_clinicians) {
            for (c2 in idx_clinicians) {
                contacts_every_day[c1, c2] <- 1
            }
        }
    }
    # Eliminate self-interaction:
    diag(contacts_every_day) <- 0

    infectiousness <- function(day_) {
        ifelse(
            infected_on_day == 0,
            0,  # not yet infected, so not infectious
            (
                pmax(0, (day - infected_on_day) - t_days_infected_to_infectious) *
                ifelse(symptomatic, 1, asymptomatic_infectivity_factor)
            )
        )
    }

    cumulative_n_infected_by_day <- c()

    for (day in 1:n_days) {
        # Who's susceptible today? Vector of boolean.
        susceptible <- infected_on_day == 0  # Those not yet infected.

        # People randomly catch from external interactions
        externally_infected_today <- (
            susceptible &
            rbinom(n = n_people, size = 1, prob = p_external_infection_per_day)
        )
        infected_on_day[externally_infected_today] <- day

        # Now, our group interact with each other.
        contacts_today <- contacts_every_day

        # Patients and their families interact
        if (home_visits) {
            # Clinicians interact with patients and families
        } else {
            # Clinicians interact with patients
        }

        cumulative_n_infected_by_day <- c(cumulative_n_infected_by_day,
                                          sum(infected_on_day > 0))
    }
    return(cumulative_n_infected_by_day)
}


# =============================================================================
# Run
# =============================================================================

fit1 <- stanfunc$load_or_run_stan(
    data = STANDATA,
    model_code = MODEL_CODE,
    fit_filename = file.path(FIT_CACHE_DIR, "model_1.rds"),
    model_name = "m1",
    seed = SEED,
    forcerun = TRUE
)
