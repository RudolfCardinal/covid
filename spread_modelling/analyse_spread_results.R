#!/usr/bin/env Rscript

# =============================================================================
# Libraries
# =============================================================================

library(conflicted)  # make conflicts explicit
library(data.table)
library(tidyverse)

source("https://egret.psychol.cam.ac.uk/rlib/debugfunc.R")
source("https://egret.psychol.cam.ac.uk/rlib/miscfile.R")

debugfunc$wideScreen()


# =============================================================================
# Directories, filenames
# =============================================================================

THIS_SCRIPT_DIR = miscfile$current_script_directory()

TOTALS_FILENAME <- file.path(THIS_SCRIPT_DIR, "disease_spread_totals.csv")
PEOPLE_FILENAME <- file.path(THIS_SCRIPT_DIR, "disease_spread_people.csv")


# =============================================================================
# Data in
# =============================================================================

totals <- data.table(read.csv(TOTALS_FILENAME))
people <- data.table(read.csv(PEOPLE_FILENAME))
totals[, prop_people_infected := n_people_infected / n_people]
totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
totals[, prop_patients_infected := n_patients_infected / n_patients]
totals[, prop_family_infected := n_family_infected / n_family]

s1 <- totals %>%
    group_by(home_visits, clinicians_meet_each_other,
             infectivity_multiple_if_either_symptomatic) %>%
    summarise(
        mean_prop_people_infected = mean(prop_people_infected),
        mean_prop_clinicians_infected = mean(prop_clinicians_infected),
        mean_prop_patients_infected = mean(prop_patients_infected),
        mean_prop_family_infected = mean(prop_family_infected)
    )
print(s1)

m1 <- lm(
    prop_people_infected ~
        home_visits * clinicians_meet_each_other +
        infectivity_multiple_if_either_symptomatic,
    data = totals)
print(m1)
print(anova(m1))

m2 <- lm(
    prop_clinicians_infected ~
        home_visits * clinicians_meet_each_other +
        infectivity_multiple_if_either_symptomatic,
    data = totals)
print(m2)
print(anova(m2))

m3 <- lm(
    n_contacts ~
        home_visits * clinicians_meet_each_other +
        infectivity_multiple_if_either_symptomatic,
    data = totals)
print(m3)
print(anova(m3))
