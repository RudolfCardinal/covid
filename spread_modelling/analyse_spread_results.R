#!/usr/bin/env Rscript

# =============================================================================
# Libraries
# =============================================================================

library(conflicted)  # make conflicts explicit
library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)
library(patchwork)
library(tidyverse)


# =============================================================================
# Directories, filenames
# =============================================================================

DATA_RESULTS_DIR <- path.expand("~/tmp/cpft_covid_modelling")

TOTALS_FILENAME <- file.path(DATA_RESULTS_DIR, "disease_spread_totals.csv")
DAILY_FILENAME <- file.path(DATA_RESULTS_DIR, "disease_spread_daily.csv")
RESULTS_FILENAME <- file.path(DATA_RESULTS_DIR, "results.txt")
FIGURE_FILENAME <- file.path(DATA_RESULTS_DIR, "figures.pdf")


# =============================================================================
# Data in
# =============================================================================

totals <- data.table(read.csv(TOTALS_FILENAME))
totals[, home_visits := as.logical(home_visits)]
totals[, clinicians_meet_each_other := as.logical(clinicians_meet_each_other)]
totals[, prop_people_infected := n_people_infected / n_people]
totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
totals[, prop_patients_infected := n_patients_infected / n_patients]
totals[, prop_family_infected := n_family_infected / n_family]
totals[, p_external_infection_per_day_factor := as.factor(
    p_external_infection_per_day)]

daily <- data.table(read.csv(DAILY_FILENAME))
daily[, home_visits := as.logical(home_visits)]
daily[, clinicians_meet_each_other := as.logical(clinicians_meet_each_other)]
daily[, group := paste0(
    "HV", as.integer(home_visits), "_",
    "CM", as.integer(clinicians_meet_each_other), "_",
    "SIMS", social_infectivity_multiple_if_symptomatic, "_",
    "EXT", p_external_infection_per_day
)]
daily[, social_infectivity_multiple_if_symptomatic_factor := as.factor(
    social_infectivity_multiple_if_symptomatic)]
daily[, p_external_infection_per_day_factor := as.factor(
    p_external_infection_per_day)]

sink(RESULTS_FILENAME)

s1 <- totals %>%
    group_by(
        home_visits, clinicians_meet_each_other,
        social_infectivity_multiple_if_symptomatic,
        p_external_infection_per_day_factor
    ) %>%
    summarise(
        mean_prop_people_infected = mean(prop_people_infected),
        mean_prop_clinicians_infected = mean(prop_clinicians_infected),
        mean_prop_patients_infected = mean(prop_patients_infected),
        mean_prop_family_infected = mean(prop_family_infected),
        mean_n_contacts = mean(n_contacts)
    )
print(s1)
daily_long <- daily %>%
    gather(key = who,
           value = n_infected,
           n_people_infected,
           n_clinicians_infected,
           n_patients_infected,
           n_family_infected) %>%
    mutate(
        who = dplyr::recode(
            who,
            n_people_infected = "people",
            n_clinicians_infected = "clinicians",
            n_patients_infected = "patients",
            n_family_infected = "family"
        )
    )


# These analyses don't take account of "day" properly:
m1 <- lm(
    n_people_infected ~
        home_visits * clinicians_meet_each_other * p_external_infection_per_day_factor +
        social_infectivity_multiple_if_symptomatic,
    data = totals)
print(m1)
print(anova(m1))

m2 <- lm(
    n_clinicians_infected ~
        home_visits * clinicians_meet_each_other * p_external_infection_per_day_factor +
        social_infectivity_multiple_if_symptomatic,
    data = totals)
print(m2)
print(anova(m2))

m3 <- lm(
    n_contacts ~
        home_visits * clinicians_meet_each_other * p_external_infection_per_day_factor +
        social_infectivity_multiple_if_symptomatic,
    data = totals)
print(m3)
print(anova(m3))

sink()

# Faceting on this doesn't work well -- scales too disparate:
plotdata_a <- daily[p_external_infection_per_day_factor == 0]
plotdata_b <- daily[p_external_infection_per_day_factor == 0.02]
common_elements <- list(
    theme_bw(),
    scale_colour_manual(values = c("blue", "red")),
    scale_linetype_manual(values = c("dotted", "solid")),
    scale_size_manual(values = c(0.5, 2)),
    scale_y_continuous(limits = c(0, NA))
)
p1a <- (
    ggplot(
        plotdata_a,
        aes(x = day, y = n_clinicians_infected, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    stat_smooth() +
    common_elements +
    theme(legend.position = "bottom") +
    ggtitle("Infected CLINICIANS: no incoming infection")
)
p1b <- (
    ggplot(
        plotdata_b,
        aes(x = day, y = n_clinicians_infected, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    stat_smooth() +
    common_elements +
    theme(legend.position = "none") +
    ggtitle("Infected CLINICIANS: external infection")
)
p2a <- (
    ggplot(
        plotdata_a,
        aes(x = day, y = n_people_infected, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    stat_smooth() +
    common_elements +
    theme(legend.position = "none") +
    ggtitle("Infected PEOPLE: no incoming infection")
)
p2b <- (
    ggplot(
        plotdata_b,
        aes(x = day, y = n_people_infected, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    stat_smooth() +
    common_elements +
    theme(legend.position = "none") +
    ggtitle("Infected PEOPLE: external infection")
)
p3a <- (
    ggplot(
        plotdata_a,
        aes(x = day, y = n_contacts, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    geom_line() +
    common_elements +
    theme(legend.position = "none") +
    ggtitle("#Contacts")
)
p3b <- (
    ggplot(
        plotdata_b,
        aes(x = day, y = n_contacts, group = group,
            colour = home_visits,
            linetype = clinicians_meet_each_other,
            size = social_infectivity_multiple_if_symptomatic_factor)
    ) +
    geom_line() +
    common_elements +
    theme(legend.position = "none") +
    ggtitle("#Contacts")
)

fig <- (
    (p1a | p1b) /
    (p2a | p2b) /
    (p3a | p3b) +
    plot_layout(guides = "collect")
)

ggsave(FIGURE_FILENAME, fig, width = 40, height = 30, units = "cm")
