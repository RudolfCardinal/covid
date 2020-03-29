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

source("https://egret.psychol.cam.ac.uk/rlib/miscfile.R")
source("https://egret.psychol.cam.ac.uk/rlib/miscstat.R")


# =============================================================================
# Directories, filenames
# =============================================================================

THIS_DIR <- miscfile$current_script_directory()
DATA_RESULTS_DIR <- file.path(THIS_DIR, "results")

TOTALS_FILENAME <- file.path(DATA_RESULTS_DIR, "disease_spread_totals.csv")
DAILY_FILENAME <- file.path(DATA_RESULTS_DIR, "disease_spread_daily.csv")
RESULTS_FILENAME <- file.path(DATA_RESULTS_DIR, "results.txt")
FIGURE_FILENAME <- file.path(DATA_RESULTS_DIR, "figures.pdf")


# =============================================================================
# Data in
# =============================================================================

totals <- data.table(read.csv(TOTALS_FILENAME))
totals[, clinicians_meet_each_other := as.logical(clinicians_meet_each_other)]
totals[, prop_people_infected := n_people_infected / n_people]
totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
totals[, prop_patients_infected := n_patients_infected / n_patients]
totals[, prop_family_infected := n_family_infected / n_family]
totals[, behavioural_infectivity_multiple_if_symptomatic := as.factor(
    behavioural_infectivity_multiple_if_symptomatic)]
totals[, p_baseline_infected := as.factor(p_baseline_infected)]
totals[, p_external_infection_per_day := as.factor(
    p_external_infection_per_day)]
# Shorter name for behavioural_infectivity_multiple_if_symptomatic:
totals[, ppe_effect := behavioural_infectivity_multiple_if_symptomatic]
totals[, behavioural_infectivity_multiple_if_symptomatic := NULL]

daily <- data.table(read.csv(DAILY_FILENAME))
daily[, clinicians_meet_each_other := as.logical(clinicians_meet_each_other)]
daily[, group := paste0(
    "AT", appointment_type, "_",
    "CM", as.integer(clinicians_meet_each_other), "_",
    "SIMS", behavioural_infectivity_multiple_if_symptomatic, "_",
    "EXT", p_external_infection_per_day
)]
daily[, behavioural_infectivity_multiple_if_symptomatic := as.factor(
    behavioural_infectivity_multiple_if_symptomatic)]
daily[, p_baseline_infected := as.factor(p_baseline_infected)]
daily[, p_external_infection_per_day := as.factor(
    p_external_infection_per_day)]
# Shorter name for behavioural_infectivity_multiple_if_symptomatic:
daily[, ppe_effect := behavioural_infectivity_multiple_if_symptomatic]
daily[, behavioural_infectivity_multiple_if_symptomatic := NULL]


# =============================================================================
# Analyses
# =============================================================================

if (FALSE) {
    sink(RESULTS_FILENAME)

    s1 <- totals %>%
        group_by(
            appointment_type, clinicians_meet_each_other,
            ppe_effect,
            p_baseline_infected,
            p_external_infection_per_day
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


    # These analyses don't take account of time (day), and just use final totals:
    m1 <- lm(
        n_people_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            ppe_effect,
        data = totals)
    print(m1)
    print(anova(m1))

    m2 <- lm(
        n_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            ppe_effect,
        data = totals)
    print(m2)
    print(anova(m2))

    m3 <- lm(
        n_contacts ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            ppe_effect,
        data = totals)
    print(m3)
    print(anova(m3))

    m4 <- lm(
        n_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day,
        data = totals[ppe_effect == 1])
    print(m4)
    print(anova(m4))

    m5 <- lm(
        n_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day,
        data = totals[ppe_effect == 0.1])
    print(m5)
    print(anova(m5))

    sink()
}


# =============================================================================
# Plotting functions
# =============================================================================

common_elements <- list(
    theme_bw(),
    scale_colour_manual(values = c("blue", "red")),
    scale_fill_manual(values = c("blue", "red")),
    scale_linetype_manual(values = c("dotted", "solid")),
    scale_size_manual(values = c(0.5, 2)),
    scale_y_continuous(limits = c(0, NA))
)


make_infected_plot <- function(data, y_varname, errbar_varname,
                               title, with_legend = TRUE,
                               y_axis_title = NULL) {
    yval <- data[[y_varname]]
    errbarval <- data[[errbar_varname]]
    p <- (
        ggplot(data, aes(x = day, group = group)) +
        geom_ribbon(
            aes(
                ymin = yval - errbarval,
                ymax = yval + errbarval,
                fill = appointment_type
            ),
            alpha = 0.25
        ) +
        geom_line(aes_string(
            y = y_varname,
            colour = "appointment_type",
            linetype = "clinicians_meet_each_other",
            size = "ppe_effect"
        )) +
        common_elements +
        theme(legend.position = ifelse(with_legend, "bottom", "none")) +
        ylab(y_axis_title) +
        ggtitle(title)
    )
    return(p)
}


make_clinician_plot <- function(data, title, with_legend = TRUE,
                                y_axis_title = NULL) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_n_clinicians_infected",
        errbar_varname = "errbar_n_clinicians_infected",
        title = title,
        with_legend = with_legend,
        y_axis_title = y_axis_title
    ))
}


make_people_plot <- function(data, title, with_legend = TRUE,
                             y_axis_title = NULL) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_n_people_infected",
        errbar_varname = "errbar_n_people_infected",
        title = title,
        with_legend = with_legend,
        y_axis_title = y_axis_title
    ))
}


make_contacts_plot <- function(data, title = "#Contacts") {
    return(
        ggplot(
            data,
            aes(x = day, y = mean_n_contacts, group = group,
                colour = appointment_type,
                linetype = clinicians_meet_each_other,
                size = ppe_effect)
        ) +
        geom_line() +
        common_elements +
        theme(legend.position = "none") +
        ggtitle(title)
    )
}


# =============================================================================
# Plots
# =============================================================================

# Choose error bars:
errbar_func <- sem
# errbar_func <- sd

# Faceting on this doesn't work well -- scales too disparate:
# daily <- daily[ppe_effect == 0.1]
plotdata <- daily %>%
    group_by(
        appointment_type, clinicians_meet_each_other,
        ppe_effect,
        p_baseline_infected,
        p_external_infection_per_day,
        group,
        day
    ) %>%
    summarise(
        mean_n_clinicians_infected = mean(n_clinicians_infected),
        errbar_n_clinicians_infected = errbar_func(n_clinicians_infected),

        mean_n_people_infected = mean(n_people_infected),
        errbar_n_people_infected = errbar_func(n_people_infected),

        mean_n_contacts = mean(n_contacts)
    ) %>%
    as.data.table()

plotdata_a <- plotdata[p_baseline_infected == 0.01 &
                       p_external_infection_per_day == 0]
plotdata_b <- plotdata[p_baseline_infected == 0.01 &
                       p_external_infection_per_day == 0.02]
plotdata_c <- plotdata[p_baseline_infected == 0.05 &
                       p_external_infection_per_day == 0]
plotdata_d <- plotdata[p_baseline_infected == 0.05 &
                       p_external_infection_per_day == 0.02]

p1a <- make_clinician_plot(plotdata_a, "Baseline 1%, external 0%",
                           y_axis_title = "Number of CLINICIANS infected")
p1b <- make_clinician_plot(plotdata_b, "Baseline 1%, external 2%")
p1c <- make_clinician_plot(plotdata_c, "Baseline 5%, external 0%")
p1d <- make_clinician_plot(plotdata_d, "Baseline 5%, external 2%")

p2a <- make_people_plot(plotdata_a, "Baseline 1%, external 0%",
                        y_axis_title = "Number of PEOPLE infected")
p2b <- make_people_plot(plotdata_b, "Baseline 1%, external 2%")
p2c <- make_people_plot(plotdata_c, "Baseline 5%, external 0%")
p2d <- make_people_plot(plotdata_d, "Baseline 5%, external 2%")

#p3a <- make_contacts_plot(plotdata_a)
#p3b <- make_contacts_plot(plotdata_b)

fig <- (
    (p1a | p1c | p1b | p1d) /
    (p2a | p2c | p2b | p2d) /
    # (p3a | p3b) +
    plot_layout(guides = "collect")
)

ggsave(FIGURE_FILENAME, fig, width = 40, height = 30, units = "cm")


# =============================================================================
# Half-life-type thoughts
# =============================================================================

p_infected <- function(t, half_life = 1) {
    # Inverse survival curve.
    1 - 0.5^(t / half_life)
}
# plot(p_infected, 0, 0.4)  # approximately linear in the range y=0 to y=0.2
