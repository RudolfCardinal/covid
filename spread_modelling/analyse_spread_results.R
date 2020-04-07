#!/usr/bin/env Rscript
# By Rudolf Cardinal, March 2020. GPL 3+ licence; see LICENSE.txt.

# =============================================================================
# Libraries
# =============================================================================

library(conflicted)  # make conflicts explicit
library(car)  # for Anova()
library(data.table)
# library(ez)
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

# Ensure correct order of factor levels
APPOINTMENT_TYPE_LEVELS <- c("remote", "clinic", "home_visit")
COLOURS_APPOINTMENT_TYPES <- c("black", "blue", "red")
BOOLEAN_LEVELS <- c("False", "True")

# For log-transforming values that can be zero:
LOG_OFFSET <- 1  # for log10(x + 1)

if (!exists("totals")) {
    cat(paste0("Loading data (1) from ", TOTALS_FILENAME, "\n"))
    totals <- data.table(read.csv(TOTALS_FILENAME))

    cat("Transforming data (1)...\n")

    totals[, appointment_type := factor(appointment_type,
                                        levels = APPOINTMENT_TYPE_LEVELS)]
    totals[, clinicians_meet_each_other := factor(clinicians_meet_each_other,
                                                  levels = BOOLEAN_LEVELS)]
    # Shorter name for behavioural_infectivity_multiple_if_symptomatic:
    totals[, sx_behav_effect := as.factor(
        behavioural_infectivity_multiple_if_symptomatic)]
    totals[, behavioural_infectivity_multiple_if_symptomatic := NULL]
    totals[, p_baseline_infected := as.factor(p_baseline_infected)]
    totals[, p_external_infection_per_day := as.factor(
        p_external_infection_per_day)]

    # Proportions:
    totals[, prop_people_infected := n_people_infected / n_people]
    totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
    totals[, prop_patients_infected := n_patients_infected / n_patients]
    totals[, prop_family_infected := n_family_infected / n_family]

    # Logs:
    totals[, log10_people_infected := log10(n_people_infected + LOG_OFFSET)]
    totals[, log10_clinicians_infected := log10(n_clinicians_infected + LOG_OFFSET)]
}

if (!exists("daily")) {
    cat(paste0("Loading data (2) from ", DAILY_FILENAME, "\n"))
    daily <- data.table(read.csv(DAILY_FILENAME))

    cat("Transforming data (2)...\n")

    daily[, appointment_type := factor(appointment_type,
                                       levels = APPOINTMENT_TYPE_LEVELS)]
    daily[, clinicians_meet_each_other := factor(clinicians_meet_each_other,
                                                 levels = BOOLEAN_LEVELS)]
    # Shorter name for behavioural_infectivity_multiple_if_symptomatic:
    daily[, sx_behav_effect := as.factor(
        behavioural_infectivity_multiple_if_symptomatic)]
    daily[, behavioural_infectivity_multiple_if_symptomatic := NULL]
    daily[, p_baseline_infected := as.factor(p_baseline_infected)]
    daily[, p_external_infection_per_day := as.factor(
        p_external_infection_per_day)]

    setkeyv(daily, c(
        # Control settings
        "appointment_type",
        "clinicians_meet_each_other",
        "sx_behav_effect",
        "p_baseline_infected",
        "p_external_infection_per_day",
        # Run details
        "iteration",
        "day"
    ))
    daily <- daily %>%
        group_by(
            appointment_type,
            clinicians_meet_each_other,
            sx_behav_effect,
            p_baseline_infected,
            p_external_infection_per_day,
            iteration
        ) %>%
        mutate(
            cum_n_people_infected = cumsum(n_people_infected),
            cum_n_clinicians_infected = cumsum(n_clinicians_infected),
            cum_n_patients_infected = cumsum(n_patients_infected),
            cum_n_family_infected = cumsum(n_family_infected)
        ) %>%
        as.data.table()

    daily[, group := paste0(
        "AT", appointment_type, "_",
        "CM", as.integer(clinicians_meet_each_other), "_",
        "SIMS", sx_behav_effect, "_",
        "BL", p_baseline_infected, "_",
        "EXT", p_external_infection_per_day
    )]

}


# =============================================================================
# Analyses
# =============================================================================
# Reasonable candidates for the dependent variable include:
# - proportion of people infected (range 0-1, directly proportional to
#   number infected since the denominator is essentially constant)
# - log number of people infected.
# The main thing is in the interpretation of interactions.
# An interaction for the proportion/raw number implies non-additivity, but we
# expect infection to be a multiplicative process, so that may not be so
# surprising. An interaction for log(#infected) implies non-additivity of logs,
# i.e. deviation from multiplicative effects.
# The number of infected clinicians can be 0, so we need an offset; see
# LOG_OFFSET.

PERFORM_ANALYSIS <- TRUE
OUTPUT_COLWIDTH <- 120

LINEBREAK_1 <- paste(c(rep("=", 79), "\n"), collapse="")
LINEBREAK_2 <- paste(c(rep("-", 79), "\n"), collapse="")
LINEBREAK_3 <- paste(c(rep("~", 79), "\n"), collapse="")

CONTRASTS_FOR_LEVEL_DIFFS <- c(unordered = "contr.treatment", ordered = "contr.poly")
# ... the default, from getOption("contrasts")
# ... good level naming but not right contrasts for type III SS

CONTRASTS_FOR_TYPE_III_SS <- c(unordered = "contr.sum", ordered = "contr.poly")
# ... sum-to-zero contrasts, for type III sums of squares
# ... see https://www.rdocumentation.org/packages/car/versions/3.0-7/topics/Anova


write_title <- function(title, subtitle = FALSE,
                        append = TRUE, filename = RESULTS_FILENAME)
{
    spacer <- ifelse(subtitle, LINEBREAK_2, LINEBREAK_1)
    sink(filename, append = append)
    cat("\n",
        "\n",
        spacer,
        title, "\n",
        spacer, sep = "")
    sink()
}


write_output <- function(x, append = TRUE, filename = RESULTS_FILENAME)
{
    x_name <- deparse(substitute(x))  # fetch the variable name passed in
    write_title(x_name, append = append, filename = filename, subtitle = TRUE)
    sink(filename, append = TRUE)
    print(x)
    cat(LINEBREAK_3)
    sink()
}


write_lm_and_anova <- function(formula, data, append = TRUE,
                               filename = RESULTS_FILENAME) {
    old_contrasts <- getOption("contrasts")
    sink(filename, append = append)

    cat(LINEBREAK_2)
    cat("\nLinear model [using contr.treatment] for:\n")
    print(formula)
    options(contrasts = CONTRASTS_FOR_LEVEL_DIFFS)
    print(lm(formula = formula, data = data))

    cat("\nANOVA with type III sums of squares [using contr.sum] for:\n")
    print(formula)
    options(contrasts = CONTRASTS_FOR_TYPE_III_SS)
    print(Anova(lm(formula = formula, data = data), type = "III"))
    cat(LINEBREAK_3)

    sink()
    options(contrasts = old_contrasts)
}


if (PERFORM_ANALYSIS) {
    cat("Performing analyses...\n")

    tmp_width <- getOption("width")
    options(width = OUTPUT_COLWIDTH)

    write_title("Summary", append = FALSE)
    s1 <- totals %>%
        group_by(
            appointment_type,
            clinicians_meet_each_other,
            sx_behav_effect,
            p_baseline_infected,
            p_external_infection_per_day
        ) %>%
        summarise(
            mean_prop_people_infected = mean(prop_people_infected),
            mean_prop_clinicians_infected = mean(prop_clinicians_infected),
            mean_prop_patients_infected = mean(prop_patients_infected),
            mean_prop_family_infected = mean(prop_family_infected),
            mean_n_contacts = mean(n_contacts)
        ) %>%
        as.data.table()
    write_output(s1)
    daily_long <- daily %>%
        gather(key = who,
               value = n_infected,
               cum_n_people_infected,
               cum_n_clinicians_infected,
               cum_n_patients_infected,
               cum_n_family_infected) %>%
        mutate(
            who = dplyr::recode(
                who,
                cum_n_people_infected = "people",
                cum_n_clinicians_infected = "clinicians",
                cum_n_patients_infected = "patients",
                cum_n_family_infected = "family"
            )
        ) %>%
        as.data.table()

    # These analyses don't take account of time (day), and just use final
    # totals:

    # -------------------------------------------------------------------------
    # Whole population infections
    # -------------------------------------------------------------------------
    write_title("Whole population infection")
    write_title("Whole population: full model", subtitle = TRUE)
    write_lm_and_anova(
        log10_people_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            sx_behav_effect,
        data = totals
    )
    write_title("Whole population: mean comparisons", subtitle = TRUE)
    write_output(mean(totals[appointment_type == "remote", prop_people_infected]))
    write_output(mean(totals[appointment_type == "clinic", prop_people_infected]))
    write_output(mean(totals[appointment_type == "home_visit", prop_people_infected]))
    write_output(
        mean(totals[appointment_type == "home_visit", prop_people_infected]) -
        mean(totals[appointment_type == "remote", prop_people_infected])
    )
    write_title(paste0(
        "Whole population: highly simplified model ",
        "(illustrates smaller 'bad' (+) effect of 'bad behaviour' ",
        "(sx_behav_effect1) at 'bad infection rate' ",
        "(p_external_infection_per_day0.02)", subtitle = TRUE))
    write_lm_and_anova(
        log10_people_infected ~
            p_external_infection_per_day *
            sx_behav_effect,
        data = totals
    )

    # -------------------------------------------------------------------------
    # Clinician infections
    # -------------------------------------------------------------------------
    write_title("Clinician infection")
    write_lm_and_anova(
        log10_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            sx_behav_effect,
        data = totals
    )

    write_title("Clinician infection: external infection == 0", subtitle = TRUE)
    # Do appointment_type * clinicians_meet_each_other interactions persist
    # in each "external infection" condition?
    write_lm_and_anova(
        log10_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            sx_behav_effect,
        data = totals[p_external_infection_per_day == 0]
    )
    write_title("Clinician infection: external infection == 0.02", subtitle = TRUE)
    write_lm_and_anova(
        log10_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            sx_behav_effect,
        data = totals[p_external_infection_per_day == 0.02]
    )

    # Effects of clinicians meeting up:

    write_title("Clinician infection: clinicians meeting up", subtitle = TRUE)
    meetups1 <- s1 %>%
        select(
            mean_prop_clinicians_infected,
            appointment_type,
            clinicians_meet_each_other,
            p_baseline_infected,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        mutate(
            p_baseline_infected := as.numeric(as.character(p_baseline_infected))
        ) %>%
        pivot_wider(
            names_prefix = "clinicians_meet_",
            names_from = clinicians_meet_each_other,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    meetups1[, no_meet_fraction_of_extra := (
        (clinicians_meet_False - p_baseline_infected) /
        (clinicians_meet_True - p_baseline_infected)
    )]
    meetups2 <- meetups1 %>%
        group_by(appointment_type) %>%
        summarise(mean_no_meet_fraction = mean(no_meet_fraction_of_extra)) %>%
        as.data.table()
    meetups3 <- meetups1 %>%
        group_by(appointment_type, p_external_infection_per_day) %>%
        summarise(mean_no_meet_fraction = mean(no_meet_fraction_of_extra)) %>%
        as.data.table()
    # NB OK to use unweighted means like this, as all groups have equal sizes
    # etc.

    write_output(meetups1)
    write_output(meetups2)
    write_output(meetups3)

    # Effects of appointment type:

    write_title("Clinician infection: appointment types", subtitle = TRUE)
    apptypes1 <- s1 %>%
        select(
            mean_prop_clinicians_infected,
            appointment_type,
            clinicians_meet_each_other,
            p_baseline_infected,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        mutate(
            p_baseline_infected := as.numeric(as.character(p_baseline_infected))
        ) %>%
        pivot_wider(
            names_prefix = "apptype_",
            names_from = appointment_type,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    apptypes1[, clinic_fraction_of_extra_vs_home_visit := (
        (apptype_clinic - p_baseline_infected) /
        (apptype_home_visit - p_baseline_infected)
    )]
    apptypes1[, remote_fraction_of_extra_vs_home_visit := (
        (apptype_remote - p_baseline_infected) /
        (apptype_home_visit - p_baseline_infected)
    )]
    apptypes2 <- apptypes1 %>%
        group_by(p_external_infection_per_day) %>%
        summarise(
            mean_clinic_fraction_of_extra_vs_home_visit = mean(clinic_fraction_of_extra_vs_home_visit),
            mean_remote_fraction_of_extra_vs_home_visit = mean(remote_fraction_of_extra_vs_home_visit),
        ) %>%
        as.data.table()
    apptypes3 <- apptypes1 %>%
        group_by(p_external_infection_per_day, clinicians_meet_each_other) %>%
        summarise(
            mean_clinic_fraction_of_extra_vs_home_visit = mean(clinic_fraction_of_extra_vs_home_visit),
            mean_remote_fraction_of_extra_vs_home_visit = mean(remote_fraction_of_extra_vs_home_visit),
        ) %>%
        as.data.table()

    write_output(apptypes1)
    write_output(apptypes2)
    write_output(apptypes3)

    # Effects of sx_behav_effect:

    write_title("Clinician infection: sx_behav_effect", subtitle = TRUE)
    behav1 <- s1 %>%
        select(
            mean_prop_clinicians_infected,
            appointment_type,
            clinicians_meet_each_other,
            p_baseline_infected,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        mutate(
            p_baseline_infected := as.numeric(as.character(p_baseline_infected))
        ) %>%
        pivot_wider(
            names_prefix = "behav_",
            names_from = sx_behav_effect,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    behav1[, ppe_fraction_of_no_ppe := (
        (behav_0.1 - p_baseline_infected) /
        (behav_1 - p_baseline_infected)
    )]
    behav2 <- behav1 %>%
        group_by(appointment_type) %>%
        summarise(mean_ppe_fraction_of_no_ppe = mean(ppe_fraction_of_no_ppe)) %>%
        as.data.table()
    behav3 <- behav1 %>%
        group_by(appointment_type, p_external_infection_per_day) %>%
        summarise(mean_ppe_fraction_of_no_ppe = mean(ppe_fraction_of_no_ppe)) %>%
        as.data.table()

    write_output(behav1)
    write_output(behav2)
    write_output(behav3)

    # Done

    options(width = tmp_width)  # restore
    rm(tmp_width)
}


# =============================================================================
# Plotting functions
# =============================================================================

COMMON_PLOT_ELEMENTS <- list(
    theme_bw(),
    scale_colour_manual(values = COLOURS_APPOINTMENT_TYPES),
    scale_fill_manual(values = COLOURS_APPOINTMENT_TYPES),
    scale_linetype_manual(values = c("dotted", "solid")),
    scale_size_manual(values = c(0.5, 2)),
    theme(
        plot.title = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold")
    )
)


make_infected_plot <- function(data, y_varname, errbar_varname,
                               title, y_axis_title = NULL) {
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
            size = "sx_behav_effect"
        )) +
        COMMON_PLOT_ELEMENTS +
        scale_y_log10() +
        ylab(y_axis_title) +
        xlab("Day") +
        ggtitle(title)
    )
    return(p)
}


make_clinician_plot <- function(data, title, y_axis_title = NULL) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_cum_n_clinicians_infected",
        errbar_varname = "errbar_cum_n_clinicians_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


make_people_plot <- function(data, title, y_axis_title = NULL) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_cum_n_people_infected",
        errbar_varname = "errbar_cum_n_people_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


#make_contacts_plot <- function(data, title = "#Contacts") {
#    return(
#        ggplot(
#            data,
#            aes(x = day, y = mean_cum_n_contacts, group = group,
#                colour = appointment_type,
#                linetype = clinicians_meet_each_other,
#                size = sx_behav_effect)
#        ) +
#        geom_line() +
#        COMMON_PLOT_ELEMENTS +
#        theme(legend.position = "none") +
#        ggtitle(title)
#    )
#}


# =============================================================================
# Plots
# =============================================================================
# One choice is of plotting on a log scale (allowing all conditions to be
# shown) or having separate plots on a linear scale. The log scale is better
# overall, and also allows faceting.
# Another choice is whether to plot raw numbers on a log scale (1, 10, 100,
# 1000) or log10 numbers (1, 2, 3, 4) -- the former is clearer
# presentationally.

cat("Making figures...\n")

# Choose error bars:
errbar_func <- miscstat$half_confidence_interval_t  # +/- 0.5 * 95% CI, i.e. 95% CI shown

plotdata <- daily %>%
    group_by(
        appointment_type, clinicians_meet_each_other,
        sx_behav_effect,
        p_baseline_infected,
        p_external_infection_per_day,
        group,
        day
    ) %>%
    summarise(
        mean_cum_n_clinicians_infected = mean(cum_n_clinicians_infected),
        errbar_cum_n_clinicians_infected = errbar_func(cum_n_clinicians_infected),
        mean_cum_n_people_infected = mean(cum_n_people_infected),
        errbar_cum_n_people_infected = errbar_func(cum_n_people_infected)
        # mean_cum_n_contacts = mean(cum_n_contacts)
    ) %>%
    mutate(
        p_baseline_infected_numeric := as.numeric(as.character(p_baseline_infected)),
        p_external_infection_per_day_numeric := as.numeric(as.character(p_external_infection_per_day)),
        infection_group := factor(
            paste0("Baseline ", p_baseline_infected_numeric * 100, "%, ",
                   "external ", p_external_infection_per_day_numeric * 100, "%"),
            levels = c(
                "Baseline 1%, external 0%",
                "Baseline 5%, external 0%",
                "Baseline 1%, external 2%",
                "Baseline 5%, external 2%"
            )
        )
    ) %>%
    as.data.table()

p1 <- (
    make_people_plot(plotdata, "A. All people",
                     y_axis_title = "Cumulative #people infected") +
    facet_grid(. ~ infection_group)
)
p2 <- (
    make_clinician_plot(plotdata, "B. Clinicians",
                        y_axis_title = "Cumulative #clinicians infected") +
    facet_grid(. ~ infection_group)
)

fig <- (
    (
        p1 |
        p2
    ) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
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

cat("Done.\n")
