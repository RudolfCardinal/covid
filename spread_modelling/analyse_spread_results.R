#!/usr/bin/env Rscript

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

if (!exists("totals")) {
    cat("Loading data (1)...\n")

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
    totals[, p_baseline_infected_numeric := p_baseline_infected]
    totals[, p_baseline_infected := as.factor(p_baseline_infected)]
    totals[, p_external_infection_per_day := as.factor(
        p_external_infection_per_day)]

    totals[, prop_people_infected := n_people_infected / n_people]
    totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
    totals[, prop_patients_infected := n_patients_infected / n_patients]
    totals[, prop_family_infected := n_family_infected / n_family]
}

if (!exists("daily")) {
    cat("Loading data (2)...\n")

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
    daily[, p_baseline_infected_numeric := p_baseline_infected]
    daily[, p_baseline_infected := as.factor(p_baseline_infected)]
    daily[, p_external_infection_per_day := as.factor(
        p_external_infection_per_day)]

    daily[, group := paste0(
        "AT", appointment_type, "_",
        "CM", as.integer(clinicians_meet_each_other), "_",
        "SIMS", sx_behav_effect, "_",
        "EXT", p_external_infection_per_day
    )]
}


# =============================================================================
# Analyses
# =============================================================================

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
    cat(spacer,
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
    cat("Linear model [using contr.treatment] for:\n")
    print(formula)
    options(contrasts = CONTRASTS_FOR_LEVEL_DIFFS)
    print(lm(formula = formula, data = data))

    cat("ANOVA with type III sums of squares [using contr.sum] for:\n")
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
            p_baseline_infected_numeric,
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
        ) %>%
        as.data.table()

    # These analyses don't take account of time (day), and just use final totals:

    # -------------------------------------------------------------------------
    # Whole population infections
    # -------------------------------------------------------------------------
    write_title("Whole population infection")
    write_lm_and_anova(
        prop_people_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            p_external_infection_per_day *
            sx_behav_effect,
        data = totals
    )
    write_output(mean(totals[appointment_type == "remote", prop_people_infected]))
    write_output(mean(totals[appointment_type == "clinic", prop_people_infected]))
    write_output(mean(totals[appointment_type == "home_visit", prop_people_infected]))
    write_output(
        mean(totals[appointment_type == "home_visit", prop_people_infected]) -
        mean(totals[appointment_type == "remote", prop_people_infected])
    )

    # -------------------------------------------------------------------------
    # Clinician infections
    # -------------------------------------------------------------------------
    write_title("Clinician infection")
    write_lm_and_anova(
        prop_clinicians_infected ~
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
        prop_clinicians_infected ~
            appointment_type *
            clinicians_meet_each_other *
            p_baseline_infected *
            sx_behav_effect,
        data = totals[p_external_infection_per_day == 0]
    )
    write_title("Clinician infection: external infection == 0.02", subtitle = TRUE)
    write_lm_and_anova(
        prop_clinicians_infected ~
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
            p_baseline_infected_numeric,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        pivot_wider(
            names_prefix = "clinicians_meet_",
            names_from = clinicians_meet_each_other,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    meetups1[, no_meet_fraction_of_extra := (
        (clinicians_meet_False - p_baseline_infected_numeric) /
        (clinicians_meet_True - p_baseline_infected_numeric)
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
            p_baseline_infected_numeric,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        pivot_wider(
            names_prefix = "apptype_",
            names_from = appointment_type,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    apptypes1[, clinic_fraction_of_extra_vs_home_visit := (
        (apptype_clinic - p_baseline_infected_numeric) /
        (apptype_home_visit - p_baseline_infected_numeric)
    )]
    apptypes1[, remote_fraction_of_extra_vs_home_visit := (
        (apptype_remote - p_baseline_infected_numeric) /
        (apptype_home_visit - p_baseline_infected_numeric)
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
            p_baseline_infected_numeric,
            p_external_infection_per_day,
            sx_behav_effect
        ) %>%
        pivot_wider(
            names_prefix = "behav_",
            names_from = sx_behav_effect,
            values_from = mean_prop_clinicians_infected
        ) %>%
        as.data.table()
    behav1[, ppe_fraction_of_no_ppe := (
        (behav_0.1 - p_baseline_infected_numeric) /
        (behav_1 - p_baseline_infected_numeric)
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

common_elements <- list(
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
                               title, with_legend = TRUE,
                               y_axis_title = NULL, ymax = NA) {
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
        common_elements +
        scale_y_continuous(limits = c(0, ymax)) +
        theme(legend.position = ifelse(with_legend, "bottom", "none")) +
        ylab(y_axis_title) +
        ggtitle(title)
    )
    return(p)
}


make_clinician_plot <- function(data, title, with_legend = TRUE,
                                y_axis_title = NULL, ymax = 18) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_n_clinicians_infected",
        errbar_varname = "errbar_n_clinicians_infected",
        title = title,
        with_legend = with_legend,
        y_axis_title = y_axis_title,
        ymax = ymax
    ))
}


make_people_plot <- function(data, title, with_legend = TRUE,
                             y_axis_title = NULL, ymax = 12000) {
    return(make_infected_plot(
        data = data,
        y_varname = "mean_n_people_infected",
        errbar_varname = "errbar_n_people_infected",
        title = title,
        with_legend = with_legend,
        y_axis_title = y_axis_title,
        ymax = ymax
    ))
}


make_contacts_plot <- function(data, title = "#Contacts") {
    return(
        ggplot(
            data,
            aes(x = day, y = mean_n_contacts, group = group,
                colour = appointment_type,
                linetype = clinicians_meet_each_other,
                size = sx_behav_effect)
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

cat("Making figures...\n")

# Choose error bars:
# errbar_func <- sem
# errbar_func <- sd
errbar_func <- miscstat$half_confidence_interval_t  # +/- 0.5 * 95% CI, i.e. 95% CI shown

# Faceting on this doesn't work well -- scales too disparate:
# daily <- daily[sx_behav_effect == 0.1]
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

p1a <- make_clinician_plot(plotdata_a, "A. Baseline 1%, external 0%",
                           y_axis_title = "Number of CLINICIANS infected",
                           ymax = 1)
p1b <- make_clinician_plot(plotdata_c, "B. Baseline 5%, external 0%",
                           ymax = 3)
p1c <- make_clinician_plot(plotdata_b, "C. Baseline 1%, external 2%")
p1d <- make_clinician_plot(plotdata_d, "D. Baseline 5%, external 2%")

p2a <- make_people_plot(plotdata_a, "E. Baseline 1%, external 0%",
                        y_axis_title = "Number of PEOPLE infected")
p2b <- make_people_plot(plotdata_c, "G. Baseline 5%, external 0%")
p2c <- make_people_plot(plotdata_b, "F. Baseline 1%, external 2%")
p2d <- make_people_plot(plotdata_d, "H. Baseline 5%, external 2%")

#p3a <- make_contacts_plot(plotdata_a)
#p3b <- make_contacts_plot(plotdata_b)

fig <- (
    (p1a | p1b | p1c | p1d) /
    (p2a | p2b | p2c | p2d) /
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

cat("Done.\n")
