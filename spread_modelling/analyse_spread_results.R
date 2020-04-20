#!/usr/bin/env Rscript
# By Rudolf Cardinal, March 2020. GPL 3+ licence; see LICENSE.txt.
#
# PLOTTING
#
# One choice is of plotting on a log scale (allowing all conditions to be
# shown) or having separate plots on a linear scale. The log scale is better
# overall, and also allows faceting.
# Another choice is whether to plot raw numbers on a log scale (1, 10, 100,
# 1000) or log10 numbers (1, 2, 3, 4) -- the former is clearer
# presentationally.
#
# ANALYSIS
#
# Similarly: analyse untransformed values, log-transformed values as
# log10(y + 1), or use a GLM with a log link function?
#
# The GLM with a log link function:
#
# - analyses mean(values) by exponentiating linear predictors
# - The error distribution is flexibly defined and determined by the variance
#   function. For example:
#
#       f <- gaussian()
#       f$variance
#
# - see also
#   https://www.theanalysisfactor.com/count-models-understanding-the-log-link-function/
# - see also FENG, Changyong, Hongyue WANG, Naiji LU, Tian CHEN, Hua HE,
#   Ying LU, and Xin M. TU. ‘Log-Transformation and Its Implications for Data
#   Analysis’. Shanghai Archives of Psychiatry 26, no. 2 (April 2014): 105–9.
#   https://doi.org/10.3969/j.issn.1002-0829.2014.02.009.
# - see also
#   https://statmath.wu.ac.at/courses/heather_turner/glmCourse_001.pdf
# - https://stats.stackexchange.com/questions/47840/linear-model-with-log-transformed-response-vs-generalized-linear-model-with-log
#
# The LM:
#
# - analyses mean(log(values)) with linear predictors
#
# Analysis of deviance for GLM:
#
# - https://online.stat.psu.edu/stat504/node/157/
# - http://www.bbk.ac.uk/ems/faculty/brooms/teaching/SMDA/SMDA-07.pdf
#
# Practicalities:
#
# - Experiment 1: 48 * 2000 runs
# - Experiment 1: 60 * 2000 runs

DEMO_GLM_VS_LOGLINEAR <- '

LOG_OFFSET <- 1
e2totals[, n_patients_per_household_factor := as.factor(n_patients_per_household)]
e1totals[, log10_clinicians_infected := log10(n_clinicians_infected + LOG_OFFSET)]
e2totals[, n_clinicians_infected_plus1 := n_clinicians_infected + 1]
options(contrasts = CONTRASTS_FOR_LEVEL_DIFFS)

glm_model <- glm(
    n_clinicians_infected_plus1 ~ n_patients_per_household_factor *
                                  clinicians_meet_each_other,
    data = e2totals, family = gaussian(link = "log")
)
print(glm_model)

llm_model <- lm(
    log10_clinicians_infected ~ n_patients_per_household_factor *
                                clinicians_meet_each_other,
    data = e2totals)
print(llm_model)

plain_lm_model <- lm(
    n_clinicians_infected_plus1 ~ n_patients_per_household_factor *
                                  clinicians_meet_each_other,
    data = e2totals)
print(plain_lm_model)

e2totals[, glm_predicted := predict(glm_model, type = "response")]
# ... must do type = "response" to get it into the right scale!
# e2totals[, llm_predicted := predict(llm_model)]
# ... fails, bizarrely! But this works:
e2totals$llm_predicted <- predict(llm_model)
e2totals[, ten_to_llm_predicted := 10 ^ llm_predicted]
e2totals[, plain_lm_predicted := predict(plain_lm_model)]
demoplot <- (
    ggplot(e2totals,
           aes(x = n_patients_per_household,
               linetype = clinicians_meet_each_other)) +
    geom_point(aes(y = n_clinicians_infected_plus1,
                   colour = clinicians_meet_each_other),
               position = "jitter", alpha = 0.1) +
    scale_y_log10() +
    geom_line(aes(y = plain_lm_predicted),
              colour = "green", size = 3) +
    geom_line(aes(y = ten_to_llm_predicted),
              colour = "red", size = 0.5) +  # NB MISMATCH
    geom_line(aes(y = glm_predicted),
              colour = "blue", size = 0.5)  # matches plain linear prediction
)
print(demoplot)

# Thus, the GLM with log link is the best one to use.
# Then, what we really want is a Poisson/log-link GLM.

'


# =============================================================================
# Warnings become errors
# =============================================================================

options(warn = 2)


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
source("https://egret.psychol.cam.ac.uk/rlib/misclang.R")
source("https://egret.psychol.cam.ac.uk/rlib/miscstat.R")


# =============================================================================
# Directories, filenames
# =============================================================================

THIS_DIR <- miscfile$current_script_directory()
RESULTS_DIR <- file.path(THIS_DIR, "results")  # simulation results
CACHE_DIR <- file.path(THIS_DIR, "cache")
OUTPUT_DIR <- file.path(THIS_DIR, "output")  # R output

EXP1_TOTALS_FILENAME <- file.path(RESULTS_DIR, "exp1_totals.csv")
EXP1_DAILY_FILENAME <- file.path(RESULTS_DIR, "exp1_daily.csv")
EXP1B_TOTALS_FILENAME <- file.path(RESULTS_DIR, "exp1b_totals.csv")
EXP1B_DAILY_FILENAME <- file.path(RESULTS_DIR, "exp1b_daily.csv")
EXP2_TOTALS_FILENAME <- file.path(RESULTS_DIR, "exp2_totals.csv")

EXP1_TOTALS_CACHE_FILENAME <- file.path(CACHE_DIR, "exp1_totals.rds")
EXP1_DAILY_CACHE_FILENAME <- file.path(CACHE_DIR, "exp1_daily.rds")
EXP2_TOTALS_CACHE_FILENAME <- file.path(CACHE_DIR, "exp2_totals.rds")
RESULTS_FILENAME <- file.path(OUTPUT_DIR, "results.txt")
FIGURE_2_FILENAME <- file.path(OUTPUT_DIR, "figure_2.pdf")
FIGURE_3_FILENAME <- file.path(OUTPUT_DIR, "figure_3.pdf")
FIGURE_4_FILENAME <- file.path(OUTPUT_DIR, "figure_4.pdf")


# =============================================================================
# Other constants
# =============================================================================

# Ensure correct order of factor levels
APPOINTMENT_TYPE_LEVELS <- c("remote", "clinic", "home_visit")
APPOINTMENT_TYPE_LABELS <- c("Remote visit (RV)",
                             "Patient only (PO)",
                             "Family contact (FC)")
COLOURS_APPOINTMENT_TYPES <- c("black", "blue", "red")
LINETYPES_CLINICIANS_MEET <- c("dashed", "solid")
SIZES_SX_BEHAV_EFFECT <- c(0.75, 1.5)
BOOLEAN_LEVELS <- c("False", "True")

HIGH_EXTERNAL_INFECTION <- 0.02

LABEL_SX_BEHAV_EFFECT <- "Sx behav. effect"
LABEL_CLINICIANS_MEET <- "Clinicians meet each other"
LABEL_APPOINTMENT_TYPE <- "Appointment type"

CORE_PLOT_ELEMENTS <- list(
    theme_bw(),
    theme(
        plot.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold")
    )
)
COMMON_PLOT_ELEMENTS <- c(CORE_PLOT_ELEMENTS, list(
    scale_colour_manual(values = COLOURS_APPOINTMENT_TYPES),
    scale_fill_manual(values = COLOURS_APPOINTMENT_TYPES),
    scale_linetype_manual(values = LINETYPES_CLINICIANS_MEET),
    scale_size_manual(values = SIZES_SX_BEHAV_EFFECT)
))

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
# ... and http://egret.psychol.cam.ac.uk/statistics/R/anova.html
# ... UNUSED; we will use type I with a fully balanced design


# =============================================================================
# Data in
# =============================================================================

get_totals <- function(filename) {
    cat(paste0("Loading totals data from ", filename, "\n"))
    e1totals <- data.table(read.csv(filename))

    cat("Transforming totals data...\n")

    e1totals[, appointment_type := factor(appointment_type,
                                          levels = APPOINTMENT_TYPE_LEVELS)]
    e1totals[, clinicians_meet_each_other := factor(clinicians_meet_each_other,
                                                    levels = BOOLEAN_LEVELS)]
    # Shorter name for behavioural_infectivity_multiple_if_symptomatic:
    e1totals[, sx_behav_effect :=
        as.factor(behavioural_infectivity_multiple_if_symptomatic)]
    e1totals[, behavioural_infectivity_multiple_if_symptomatic := NULL]
    e1totals[, p_baseline_infected := as.factor(p_baseline_infected)]
    e1totals[, p_external_infection_per_day :=
        as.factor(p_external_infection_per_day)]

    # Proportions:
    e1totals[, prop_people_infected := n_people_infected / n_people]
    e1totals[, prop_clinicians_infected := n_clinicians_infected / n_clinicians]
    e1totals[, prop_patients_infected := n_patients_infected / n_patients]
    e1totals[, prop_family_infected := n_family_infected / n_family]

    return(e1totals)
}


get_exp1_totals <- function() {
    return(rbind(
        get_totals(EXP1_TOTALS_FILENAME),
        get_totals(EXP1B_TOTALS_FILENAME)
    ))
}
get_exp2_totals <- function() {
    return(get_totals(EXP2_TOTALS_FILENAME))
}


get_daily <- function(filename) {
    cat(paste0("Loading daily data from ", filename, "\n"))
    e1daily <- data.table(read.csv(filename))

    cat("Transforming daily data...\n")

    e1daily[, appointment_type := factor(appointment_type,
                                         levels = APPOINTMENT_TYPE_LEVELS)]
    e1daily[, clinicians_meet_each_other := factor(clinicians_meet_each_other,
                                                   levels = BOOLEAN_LEVELS)]
    # Shorter name for behavioural_infectivity_multiple_if_symptomatic:
    e1daily[, sx_behav_effect :=
        as.factor(behavioural_infectivity_multiple_if_symptomatic)]
    e1daily[, behavioural_infectivity_multiple_if_symptomatic := NULL]
    e1daily[, p_baseline_infected := as.factor(p_baseline_infected)]
    e1daily[, p_external_infection_per_day :=
        as.factor(p_external_infection_per_day)]

    setkeyv(e1daily, c(
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
    e1daily <- e1daily %>%
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

    e1daily[, group := paste0(
        "AT-", appointment_type, "_",
        "CM-", as.integer(clinicians_meet_each_other), "_",
        "SIMS-", sx_behav_effect, "_",
        "BL-", p_baseline_infected, "_",
        "EXT-", p_external_infection_per_day
    )]

    return(e1daily)
}


get_exp1_daily <- function() {
    return(rbind(
        get_daily(EXP1_DAILY_FILENAME),
        get_daily(EXP1B_DAILY_FILENAME)
    ))
}


# Use an exists() check to save reloading from disk every time we re-source().
if (!exists("e1totals")) {
    e1totals <- misclang$load_rds_or_run_function(EXP1_TOTALS_CACHE_FILENAME,
                                                  get_exp1_totals)
}
if (!exists("e1daily")) {
    e1daily <- misclang$load_rds_or_run_function(EXP1_DAILY_CACHE_FILENAME,
                                                 get_exp1_daily)
}
if (!exists("e2totals")) {
    e2totals <- misclang$load_rds_or_run_function(EXP2_TOTALS_CACHE_FILENAME,
                                                  get_exp2_totals)
}

# Sanity checks

stopifnot(
    mean(e1daily[day == 60]$cum_n_people_infected) ==
    mean(e1totals$n_people_infected)
)

# stop("Loaded.")


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


write_glm_and_anova <- function(formula, data, append = TRUE,
                                family = poisson(link = "log"),
                                filename = RESULTS_FILENAME) {
    old_contrasts <- getOption("contrasts")
    sink(filename, append = append)

    cat(LINEBREAK_2)
    cat("\nGeneralized linear model with log link [using contr.treatment] for:\n")
    print(formula)
    print(family)
    options(contrasts = CONTRASTS_FOR_LEVEL_DIFFS)
    model_for_coeffs <- glm(formula = formula, data = data, family = family)
    print(model_for_coeffs)

    cat("\nANOVA with type III sums of squares of that model [redone using contr.sum] for:\n")
    print(formula)
    options(contrasts = CONTRASTS_FOR_TYPE_III_SS)
    model_for_type_III <- glm(formula = formula, data = data, family = family)
    print(car::Anova(model_for_type_III, type = "III"))
    cat(LINEBREAK_3)

    # -------------------------------------------------------------------------
    # Ways not to run this analysis:
    # -------------------------------------------------------------------------

    # print(car::Anova(model_for_coeffs, type = "III"))  # NB DIFFERENT; WRONG;
    # based on model with wrong contrasts; see "Warnings" section in
    # ?car::Anova.

    # drop1(model_for_type_III, ~., test = "F")
    # ... produces a warning with family = poisson()
    #     ("F test assumes 'quasipoisson' family")

    # -------------------------------------------------------------------------
    # Alternative ways:
    # -------------------------------------------------------------------------

    # drop1(model_for_type_III, ~., test = "Chisq")

    # -------------------------------------------------------------------------
    # See also
    # -------------------------------------------------------------------------

    # https://stat.ethz.ch/pipermail/r-sig-mixed-models/2016q1/024465.html
    # ... car::Anova() can be used for lme4::glmer() too, if the contrast
    #     coding is set correctly when the model is created.
    #     ... as, for lmer4 models, can lmerTest::anova().
    #     ... see https://cran.r-project.org/web/packages/lme4/lme4.pdf

    sink()
    options(contrasts = old_contrasts)
}


cat("Performing analyses...\n")

tmp_width <- getOption("width")
options(width = OUTPUT_COLWIDTH)

# -------------------------------------------------------------------------
# Experiment 1
# -------------------------------------------------------------------------

write_title("Experiment 1", append = FALSE)
write_title("Exp 1: Summary")
s1 <- e1totals %>%
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
daily_long <- e1daily %>%
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
# e1totals:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Whole population infections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
write_title("Exp 1: Whole population infection")
write_title("Exp 1: Whole population: full model", subtitle = TRUE)
write_glm_and_anova(
    n_people_infected ~
        appointment_type *
        clinicians_meet_each_other *
        p_baseline_infected *
        p_external_infection_per_day *
        sx_behav_effect,
    data = e1totals
)
write_title("Exp 1: Whole population: mean comparisons", subtitle = TRUE)
write_output(mean(e1totals[appointment_type == "remote", prop_people_infected]))
write_output(mean(e1totals[appointment_type == "clinic", prop_people_infected]))
write_output(mean(e1totals[appointment_type == "home_visit", prop_people_infected]))
write_output(
    mean(e1totals[appointment_type == "home_visit", prop_people_infected]) -
    mean(e1totals[appointment_type == "remote", prop_people_infected])
)
write_title(
    paste0(
        "Exp 1: Whole population: highly simplified model ",
        "(illustrates smaller 'bad' (+) effect of 'bad behaviour' ",
        "(sx_behav_effect1) at 'bad infection rate' ",
        "(p_external_infection_per_day", HIGH_EXTERNAL_INFECTION, ")"
    ), subtitle = TRUE)
write_glm_and_anova(
    n_people_infected ~
        p_external_infection_per_day *
        sx_behav_effect,
    data = e1totals
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Clinician infections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
write_title("Exp 1: Clinician infection")
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        clinicians_meet_each_other *
        p_baseline_infected *
        p_external_infection_per_day *
        sx_behav_effect,
    data = e1totals
)

# Effects of appointment type:

write_title("Exp 1: Clinician infection: appointment types", subtitle = TRUE)
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

write_title(paste0(
    "Exp 1: Clinician infection: external infection == 0 ",
    "(Do appointment_type * clinicians_meet_each_other interactions persist ",
    "in each 'external infection' condition?)"
), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        clinicians_meet_each_other *
        p_baseline_infected *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == 0]
)
write_title(paste("Exp 1: Clinician infection: external infection == ",
                  HIGH_EXTERNAL_INFECTION), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        clinicians_meet_each_other *
        p_baseline_infected *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == HIGH_EXTERNAL_INFECTION]
)

# Effects of clinicians meeting up:

write_title("Exp 1: Clinician infection: clinicians meeting up", subtitle = TRUE)
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

# Effects of sx_behav_effect:

write_title("Exp 1: Clinician infection: sx_behav_effect", subtitle = TRUE)
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
behav4 <- behav1 %>%
    group_by(clinicians_meet_each_other) %>%
    summarise(mean_ppe_fraction_of_no_ppe = mean(ppe_fraction_of_no_ppe)) %>%
    as.data.table()

write_output(behav1)
write_output(behav2)
write_output(behav3)
write_output(behav4)

write_title(paste0(
    "Exp 1, clinician infection : SIMPLIFIED MODEL: ",
    "Effect of sx_behav_effect (and appointment_type) ",
    "with external infection at 0%"), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == 0]
)
write_title(
    paste0(
        "Exp 1, clinician infection : SIMPLIFIED MODEL: ",
        "Effect of sx_behav_effect (and appointment_type) ",
        "with external infection at ", 100 * HIGH_EXTERNAL_INFECTION, "%"
    ), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == HIGH_EXTERNAL_INFECTION]
)
write_title(paste0(
    "Exp 1, clinician infection : SIMPLIFIED MODEL: ",
    "Effect of sx_behav_effect (and clinicians_meet_each_other) ",
    "with external infection at 0%"), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        clinicians_meet_each_other *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == 0]
)
write_title(
    paste0(
        "Exp 1, clinician infection : SIMPLIFIED MODEL: ",
        "Effect of sx_behav_effect (and clinicians_meet_each_other) ",
        "with external infection at ", 100 * HIGH_EXTERNAL_INFECTION, "%"
    ), subtitle = TRUE)
write_glm_and_anova(
    n_clinicians_infected ~
        clinicians_meet_each_other *
        sx_behav_effect,
    data = e1totals[p_external_infection_per_day == HIGH_EXTERNAL_INFECTION]
)


# -------------------------------------------------------------------------
# Experiment 2
# -------------------------------------------------------------------------

e2totals[, n_patients_per_household_factor :=
    as.factor(n_patients_per_household)]

write_title("Experiment 2")
write_title("Exp 2: Whole population infection")
write_glm_and_anova(
    n_people_infected ~
        appointment_type *
        clinicians_meet_each_other *
        n_patients_per_household_factor,
    data = e2totals
)
write_title("Exp 2: Clinician infection")
write_glm_and_anova(
    n_clinicians_infected ~
        appointment_type *
        clinicians_meet_each_other *
        n_patients_per_household_factor,
    data = e2totals
)

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------

options(width = tmp_width)  # restore
rm(tmp_width)


# =============================================================================
# Plotting functions
# =============================================================================

make_infected_by_day_plot <- function(data, y_varname, errbar_varname,
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
        labs(
            colour = LABEL_APPOINTMENT_TYPE,
            fill = LABEL_APPOINTMENT_TYPE,
            linetype = LABEL_CLINICIANS_MEET,
            size = LABEL_SX_BEHAV_EFFECT
        ) +
        COMMON_PLOT_ELEMENTS +
        scale_y_log10() +
        ylab(y_axis_title) +
        xlab("Day") +
        ggtitle(title)
    )
    return(p)
}


make_clinician_by_day_plot <- function(data, title, y_axis_title = NULL) {
    return(make_infected_by_day_plot(
        data = data,
        y_varname = "mean_cum_n_clinicians_infected",
        errbar_varname = "errbar_cum_n_clinicians_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


make_people_by_day_plot <- function(data, title, y_axis_title = NULL) {
    return(make_infected_by_day_plot(
        data = data,
        y_varname = "mean_cum_n_people_infected",
        errbar_varname = "errbar_cum_n_people_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


#make_contacts_by_day_plot <- function(data, title = "#Contacts") {
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


make_exp2_plot <- function(data, y_varname, errbar_varname,
                           title, y_axis_title = NULL) {
    yval <- data[[y_varname]]
    errbarval <- data[[errbar_varname]]
    xval <- sort(unique(data$n_patients_per_household))
    p <- (
        ggplot(data, aes(x = n_patients_per_household,
                         group = group)) +
        geom_ribbon(
            aes(
                ymin = yval - errbarval,
                ymax = yval + errbarval
            ),
            alpha = 0.25
        ) +
        geom_line(aes_string(
            y = y_varname,
            colour = "appointment_type",
            linetype = "clinicians_meet_each_other"
        )) +
        labs(
            colour = LABEL_APPOINTMENT_TYPE,
            linetype = LABEL_CLINICIANS_MEET
        ) +
        geom_point(aes_string(y = y_varname)) +
        COMMON_PLOT_ELEMENTS +
        scale_x_continuous(
            breaks = xval,
            minor_breaks = NULL,
            labels = xval
        ) +
        scale_y_log10() +
        ylab(y_axis_title) +
        xlab("#Patients per household") +
        ggtitle(title)
    )
    return(p)
}


make_clinician_exp2_plot <- function(data, title, y_axis_title = NULL) {
    return(make_exp2_plot(
        data = data,
        y_varname = "mean_n_clinicians_infected",
        errbar_varname = "errbar_n_clinicians_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


make_people_exp2_plot <- function(data, title, y_axis_title = NULL) {
    return(make_exp2_plot(
        data = data,
        y_varname = "mean_n_people_infected",
        errbar_varname = "errbar_n_people_infected",
        title = title,
        y_axis_title = y_axis_title
    ))
}


# =============================================================================
# Figures
# =============================================================================

cat("Making figures...\n")

# Choose error bars:
errbar_func <- miscstat$half_confidence_interval_t  # +/- 0.5 * 95% CI, i.e. 95% CI shown

# -----------------------------------------------------------------------------
# Figure 2: Experiment 1
# -----------------------------------------------------------------------------

plotdata2 <- e1daily %>%
    group_by(
        appointment_type,
        clinicians_meet_each_other,
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
# Do the following in a separate step, or you get the warning:
# Error in mutate_impl(.data, dots, caller_env()) :
#   (converted from warning) Unequal factor levels: coercing to character
plotdata2 <- plotdata2 %>%
    mutate(
        infection_sxbehav_group := factor(
            paste0(as.character(infection_group), "\n",
                   LABEL_SX_BEHAV_EFFECT, " ", sx_behav_effect)
        )
    ) %>%
    as.data.table()
# More precise names for appointment type, for plotting:
plotdata2[, appointment_type := factor(appointment_type,
                                       levels = APPOINTMENT_TYPE_LEVELS,
                                       labels = APPOINTMENT_TYPE_LABELS)]

f2p1 <- (
    make_people_by_day_plot(plotdata2, "A. All people",
                            y_axis_title = "Cumulative #people infected") +
    facet_grid(. ~ infection_sxbehav_group)
)
f2p2 <- (
    make_clinician_by_day_plot(plotdata2, "B. Clinicians",
                               y_axis_title = "Cumulative #clinicians infected") +
    facet_grid(. ~ infection_sxbehav_group)
)

fig2 <- (
    (
        f2p1 /
        f2p2 +
        plot_layout(heights = c(6, 12))
    ) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
)
ggsave(FIGURE_2_FILENAME, fig2, width = 40, height = 40, units = "cm")


# -----------------------------------------------------------------------------
# Figure 3: Experiment 2
# -----------------------------------------------------------------------------

plotdata3 <- e2totals %>%
    group_by(
        appointment_type,
        clinicians_meet_each_other,
        n_patients_per_household
    ) %>%
    summarise(
        mean_n_clinicians_infected = mean(n_clinicians_infected),
        errbar_n_clinicians_infected = errbar_func(n_clinicians_infected),
        mean_n_people_infected = mean(n_people_infected),
        errbar_n_people_infected = errbar_func(n_people_infected)
    ) %>%
    mutate(
        group = paste0(
            "AT-", appointment_type, "_",
            "CM-", as.integer(clinicians_meet_each_other)
        )
    ) %>%
    as.data.table()
plotdata3[, appointment_type := factor(appointment_type,
                                       levels = APPOINTMENT_TYPE_LEVELS,
                                       labels = APPOINTMENT_TYPE_LABELS)]

f3p1 <- (
    make_people_exp2_plot(plotdata3, "A. All people",
                          y_axis_title = "Total #people infected")
)
f3p2 <- (
    make_clinician_exp2_plot(plotdata3, "B. Clinicians",
                             y_axis_title = "Total #clinicians infected")
)

fig3 <- (
    (
        f3p1 |
        f3p2
    ) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
)
ggsave(FIGURE_3_FILENAME, fig3, width = 20, height = 10, units = "cm")


# =============================================================================
# Half-life-type thoughts
# =============================================================================

p_infected <- function(t, half_life = 1) {
    # Inverse survival curve.
    1 - 0.5^(t / half_life)
}
# plot(p_infected, 0, 0.4)  # approximately linear in the range y=0 to y=0.2


# =============================================================================
# SEIR model, exploring the nature of synergy in manipulations.
# =============================================================================

seir_total <- function(
        final_time = 1000,
        transmission_rate = 0.1,
        initial_proportion_exposed = 0.01,
        rate_from_exposed_to_infectious = 1/5,
        recovery_rate = 1/7,
        per_capita_death_rate_and_pop_birth_rate = 0,
        initial_proportion_susceptible = 1 - initial_proportion_exposed,
        initial_proportion_infectious = 0,
        initial_proportion_recovered = 0,
        verbose = FALSE) {
    stopifnot(
        initial_proportion_susceptible +
        initial_proportion_exposed +
        initial_proportion_infectious +
        initial_proportion_recovered == 1
    )
    seir_info <- EpiDynamics::SEIR(
        pars = c(
            mu = per_capita_death_rate_and_pop_birth_rate,
            beta = transmission_rate,
            sigma = rate_from_exposed_to_infectious,
            gamma = recovery_rate
        ),
        init = c(
            S = initial_proportion_susceptible,
            E = initial_proportion_exposed,
            I = initial_proportion_infectious,
            R = initial_proportion_recovered
        ),
        time = c(
            0,  # initial time,
            final_time
        )
    )
    if (verbose) {
        print(seir_info)
    }
    results <- seir_info$results
    # Special case: if final_time == 0, you get two duplicate rows, not one. So:
    final_values <- tail(results[results$time == final_time, ], 1)
    final_s <- final_values$S  # final proportion susceptible
    cumulative_infected <- 1 - final_s  # final proportion infected (exposed + infectious + recovered)
    return(cumulative_infected)
}
# seir_total(final_time = 10000)


seir_total_vector <- Vectorize(seir_total)
# plot(seir_total_vector, 0, 1000)


seirdata <- data.table(expand.grid(
    transmission_rate = seq(0.00, 0.5, by = 0.001),
    initial_proportion_exposed = c(0.01, 0.05)
))
seirdata[, initial_proportion_exposed_factor := factor(
    initial_proportion_exposed,
    levels = c(0.01, 0.05),
    labels = c("1%", "5%")
)]
seirdata[, cumulative_infected := seir_total_vector(
    transmission_rate = transmission_rate,
    initial_proportion_exposed = initial_proportion_exposed)]


# The SEIR model, EpiDynamics::SEIR, is program 2.6 in Keeling & Rohani (2007),
# as per http://www.modelinginfectiousdiseases.org/ -> Chapter 2 -> Program 2.6
# So we can get the exact meaning of EpiDynamics::SEIR parameters from there.
# Their notes:
KEELING_ROHANI_NOTES <- '

We now introduce a refinement to the SIR model (Program 2.2) which takes into
account a latent period. The process of transmission often occurs due to an
initial inoculation with a very small number of pathogen units (e.g., a few
bacterial cells or virions). A period of time then ensues during which the
pathogen reproduces rapidly within the host, relatively unchallenged by the
immune system. During this stage, pathogen abundance is too low for active
transmission to other susceptible hosts, and yet the pathogen is present.
Hence, the host cannot be categorized as susceptible, infectious, or recovered;
we need to introduce a new category for these individuals who are infected but
not yet infectious. These individuals are referred to as Exposed and are
represented by the variable E in SEIR models.

    dS/dt = μ - (βI + μ)S

    dE/dt = βSI - (μ + σ)E

    dI/dt = σE - (μ + γ)I

    dR/dt = γI - μR

Parameters

μ	is the per capita death rate, and the population level birth rate.
β	is the transmission rate and incorporates the encounter rate between
    susceptible and infectious individuals together with the probability of
    transmission.
γ	is called the removal or recovery rate, though often we are more interested
    in its reciprocal (1/γ) which determines the average infectious period.
σ   is the rate at which individuals move from the exposed to the infectious
    classes. Its reciprocal (1/σ) is the average latent (exposed) period.

S(0)	is the initial proportion of the population that are susceptible.
E(0)	is the initial proportion of the population that are exposed (infected
        but not infectious)
I(0)    is the initial proportion of the population that are infectious

All rates are specified in days.

Requirements.

All parameters must be positive, and S(0)+E(0)+I(0) ≤ 1.

'
# So... we were correct to think that sigma = 1/(average latent period) and
# gamma = 1/(average infectious period).
SEIRC_LEVELS <- c("S", "E", "I", "R", "C")  # C = cumulative infected
SEIRC_COLOURS <- c(S="black", E="orange", I="red", R="blue", C="green")
seir_specimen <- data.table(EpiDynamics::SEIR(
    pars = c(
        mu = 0,  # birth/death rate
        beta = 0.15,    # specimen transmission rate, as above
        sigma = 1/5,    # rate, exposed -> infectious
        gamma = 1/7     # rate, infectious -> recovery
    ),
    init = c(
        S = 0.99,   # initial proportion susceptible
        E = 0.01,   # initial proportion exposed
        I = 0,      # initial proportion infectious
        R = 0       # initial proportion recovered
    ),
    time = seq(0, 1000)
)$results)
seir_specimen[, C := E + I + R]  # cumulative infected
seir_specimen_plotdata <- seir_specimen %>%
    pivot_longer(
        cols = c("S", "E", "I", "R", "C"),
        names_to = "state"
    ) %>%
    dplyr::filter(
        state != "S",
        state != "C",
        time <= 500
    ) %>%
    as.data.table()
seir_specimen_plotdata[, state := factor(state, levels = SEIRC_LEVELS)]
seir_specimen_plot <- (
    ggplot(seir_specimen_plotdata,
           aes(x = time, y = value,
               colour = state, linetype = state, shape = state)) +
    geom_line() +
    scale_colour_manual(values = SEIRC_COLOURS) +
    # scale_shape_manual() +
    CORE_PLOT_ELEMENTS
)
# print(seir_specimen_plot)


fig4 <- (
    ggplot(seirdata, aes(x = transmission_rate, y = cumulative_infected,
                         colour = initial_proportion_exposed_factor)) +
    geom_line() +
    # geom_point() +
    scale_y_log10() +
    scale_colour_manual(values = c("blue", "red")) +
    labs(colour = "Initial proportion exposed") +
    xlab("Transmission rate, β") +
    ylab("Cumulative proportion infected") +
    ggtitle("Deterministic SEIR model") +
    CORE_PLOT_ELEMENTS
)
ggsave(FIGURE_4_FILENAME, fig4, width = 15, height = 15, units = "cm")


# =============================================================================
# Done.
# =============================================================================

cat("Done.\n")
