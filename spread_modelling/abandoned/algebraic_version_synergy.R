library(data.table)
library(EpiDynamics)
library(tidyverse)


# =============================================================================
# IGNORE. Iterative approach, via a difference equation
# =============================================================================
#
# t:    time
# n:    size of population
# x:    cumulative number infected at time t
# x0:   number infected at t = 0

next_x <- function(x, r, n) {
    # x(t+1) = x(t) + r(n - x(t))
    # cumulatively_infected = previously_infected + infectivity * available
    return(
        x + r * (n - x)
    )
}


x_t_iterative <- function(t, x0 = 10, n = 1000, r = 0.1) {
    stopifnot(t >= 0)
    stopifnot(x0 >= 0)
    stopifnot(n > 0)
    # Time 0:
    x <- x0
    # Other times:
    if (t > 0) {
        for (i in 1:t) {
            x <- next_x(x, r, n)
        }
    }
    return(x)
}
# x_t_iterative(t = 0)
# x_t_iterative(t = 10)


x_t_iterative_vector <- Vectorize(x_t_iterative)


plot(x_t_iterative_vector, 0, 100, col = "red")


# =============================================================================
# IGNORE. Differential equation version of the above...
# =============================================================================
# https://math.stackexchange.com/questions/145523/links-between-difference-and-differential-equations
# https://math.stackexchange.com/questions/1493004/difference-equation-discrete-time-to-differential-equation-continuous-time

NOTES <- "

    x(t+1) = x(t) + r(n - x(t))
    x(t+1) = x(t) + rn - rx(t)
    x(t+1) = (1 - r)x(t) + rn
    x(t+1) - (1 - r)x(t) - rn = 0

    ...

"

# ABANDONED -- see SEIR model in analyse_spread_results.R
