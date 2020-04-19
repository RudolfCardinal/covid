# https://octave.org/doc/v4.2.1/Script-Files.html
# https://octave.org/doc/v4.4.1/Ordinary-Differential-Equations.html

pkg load symbolic
syms r n t x

dx_dt = r * (n - x)
xfunc = int(dx_dt, t)
