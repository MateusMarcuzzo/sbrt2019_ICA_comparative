21/09/2019
WE had to rename the original name files (but they still have their original name inside
as saved_sim)

because the load files with too long file names on Octave.

22/09/2019
renamed the p5k8 experiment to diff_setup1_part4
^This file has different format than the others. Inside the script, we must manage these differences.

07/10/2019
Every .mat file has:
n_samples = [ 256 512 1024 2048 4096]
algorithm_name = ['america','sa4ica','glica']

---

setup1.mat constitutes of the following values as simulation parameters:
n_trials = 50
some_primes = [ 2 3 5 7]
n_sources = [ 2 3 4 5]

---

diff_setup1_part1.mat:
n_trials = 50
primes evaluated
[2 3]
n_sources evaluated
[6 7]

---

diff_setup1_part2.mat:
n_trials = 50
primes evaluated
[5]
n_sources evaluated
[6]

---

diff_setup1_part3.mat
n_trials = 50
primes evaluated
[5]
n_sources evaluated
[7]

---

diff_setup1_part4.mat
will handle later (TODO), but we did handle the P5K8 problem here. It's Daniel's file.

---

diff2_setup1.mat:
n_trials = 50
primes evaluated
[2 3]
n_sources evaluated
[8]

---------

If we could compile everyone, with the singular exception of n_trials = 40 on p5k8 experiment, we would have the following setup:

some_primes = [ 2 3 5 7]
n_sources = [2 3 4 5 6 7 8]
# p7 and k = 6,7,8, not evaluated. Too long to run