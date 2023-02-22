/*

   000_simplest_md version 0.0.1
   copyright 2012 anthony beardsworth costa
   email anthony.costa@gmail.com

   this is the simplest md program imaginable. it is intended
   for instruction purposes for the first-time md programmer.
   it computes the hamiltonian trajectory (constant energy)
   for a preterbed equilateral 12-6 lennard-jones (LJ) trimer.
   it outputs the kinetic, potential, and total energy, to a
   properties file. it outputs the trajectory in xyz format.
   absolutely nothing fancy is attempted. it has no periodic
   boundary conditions, cutoffs, bonds, angles, dihedrals,
   electrostatics, etc. importantly, it has no parallelization.

   this is the base program in a series of simple md programs.
   each new program will have an incremented index (here, that
   index is 000), and will have a few new features, so that we
   build our understanding of doing more complex things
   incrementally. each incremented program will list the new
   features in detail.

   compilation:
   gcc 000_simplest_md.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* this function gives the proper 0-indexing for column-major
   matrix storage in contiguous arrays. typically, (i)
   corresponds to the atom number and (j) corresponds to the
   dimension. ld is the leading dimension, which in this case
   is the number of rows, so typically the number of atoms.
   the leading dimension can be thought of as the offset */
#define idx(__i, __j, __ld) (((__j) * (__ld)) + (__i))

/* memory calloc (allocate on heap stack and zero) and free
   functions. these macros allow the user to calloc any type
   they like very easily */
#define memory_calloc_array(__ptr, __type, __size) \
  { \
    __ptr = (__type*) calloc ((__size), sizeof (__type)); \
    if (__ptr == NULL) \
      exit (EXIT_FAILURE); \
  }

#define memory_free_array(__ptr) \
  { \
    free (__ptr); \
  }

/* returns the gradient for the 12-6 lennard-jones (LJ) potential,
   with no cutoffs or other complications */
double
gradient_return (const double r_t, const double epsilon, \
  const double sigma_t)
{
  double z = sigma_t / r_t;
  double u = z * z * z;
  return (24 * epsilon * u * (1 - 2 * u) / r_t);
}

/* returns the potential energy for the 12-6 lennard jones (LJ)
   potential, with no cutoffs or other complications */
double
potential_return (const double r_t, const double epsilon, \
  const double sigma_t)
{
  double z = sigma_t / r_t;
  double u = z * z * z;
  return (- 4 * epsilon * u * (1 - u));
}

/* builds the forces given the current system configuration.
   this is the simplest possible force function. no cutoffs,
   no periodic boundary conditions, etc. takes as argument a
   list of interactions and updates forces and potentials. */
void
force (const int natm, const int ndim, const double *positions, \
  double *forces, const int npar, const double *interactions, \
  double *potentials)
{
  /* zero the old forces before building new ones. this is necessary
     because each configurational degree of freedom will have
     incremental forces added to it for each pairwise interaction. */
  memset (&forces[0], 0, ndim * natm * sizeof (double));

  /* ij is the counter over all the pairs in the interaction list.
     atm_a, atm_b are atom indices for the pairwise calculation.
     dim is the index for the dimension.
     r_ij_t is the square of the distance between each atom pair.
     dim_distance gives the distance in each dimension between pairs. */
  int ij, atm_a, atm_b, dim;
  double r_ij_t, dim_distance[ndim];
  for (ij = 0; ij < npar; ij++)
  {
    /* get the atom indices for this interaction */
    atm_a = interactions[idx (ij, 0, npar)];
    atm_b = interactions[idx (ij, 1, npar)];

    /* accumulate the square of the distance for each pair */
    r_ij_t = 0.0;
    for (dim = 0; dim < ndim; dim++)
    {
      dim_distance[dim] = positions[idx (atm_a, dim, natm)] - \
        positions[idx (atm_b, dim, natm)];
      r_ij_t += dim_distance[dim] * dim_distance[dim];
    }

    /* get the gradient for square distance with epsilon and
       the square of sigma (and therefore sigma) set to 1
       for the 12-6 LJ potential */
    double gradient, potential;
    gradient = gradient_return (r_ij_t, 1, 1);
    potential = potential_return (r_ij_t, 1, 1);

    /* increment the forces on each atom for this pair. we
       kept the per-dimension distances for a good reason. */
    for (dim = 0; dim < ndim; dim++)
    {
      forces[idx (atm_a, dim, natm)] -= gradient * dim_distance[dim];
      forces[idx (atm_b, dim, natm)] += gradient * dim_distance[dim];
    }

    /* assign the potential for the current pair */
    potentials[ij] = potential;
  }
}

/* these are both steps of the velocity verlet integrator. we
   separate the predictor and corrector steps according to the
   following algorithm.

     (1) v(t+(1/2)dt) = x(t) + v(t)*dt + (1/2)*(f(t)/m)*dt.
     (2) x(t+dt) = x(t) + v(t+(1/2)dt)*dt.
     (3) update forces to generate f(t+dt).
     (4) v(t+dt)=v(t+(1/2)dt)+(1/2)*(f(t+dt)/m)*dt.

   here, v are the velocities, x are the positions, f are the forces,
   and m are the masses. refined indices (atoms, dimensions) have been
   left off for clarity the predictor performs steps 1 and 2, the
   force function performs step 3, and the corrector performs step 4.
   calling these functions in that order is completed in the run loop.
   by newton's 2nd law, this could be equivalently formulated by deriving
   the accelerations instead of the forces in step 3, thereby eliminating
   the mass dividend in the integrator steps. having the forces stored
   is more convenient for things we will want in the future, hence this
   particular implementation. */
void
predictor (const int natm, const int ndim, const double timestep, \
  const double *masses, double *positions, double *velocities, \
  const double *forces)
{
  /* atm is the atom index.
     dim is the dimension index.
     inx is the matrix-array index given by the macro idx. */
  int atm, dim, inx;
  for (atm = 0; atm < natm; atm++)
    for (dim = 0; dim < ndim; dim++)
    {
      inx = idx (atm, dim, natm);
      velocities[inx] += forces[inx] * timestep / 2 / masses[atm];
      positions[inx] += velocities[inx] * timestep;
    }
}

void
corrector (const int natm, const int ndim, const double timestep, \
  const double *masses, double *velocities, const double *forces)
{
  /* atm is the atom index.
     dim is the dimension index.
     inx is the matrix-array index given by the macro idx. */
  int atm, dim, inx;
  for (atm = 0; atm < natm; atm++)
    for (dim = 0; dim < ndim; dim++)
    {
      inx = idx (atm, dim, natm);
      velocities[inx] += forces[inx] * timestep / 2 / masses[atm];
    }
}

/* this function prints the configuration point in xyz format to
   a supplied file pointer. this is what writes the trajectory. */
void
write_trajectory (const int natm, const int ndim, \
  const double currenttime, FILE *file, const double *positions)
{
  /* atm is the atom index.
     dim is the dimension index. */
  int atm, dim;
  fprintf (file, "%d\n", natm);
  fprintf (file, "Generated by FFEDS -- Time is %f\n", currenttime);
  for (atm = 0; atm < natm; atm++)
  {
    fprintf (file, "LJ");
    for (dim = 0; dim < ndim; dim++)
      fprintf (file, "%16.6f", positions[idx (atm, dim, natm)]);

    fprintf (file, "\n");
  }
}

/* this function returns the kinetic energy for a supplied momentum
   point (masses and velocities). */
double
compute_kinetic_energy (const int natm, const int ndim, \
  const double *masses, const double *velocities)
{
  /* atm is the atom index.
     dim is the dimension index.
     inx is the matrix-array index given by the macro idx.
     kinetic_energy is the accumulating variable to be returned */
  int atm, dim, inx;
  double kinetic_energy = 0.0;
  for (atm = 0; atm < natm; atm++)
    for (dim = 0; dim < ndim; dim++)
    {
      inx = idx (atm, dim, natm);
      kinetic_energy += 0.5 * masses[atm] * velocities[inx] * \
        velocities[inx];
    }

  return (kinetic_energy);
}

/* this function returns the potential energy for a supplied set
   of pairwise potentials via a simple sum. */
double
compute_potential_energy (const int npar, const double *potentials)
{
  /* ij is the interaction index
     potential_energy is the accumulating variable to be returned */
  int ij;
  double potential_energy = 0.0;
  for (ij = 0; ij < npar; ij++)
    potential_energy += potentials[ij];

  return (potential_energy);
}

/* this function returns the total energy for supplied values of
   the kinetic and potential energy. */
double
compute_total_energy (const double kinetic, const double potential)
{
  return (kinetic + potential);
}

/* this function writes a given name/value pair to the supplied
   file pointer. this is used to write properties to the property
   file. */
void
write_property (const char *name, const double value, FILE *file)
{
  fprintf (file, "%20s%16.6f\n", name, value);
}

/* this function is used to generate a message to the user if
   the code exists with an error */
void
exit_error (const char *message)
{
  fprintf (stdout, "%s\n", message);
  exit (EXIT_FAILURE);
}

/* this is the driver function, where the md loop is actually contained.
   it controls all memory and variables locally, and passes everything
   each function needs explicitly. all traditional run-time variables
   are defined here. */
int
main (int argc, char **argv)
{
  /* ndim is the number of dimensions.
     natm is the number of atoms.
     nsteps is the number of md steps to run.
     currentstep keeps track of the number of elapsed steps.
     timestep is time increment for integration. */
  const int ndim = 3;
  const int natm = 3;
  const int nsteps = 10000;
  int currentstep = 0;
  double currenttime = 0.0;
  const double timestep = 0.001;

  /* this initializes the masses array. all masses are set to 1.0 in
     the next loop. atm is the atom index. need one mass per atom. */
  double *masses;
  memory_calloc_array (masses, double, natm);

  /* atm is the atom index */
  int atm;
  for (atm = 0; atm < natm; atm++)
    masses[atm] = 1.0;

  /* this is the positions array, which has dimensions natm (rows) and
     ndim (columns), so it needs ndim * natm elements. */
  double *positions;
  memory_calloc_array (positions, double, ndim * natm);

  /* these are the perturbed 12-6 lennard-jones (LJ) trimer initial
     coordinates. they are hard-coded into the main function here. */
  positions[idx (0, 0, natm)] = 0.5391356726;
  positions[idx (0, 1, natm)] = 0.1106588251;
  positions[idx (0, 2, natm)] = -0.4635601962;
  positions[idx (1, 0, natm)] = -0.5185079933;
  positions[idx (1, 1, natm)] = 0.4850176090;
  positions[idx (1, 2, natm)] = 0.0537084789;
  positions[idx (2, 0, natm)] = 0.0793723207;
  positions[idx (2, 1, natm)] = -0.4956764341;
  positions[idx (2, 2, natm)] = 0.5098517173;

  /* the memory footprint of the velocities and forces arrays are
     the same as those for positions. they are allocated and
     zeroed here. */
  double *velocities;
  memory_calloc_array (velocities, double, ndim * natm);

  double *forces;
  memory_calloc_array (forces, double, ndim * natm);

  /* to compute the potential energy properly, we need some memory
     to keep track of the potential energy for each pariwise
     interaction. since we have no cutoffs, the number of
     pairwise goes as natm(natm-1)/2. here npar is the number of
     these pairwise interactions. note that this is not statically
     coded but derived from the variable natm. from here we generate
     the memory for a list of all interactions and the memory for
     storing the potential for each of these interactions. since all
     interactions are pairwise, the memory for the interactions list
     has npar * 2 elements, whereas the memory for storing the data
     has simply npar elements. */
  double *potentials;
  double *interactions;
  const int npar = (natm * (natm - 1)) / 2;
  memory_calloc_array (potentials, double, npar);
  memory_calloc_array (interactions, double, npar * 2);

  /* we generate the list of interactions by looping over all pairs.
     here, atm_a and atm_b are atom indices.
     ij is the row number of the pair list. */
  int ij = 0;
  int atm_a, atm_b;
  for (atm_a = 0; atm_a < natm - 1; atm_a++)
    for (atm_b = atm_a + 1; atm_b < natm; atm_b++)
    {
      interactions[idx (ij, 0, npar)] = atm_a;
      interactions[idx (ij, 1, npar)] = atm_b;

      /* increment to the next interaction. */
      ij++;
    }

  /* we need an initial set of forces for the first predictor
     integration step in the loop. this is the first force call,
     which generates the forces for the initial configuration
     point. it also generates the first potentials. */
  force (natm, ndim, positions, forces, npar, interactions, potentials);

  /* these files contain data written by the loop. the properties file
     will have things like the kinetic, potential, and total energies
     written to it, while the trajectory file pointer is passed to the
     trajectory function to write the configuration point in xyz format.
     check to see if files already exist so we don't overwrite them. */
  if (!access ("properties.dat", F_OK))
    exit_error ("will not overwrite properties.dat file");
  FILE *properties_file = fopen ("properties.dat", "a");

  if (!access ("trajectory.xyz", F_OK))
    exit_error ("will not overwrite trajectory.xyz file");
  FILE *trajectory_file = fopen ("trajectory.xyz", "a");

  /* during the loop, we will calculate the kinetic, potential, and
     total energies. assign variables for them here */
  double kinetic_energy, potential_energy, total_energy;

  while (currentstep < nsteps)
  {
    /* we start the next step here, so increment the step counter and
       increment the time by the integration time step. */
    currentstep++;
    currenttime += timestep;

    /* this is the velocity verlet procedure. it includes the predictor,
       new force calculation, and corrector steps. see the exhaustive
       description in the comment above the predictor function in this
       source file. */
    predictor (natm, ndim, timestep, masses, positions, velocities, forces);
    force (natm, ndim, positions, forces, npar, interactions, potentials);
    corrector (natm, ndim, timestep, masses, velocities, forces);

    /* now that the positions have been updated to the next step, here
       we write the new phase point to the trajectory file in xyz format. */
    write_trajectory (natm, ndim, currenttime, trajectory_file, positions);

    /* use the compute functions to generate the kinetic, potential, and
       total energies to be written out to the properties file. */
    kinetic_energy = compute_kinetic_energy (natm, ndim, masses, velocities);
    potential_energy = compute_potential_energy (npar, potentials);
    total_energy = compute_total_energy (kinetic_energy, potential_energy);

    /* write the kinetic, potential, and total energies to the properties
       file in a fixed format way using the properties function. */
    write_property ("kinetic_energy", kinetic_energy, properties_file);
    write_property ("potential_energy", potential_energy, properties_file);
    write_property ("total_energy", total_energy, properties_file);
  }

  /* this closes the file pointers for the properties and trajectory
     files, since we are done writing to them. */
  fclose (properties_file);
  fclose (trajectory_file);

  /* this frees all the dynamically allocated memory. */
  memory_free_array (masses);
  memory_free_array (positions);
  memory_free_array (velocities);
  memory_free_array (forces);
  memory_free_array (potentials);
  memory_free_array (interactions);

  /* exit the program, indicating its run was successful. */
  return (EXIT_SUCCESS);
}
