# WarpII Contribution Guidelines

This document sets out conventions for the use of C++, git, and documentation
for the WarpII project.
When adding to this list, consider the following question:

> Is it possible to enforce this rule via an automated check?

If so, create the automated check instead.
A simple example of a guideline which is enforced by automation is the `Pragmas` test,
which enforces that all `*.h` files begin with the line `#pragma once`.

## Project usage and architecture

- **Use tagged releases.** Users of WarpII should primarily be running tagged release
  versions of the code, or a build of a commit on `main`. Running simulations from
  feature branch builds is a major exception, not the norm.
- **No user code.** That means there will never be a `user_runs` directory for users of the code
  to put their own scripts or data analysis. All usage of WarpII should be done outside the
  source tree, and no artifacts of runs should ever be committed.
- **No problem-specific code.** We want to avoid committing code to WarpII that is for the purpose of
  setting up a specific problem. Some aspects of simulation are particularly vulnerable to 
  this anti-pattern: mesh generation and initial and boundary condition definitions.
  It is anticipated that the input file mechanism suffices for simple function definitions for initial
  and boundary conditions.
  More complex use cases should be handled through the [extension](https://uw-computational-plasma-group.github.io/warpii/extension_tutorial.html) mechanism.
- **Eschew dependencies.** Dependencies add substantial friction to any software project, and
  add enormous amounts of friction to C++ projects. WarpII has one enormous dependency, deal.II.
  Fortunately, deal.II is unusually well-packaged and works very cleanly. We should be very
  suspicious of adding further dependencies.
  Note that Python is also a dependency that we would very much like to avoid.

## Git

- **Merge early and often.** Feature branches should be short-lived and merged frequently.
  If a feature is not done, it can still be merged behind a feature flag, or simply
  not exposed in the public API. Merging early and often supports the **Use tagged releases**
  principle: the earlier that work is merged, the more useful things are available to use
  on the `main` branch.
- **Rebase before merging feature branches.** We want to maintain a clean history in git
  free of work-in-progress commits. Running serious simulations with a feature branch
  build of WarpII is highly discouraged. If it is absolutely necessary, preserving the git history of that
  branch is the responsibility of the user, not of the WarpII project.
  This is a corollary of the **No user code** principle: git history that only exists to
  ensure reproducibility of a user's simulation is a violation of the principle.
  And it should be no burden if you were already following the **Use tagged releases** principle.

