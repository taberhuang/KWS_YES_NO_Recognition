.. _docs-code_reviews:

======================
Code review guidelines
======================
All Pigweed development happens on Gerrit, following the `typical Gerrit
development workflow <http://ceres-solver.org/contributing.html>`_. Consult the
`Gerrit User Guide
<https://gerrit-documentation.storage.googleapis.com/Documentation/2.12.3/intro-user.html>`_
for more information on using Gerrit.

You may add the special address
``gwsq-pigweed@pigweed.google.com.iam.gserviceaccount.com`` as a reviewer to
have Gerrit automatically choose an appropriate person to review your change.

-----------
For authors
-----------

.. _docs-code_reviews-small-changes:

Small changes
=============
Please follow the guidance in `Google's Eng-Practices Small CLs
<https://google.github.io/eng-practices/review/developer/small-cls.html>`_.

Complete changes
================
.. todo-check: disable

In summary, CLs must be complete, tested, and include documentation and unit
tests for new code, bug fixes, and any code changes that merit it. However, to
enable iterative work and small changes, ``TODO`` comments are acceptable. They
must include an explanation of the problem and an action to take.

.. todo-check: enable

We will not take over incomplete changes to avoid shifting our focus. We may
reject changes that do not meet the criteria above.

-------------
For reviewers
-------------

Review speed
============
Follow the advice at `Speed of Code Reviews
<https://google.github.io/eng-practices/review/reviewer/speed.html>`_.  Most
importantly,

* If you are not in the middle of a focused task, **you should do a code review
  shortly after it comes in**.
* **One business day is the maximum time it should take to respond** to a code
  review request (i.e., first thing the next morning).
* If you will not be able to review the change within a business day, comment
  on the change stating so, and reassign to another reviewer if possible.

Attention set management
========================
Remove yourself from the `attention set
<https://gerrit-review.googlesource.com/Documentation/user-attention-set.html>`_
for changes where another person (author or reviewer) must take action before
you can continue to review. You are encouraged, but not required, to leave a
comment when doing so, especially for changes by external contributors who may
not be familiar with our process.

----------------------
Common advice playbook
----------------------
What follows are bite-sized copy-paste-able advice when doing code reviews.
Feel free to link to them from code review comments, too.

.. _docs-code_reviews-playbook-platform-design:

Shared platforms require careful design
=======================================
Pigweed is a platform shared by many embedded projects. This makes contributing
to Pigweed rewarding: your change will help teams around the world! But it also
makes contributing *hard*:

* Edge cases that may not matter for one project can, and eventually will, come
  up in another one.
* Pigweed has many modules that can be used in isolation, but should also work
  together, exhibiting a unified design philosophy and guiding users towards
  safe, scalable patterns.

As a result, Pigweed can't be as nimble as individual embedded projects, and
often needs to engage in more careful design review, either in meetings with
the core team or through :ref:`SEED-0001`. But we're committed to working
through this with you!


.. _docs-code_reviews-playbook-stale-changes:

Stale changes
=============
Sometimes, a change doesn't make it out of the review process: after some
rounds of review, there are unresolved comments from the Pigweed team, but the
author is no longer actively working on the change.

For any change that's not seen activity for 3 months, the Pigweed team will,

#. `File a bug <https://issues.pigweed.dev/issues?q=status:open>`_ for the
   feature or bug that the change was addressing, referencing the change.
#. Mark the change Abandoned in Gerrit.

This does *not* mean the change is rejected! It just indicates no further
action on it is expected. As its author, you should feel free to reopen it when
you have time to work on it again.

Before making or sending major changes or SEEDs, please reach out in our
`chat room <https://discord.gg/M9NSeTA>`_ or on the `mailing list
<https://groups.google.com/forum/#!forum/pigweed>`_ first to ensure the changes
make sense for upstream. We generally go through a design phase before making
large changes. See :ref:`SEED-0001` for a description of this process; but
please discuss with us before writing a full SEED. Let us know of any
priorities, timelines, requirements, and limitations ahead of time.

Gerrit for PRs
==============
We don't currently support GitHub pull requests. All Pigweed development takes
place on `our Gerrit instance <https://pigweed-review.googlesource.com/>`_.
Please resubmit your change there!

See :ref:`docs-contributing` for instructions, and consult the `Gerrit User
Guide
<https://gerrit-documentation.storage.googleapis.com/Documentation/2.12.3/intro-user.html>`_
for more information on using Gerrit.

.. _docs-code_reviews-incomplete-docs-changes:

Docs-Only Changes Do Not Need To Be Complete
============================================
Documentation-only changes should generally be accepted if they make the docs
better or more complete, even if the documentation change itself is incomplete.

.. _docs-code_reviews-experimental:

Experimental repository and where to land code
==============================================
.. _experimental repository: https://pigweed.googlesource.com/pigweed/experimental
.. _main repository: https://pigweed.googlesource.com/pigweed/pigweed
.. _no larger than 500 lines each: https://google.github.io/eng-practices/review/developer/small-cls.html

Pigweed has an `experimental repository`_ which differs from
our `main repository`_ in a couple key ways:

* Code is not expected to become production grade.
* Code review standards are relaxed to allow experimentation.
* In general the value of the code in the repository is the knowledge gained
  from the experiment, not the code itself.
* The repository is minimally maintained, and might contain
  a significant amount of broken code.

.. warning::

   Adding the experimental repository as a dependency is strongly discouraged.
   Pigweed will never provide updates or bug fixes for code that lives in the
   experimental repository.

Good uses of the repo include:

* Experimenting with using an API (e.g. C++20 coroutines) with no plans to
  turn it into production code.
* One-off test programs to gather data.

We would like to avoid large pieces of code being developed in the experimental
repository and then imported into the main repository. If large amounts of code
end up needing to migrate from experimental to main, then it must be landed
incrementally as a series of reviewable patches, typically `no larger than 500
lines each`_. This creates a large code review burden that often results in
poorer reviews. Therefore, if the eventual location of the code will be the
main Pigweed repository, it is **strongly encouraged** that the code be
developed in the **main repository under an experimental flag**.
