.. _docs-rollers:

=======
Rollers
=======

-----------------
What is a roller?
-----------------
A roller is an infra job that updates the revision of a pinned project (or
version of a pinned CIPD package) in a repository with submodules or an
Android Repo Tool manifest. The Pigweed Infra team generally maintains the
logic and configuration of rollers, while various project and/or package
owners across are responsible for keeping them green.

-----------
Our rollers
-----------
Rollers from Pigweed into downstream projects can be seen on the
`Pigweed Roll Console <https://ci.chromium.org/p/pigweed/g/pigweed.pigweed.roll/console>`_.
Many more rollers are visible when logged in using an @google.com account than
when not.

Rollers specific to individual downstream projects can be found by browsing
the :ref:`builder visualization <docs-builder-viz>` and looking under the
"roll" columns.

--------------------
How do rollers work?
--------------------
Project rollers will poll Gitiles periodically (often every three hours, but
this is configurable) for new commits to watched
repositories. Package rollers will poll CIPD at their configured frequencies
for new packages which match their watched refs (e.g. 'latest').

When a new change is detected, luci-scheduler emits a 'trigger' for the
roller. The 'trigger' does not necessarily invoke the roller immediately as
luci-scheduler is configured to only allow a single job of each roller to be
running at once. It will batch triggers until there is no job running, then
create a roller job with the properties of the latest trigger in the batch.

Once the roller job begins, it creates a change which updates the relevant
project or package pin(s) and attempts to submit the change via CQ. If the CQ
run fails, then the roller will abandon that change. If the CQ run succeeds, the
change is submitted and the roller succeeds.

On most hosts, rollers vote on the ``Bot-Commit`` label which bypasses the
``Code-Review`` requirement. On other hosts, rollers vote on ``Code-Review``,
or Gerrit is configured to not enforce the ``Code-Review`` requirement for
changes uploaded by the roller.

In the event of a CQ failure, if the roller attempts to re-roll the exact same
revision or package(s), it will re-use the existing change and only re-run the
failing portions of CQ.

-------------------------------
How do I fix a roller breakage?
-------------------------------
Most rollers will fix themselves. Failing rollers are periodically rerun by
the
`"rerunner" builders <https://ci.chromium.org/ui/p/pigweed/g/rerunner/builders>`_,
so any flaky tests or temporary failures should resolve themselves.

However, if you are a Googler on a team using Pigweed's infrastructure, you
should be able to manually trigger a roller by finding it in the Builder Viz
and clicking the "scheduler" link. (You may need to log in to
luci-scheduler.) This is batched with all other 'triggers' as to not disrupt
the normal roller flow, but will allow you to restart the roll process.

-----------------------------------------------
How do I disable a roller? Create a new roller?
-----------------------------------------------
*(Googlers only)* See `go/pw-config-tasks <http://go/pw-config-tasks>`_. But most
of the time the appropriate action is to file a bug at
`go/pw-infra-bug <http://go/pw-infra-bug>`_.
