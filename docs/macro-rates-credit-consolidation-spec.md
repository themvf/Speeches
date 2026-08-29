# Making the Macro page tell one story about rates and credit

Written 2026-08-28. **Proposal only — no code changes.** Plain language throughout;
there is a short glossary at the end, and an appendix with the exact file and
series names for whoever builds it.

---

## 1. The one-paragraph version

The Macro page has grown three separate panels that all talk about the same
thing: what the government pays to borrow, and what everyone else pays on top of
that. They were built at different times by different people, they fetch the
same numbers from the same place separately, and they are allowed to disagree
with each other on screen. This document proposes merging them into the single
workspace the project already decided it wanted, and then adding the two things
none of them currently answers.

---

## 2. A little background, so the rest reads easily

Four ideas are enough to follow everything below.

**The government's borrowing cost.** The US Treasury borrows money for different
lengths of time — three months, two years, ten years, thirty years. Each length
has its own interest rate. Together those rates are called *the curve*. The
shape of the curve is meaningful: normally you get paid more for lending for
longer, and when that stops being true it usually means something.

**The spread.** Nobody else borrows as cheaply as the government. A homeowner
with a mortgage, or a company issuing a bond, pays the government's rate *plus
something extra*. That extra is the **spread**. So:

> what a borrower pays  =  the government's rate  +  the spread

This is the central idea. When mortgage rates rise, it is either because the
government's borrowing cost rose, or because the spread widened — and those two
have completely different causes. Separating them is the entire point of this
part of the page.

**Basis points.** One basis point is one hundredth of a percentage point. Rate
moves are small and get quoted this way: a rise from 6.42% to 6.66% is 24 basis
points. It is jargon, but it is unavoidable jargon.

**A cached copy.** Fetching live data on every page view would be slow and
wasteful, so the app saves a copy and reuses it for a set period. How long it
keeps that copy is a choice, and — as section 3 explains — the choice currently
differs between panels that show the same number.

---

## 3. What is on the page today, and what is wrong with it

Three panels, plus a chart, all covering overlapping ground:

| On screen | What it shows | Where its numbers come from |
|---|---|---|
| **Rates & Credit** (top of the tab) | Ten Treasury maturities, real yields, signals | FRED, refreshed every 15 min |
| **Yield Curve** chart (inside Financial Conditions) | Thirteen Treasury maturities plotted | The US Treasury's own file, refreshed hourly |
| **Rate Transmission** (inside Financial Conditions) | Mortgage and corporate split into rate + spread; four curve gaps | FRED, refreshed hourly |
| Individual indicator cards | Credit spread, real yield, mortgage rate, and others, one per card | FRED, refreshed every 15 min |

### Problem 1 — the same number, four different ways, allowed to disagree

The ten-year Treasury yield currently reaches that page by four separate routes,
two of which use different upstream sources, and three of which keep their saved
copy for different lengths of time.

The consequence is not theoretical. Because one panel holds its copy for fifteen
minutes and another holds it for an hour, **two panels on the same screen can be
showing the ten-year yield as it stood an hour apart.** A reader comparing them
sees a discrepancy that is real on screen and meaningless in the world.

This project has been bitten by exactly this before and fixed it once: the yield
curve chart used to calculate its own version of the ten-year-minus-two-year gap
while a card beside it showed the official figure, and the two drifted apart.
The fix then was to make one source own the number. That fix was applied to one
number; the same problem has since regrown around several others.

### Problem 2 — the page repeats itself

The Treasury curve is drawn or listed three times in three different formats
within one scroll. None of the three is wrong. But a reader has to work out that
they are the same thing before they can ignore two of them, and the page's job
is to save that effort, not create it.

### Problem 3 — a lot of wasted fetching

Loading the Macro tab currently triggers **85 separate requests** to the data
provider. Eight of those series are fetched more than once because different
panels each ask for them independently.

The most striking case: the Rate Transmission panel fetches seven series, and
**all seven are already being fetched by one of the other two panels.** It
requests nothing that the page does not already have in hand.

This is not primarily a cost problem — the data is free. It is the mechanism
behind Problem 1. Every duplicated fetch is another copy of a number that can
fall out of step with its twin.

### Problem 4 — warnings that do not look like warnings

Throughout the app, "something is wrong here" notices are styled amber. Because
of a one-line mistake in the styling configuration, **none of those amber styles
have ever actually applied** — the affected text renders in the ordinary body
colour. Eighteen files are affected.

So the places designed to catch the eye do not catch the eye. A reader has no
visual cue separating "this figure is unavailable" from ordinary content. This
was found while checking something unrelated and is not yet fixed, because the
one-line fix switches amber on across all eighteen files at once and that is a
deliberate visual change somebody should choose to make.

### Problem 5 — the actual question is still only half answered

The original request behind this work was to track *how rate spreads affect
mortgage rates and corporate bonds, and vice versa*. What exists today answers
the first half of that: it can show how much of a move came from the government
rate and how much from the spread.

It cannot yet answer either of:

- **Does a change in the government's borrowing cost actually reach borrowers?**
  When the ten-year yield rises by 25 basis points, do mortgage rates rise by 25,
  or by 15, with the spread quietly absorbing the rest?
- **Which moves first?** Do corporate borrowing costs move ahead of government
  ones, or after? That is the "and vice versa" half, and nothing on the page
  touches it.

---

## 4. What good looks like

The project already has a written strategy for this area, and it is clear: **one
Rates & Credit workspace, which interprets relationships rather than displaying
a larger pile of independent cards.** It also states that every conclusion must
show the observations behind it.

Measured against that, the current page is a step backwards — three panels of
independent cards where the strategy asked for one interpreted workspace. So
this proposal is not a new direction. It is finishing the one already chosen.

The target, in plain terms:

> One section on the Macro page about the cost of borrowing. It shows the
> government's curve once. Beneath it, what each kind of borrower actually pays,
> split into the government's part and the spread. Then how much of the recent
> move came from each part, whether those moves are reaching borrowers, and who
> is moving first. Every number traceable to a source and a date, and every
> number on the page consistent with every other number on the page.

---

## 5. The plan

Five stages. Each is independently shippable and each leaves the page working.
Stages 1–3 are cleanup that makes the page trustworthy; stages 4–5 add the
missing answers.

### Stage 1 — One source per number

**Problem addressed:** 1 and 3.

Fetch each series once, in one place, and let all the panels read from that.
Decide deliberately, and write down, which upstream source owns each figure —
the data provider or the Treasury's own file — rather than letting it depend on
which panel a reader happens to look at.

Give the whole section one refresh interval, so nothing on screen can be an hour
older than the thing beside it.

**Done when:** every number appears once, from one source; the request count
drops from 85 toward roughly 30; and a stated test confirms that the same figure
shown in two places is byte-identical.

**Risk:** low. No visible change other than internal consistency.

### Stage 2 — Say each thing once

**Problem addressed:** 2.

Merge the three overlapping panels into one section, following the structure the
strategy document already lays out. Draw the curve once. Keep the individual
cards only where they say something the workspace does not.

**Done when:** the Treasury curve appears once on the tab, and a reader can find
any single figure in exactly one place.

**Risk:** medium — this is the visible change, and it means retiring panels that
currently work. Worth a look at the layout before building.

### Stage 3 — Make failures visible

**Problem addressed:** 4.

Fix the styling configuration so warning styles work, and check the eighteen
affected files to make sure the now-visible amber looks intentional rather than
merely loud.

**Done when:** an unavailable figure is visibly distinct from an available one,
and a reviewer has signed off on the appearance of all eighteen.

**Risk:** low technically, but it changes appearance in eighteen places at once,
so it needs an explicit yes.

### Stage 4 — Does the move reach borrowers?

**Problem addressed:** 5, first half.

Add a measure of **pass-through**: over the last year, when the government's
borrowing cost moved, how much of that move showed up in what mortgage borrowers
and companies actually paid.

Present it plainly — "over the past year, about 80% of each Treasury move has
reached mortgage borrowers" — alongside how many observations that rests on and
how reliable the relationship is. Suppress it entirely when there is not enough
history, rather than printing a confident-looking number built on very little.

**Done when:** the figure is shown with its sample size and a plain statement of
how much of the variation it explains, and it disappears rather than degrades
when data is thin.

**Risk:** medium. Unlike everything above, this is an *estimate* rather than
arithmetic, so it can be wrong in ways that still look plausible. See section 7.

### Stage 5 — Who moves first?

**Problem addressed:** 5, second half — the "vice versa".

Add a timing comparison: do corporate borrowing costs tend to move before, with,
or after government ones?

There is a specific trap here that must be designed around, because getting it
wrong produces a confident and completely meaningless result. The corporate
spread is *defined* as the corporate rate minus the government rate. Comparing
that spread against the government rate therefore always shows them moving in
opposite directions — not because anything interesting is happening, but because
one is calculated by subtracting the other. **The comparison has to be made
between the two borrowing rates themselves, never between a spread and the rate
it was derived from.**

**Done when:** the panel names a leader only when the relationship is strong
enough to be worth naming, says "no clear lead" otherwise, and states in plain
words that this is a timing observation and not a claim about cause.

**Risk:** highest of the five. Ship last, or not at all if stage 4 proves
awkward.

---

## 6. Deliberately not doing

- **High-yield ("junk") bonds.** The standard data for this is licensed and
  cannot be published on a public page. The code already fetches it and
  correctly refuses to display it. This section therefore covers safer corporate
  borrowers only, and should say so rather than leaving readers to wonder.
- **Predicting anything.** Everything here describes what has happened. No
  forecasts, no recommendations. The project already enforces this with an
  automated check on the wording, and that check should extend to any new text.
- **Rebuilding what works.** The individual indicator cards, the release
  calendar, and the conditions summary are not in scope except where they
  duplicate the new section.

---

## 7. How we will know it is right

Two kinds of checking, because two kinds of thing can go wrong.

**Arithmetic** — stages 1 to 3. The split of a borrowing rate into government
part plus spread is definitional; there is nothing to estimate. It should be
verified by confirming the parts add up to the whole *as displayed*, not merely
before rounding. This has already caused one real defect: the panel carried a
caption promising the components summed to the total while rounding each part
separately, so the visible numbers sometimes did not add up.

**Estimates** — stages 4 and 5. These can be confidently wrong. Three rules:

1. **Never guess a date.** The mortgage figure is published weekly; Treasury
   yields daily. They must be matched by *date*, not by position in a list.
   Matching by position produced a spread that was wrong by up to 54 basis
   points in real data — and, crucially, did not look wrong. This is the single
   easiest mistake to make here.
2. **Always say how much history a claim rests on.** "The widest spread on
   record" means something very different over twenty years than over eighteen
   months.
3. **Show nothing rather than something thin.** Below a stated minimum number of
   observations, the estimate does not appear at all.

---

## 8. Questions for you

1. **Layout.** Should the merged workspace sit at the top of the Macro tab where
   Rates & Credit is now, or inside the collapsible Financial Conditions group
   where the other two live? This changes how prominent it is.
2. **Amber.** Do you want the warning styles switched on across all eighteen
   files, or the token renamed so only new work uses it and existing pages are
   untouched?
3. **Stopping point.** Stages 1–3 make the page consistent and honest. Stages
   4–5 make it answer more. If you only want the first three, that is a coherent
   place to stop and worth saying now.
4. **The curve's owner.** Two sources are in use for the same Treasury curve —
   the data provider and the Treasury's own file. They differ slightly in timing
   and in which maturities they carry. Picking one is a small decision with a
   long tail, and I would rather you made it than inherited it.

---

## Glossary

- **Basis point** — one hundredth of a percentage point. 100 basis points = 1%.
- **The curve** — the set of interest rates the US government pays across
  borrowing lengths from three months to thirty years.
- **Spread** — the extra a borrower pays above the government's rate for the
  same length of time.
- **Pass-through** — how much of a change in the government's rate actually
  reaches a borrower.
- **Investment grade / high yield** — safer corporate borrowers versus riskier
  ones. Only the safer group is covered here, for licensing reasons.
- **Real yield** — an interest rate after subtracting expected inflation.

---

## Appendix — for whoever builds it

Current surfaces: `components/market/rates-credit-section.tsx`,
`components/market/yield-curve.tsx`, `components/market/rate-transmission.tsx`,
all rendered from `components/market/macro-tab.tsx`.

Routes and cache lifetimes: `/api/market/rates-credit` (22 series, 900s),
`/api/market/rate-transmission` (7 series, 3600s), `/api/market/macro`
(28 indicators, 2 requests each, 900s), `/api/market/bonds` (Treasury XML, 3600s).

Series fetched more than once per refresh cycle: `DGS3MO`, `DGS2`, `DGS10`,
`DGS30` (rates-credit and rate-transmission); `DFF`, `BAA10Y`, `MORTGAGE30US`
(macro and rate-transmission); `DFII10` (macro and rates-credit).

Governing documents: `docs/rates-credit-intelligence-strategy.md` (the
architecture this returns to) and `apps/web/docs/rate-transmission-spec.md`
(stages 4 and 5 in technical form, as its phases 2 and 3, including the
derived-spread trap and the date-matching rule).

Licensing: `BAA10Y` is *citation required* and usable with the page's existing
attribution. The ICE BofA family is *pre-approval required* and must stay gated.
Both facts are recorded in `CLAUDE.md`; the second was briefly recorded wrongly
and cost the panel its corporate half.

The styling defect: `tailwind.config.ts` sets `colors.amber` to a single string,
which replaces Tailwind's amber scale instead of extending it, so every
`amber-<number>` utility matches no rule.
