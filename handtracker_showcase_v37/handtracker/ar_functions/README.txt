Drop .txt files in this folder and they show up under the ƒ(x) button in AR mode.
(Re-scanned every time you open the Functions row - no restart needed.)

FORMAT: one line, a function of x and y (u/v also work, same thing).
Everything up to and including the last "=" is ignored, so any of these
are fine:

    z = sin(sqrt(x^2 + y^2) * 6)
    f(x, y) = sin(sqrt(x^2 + y^2) * 6)
    sin(sqrt(x^2 + y^2) * 6)

"^" means power (x^2, not XOR) - the usual way people write it by hand.
The result is automatically rescaled to fit the hologram, so it doesn't
matter if your function's output is tiny or huge - only its SHAPE
matters.

ALLOWED: + - * / ^ %, comparisons (<, >, ==, ...), and/or, an if/else
ternary, and: sin cos tan asin acos atan atan2 sinh cosh tanh exp log
log2 log10 sqrt abs sign hypot min max floor ceil round clip, and the
constants pi, e, tau. Nothing else is allowed - no other names, no
imports, no attribute access - an expression using anything else is
refused with an explanation rather than silently doing something
unexpected.

A value that would be undefined at some point on the grid (1/x at x=0,
log of a negative number, etc.) just shows as 0 there rather than
crashing or leaving a hole in the surface.

If a .txt file doesn't parse, it shows up in the Functions row as a red
tile with a "!" instead of being skipped silently - fix the file and
reopen the row (close/open the ƒ(x) button) to pick it up.

One real limitation worth knowing: this is a HEIGHT FIELD (one z per
x,y point), so it can only ever make dome/bump/ripple/saddle shapes -
never a shape with a silhouette that curves back on itself, like a
heart's top notch or a star's points meeting. That needs an "implicit
curve" (a 2D region defined by an inequality, like the classic heart
curve x^2 + (1.3y - |x|^0.67)^2 < 1), not a height function. See
heart_mound.txt for the honest equivalent: a smooth dome over that
curve's interior, not a heart-shaped silhouette.
