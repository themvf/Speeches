import assert from "node:assert/strict";
import test from "node:test";
import config from "../tailwind.config.ts";

/**
 * Guards the defect that made every warning in this app invisible.
 *
 * Setting a colour to a bare string REPLACES Tailwind's scale for that name
 * instead of extending it, so `amber-300` and friends match no rule and render
 * in the inherited ink colour. Eighteen files styled warnings that way and none
 * of them ever looked like warnings. The failure is silent - no build error, no
 * console warning - so it needs a test rather than vigilance.
 */
test("palette colours that are used with numeric shades keep their scale", () => {
  const colors = (config.theme?.extend?.colors ?? {}) as Record<string, unknown>;
  for (const [name, value] of Object.entries(colors)) {
    if (typeof value !== "object" || value === null) continue;
    const shades = Object.keys(value as Record<string, unknown>).filter((key) => /^\d+$/.test(key));
    assert.ok(shades.length >= 5, `${name} declares an object but almost no numeric shades`);
  }
  assert.equal(typeof colors.amber, "object",
    "amber is used as amber-300/amber-500 in 18 files; a bare string here silently kills all of them");
  const amber = colors.amber as Record<string, unknown>;
  for (const shade of ["200", "300", "400", "500"]) {
    assert.ok(amber[shade], `amber-${shade} is used in the app and must resolve`);
  }
  assert.equal(amber.DEFAULT, "var(--amber)", "the brand token stays reachable as bare `amber`");
});
