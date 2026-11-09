import assert from "node:assert/strict";
import test from "node:test";
import { normalizeYahooOhlcv } from "./yahoo.ts";

test("normalizes complete OHLCV candles and rejects incomplete price rows", () => {
  const candles = normalizeYahooOhlcv({
    timestamp: [100, 200, 300],
    indicators: { quote: [{
      open: [10, null, 12], high: [12, 13, 9], low: [9, 10, 10], close: [11, 12, 10.5], volume: [1000, 2000, null],
    }] },
  });
  assert.deepEqual(candles, [{ t: 100, open: 10, high: 12, low: 9, close: 11, volume: 1000 }]);
});
